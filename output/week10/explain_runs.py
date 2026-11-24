"""Figures that show what a Week 9 training run does, drawn from the trained runs in runs/.

    python output/week10/explain_runs.py

Writes fig1_lqr.png, fig2_arm_episode.png, fig3_arm_policy_value.png and fig4_arm_learning.png
next to this file. Single-run pictures use the seed-80 runs; learning curves use every seed.
"""

import ast
import sys
from pathlib import Path

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import envs  # noqa: F401,E402  (registers the environments)
from envs import planar_arm as arm  # noqa: E402
from envs import scalar_lqr as lqr  # noqa: E402
from harness.registry import build_agent  # noqa: E402
from harness.results import load_run  # noqa: E402
from models.reinforce import reward_to_go  # noqa: E402

OUT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
SEEDS = (80, 81, 82)
SHOW = 80
COLOR = {"REINFORCE": "tab:orange", "REINFORCE-BASELINE": "tab:blue", "plr3e4": "tab:green"}
DT = arm.SUBSTEP * arm.FRAME_SKIP          # 0.02 s per environment step
torch.set_num_threads(1)


def load(run_dir, checkpoint="final.pt"):
    """The trained agent of a run folder, rebuilt the way watch.py does it."""
    run = load_run(str(run_dir))
    cfg = run["config"]
    overrides = {k: ast.literal_eval(v) if v[:1] not in "\"'" else v.strip("'\"")
                 for k, v in cfg["hyperparameters"].items()}
    env = gym.make(cfg["env"])
    agent, _ = build_agent(cfg["algo"], env, overrides)
    env.close()
    agent.load(Path(run_dir) / checkpoint)
    return agent, run


def gain_and_std(agent):
    """K and sigma of a linear Gaussian policy u = -K x + bias + sigma * noise."""
    return -agent.policy.mean_net[0].weight.item(), agent.policy.log_std.exp().item()


# --------------------------------------------------------------------------------------------
# Figure 1: scalar LQR, where the answer is known
# --------------------------------------------------------------------------------------------
def figure_lqr():
    p_star, k_star = lqr.riccati_solution()
    ks = np.linspace(0.25, 1.75, 400)
    c = lqr.A - lqr.B * ks
    horizon = np.arange(lqr.EPISODE_STEPS)
    J = -(lqr.Q + lqr.R * ks**2) * ((lqr.GAMMA * c[:, None] ** 2) ** horizon).sum(axis=1)

    fig, ax = plt.subplots(2, 2, figsize=(15, 10.5))
    a = ax[0, 0]
    a.plot(ks, J, color="black", lw=1.5, label="J(K): mean discounted return of u = −Kx")
    a.axvline(k_star, color="black", ls="--", lw=1)
    a.text(k_star + 0.02, -1.6, f"K* = {k_star:.3f}\n(Riccati)", va="center")

    histories = {}
    for algo in ("REINFORCE", "REINFORCE-BASELINE"):
        for seed in SEEDS:
            run_dir = RUNS / f"ScalarLQR-v0_{algo}_base_seed{seed}"
            steps, gains, stds = [0], [0.6], [np.exp(-0.5)]
            for ck in sorted(run_dir.glob("step_*.pt")):
                agent, _ = load(run_dir, ck.name)
                k, s = gain_and_std(agent)
                steps.append(int(ck.stem.split("_")[1])); gains.append(k); stds.append(s)
            histories[(algo, seed)] = (np.array(steps), np.array(gains), np.array(stds))
            g = np.array(gains)
            jg = -(lqr.Q + lqr.R * g**2) * ((lqr.GAMMA * (lqr.A - lqr.B * g)[:, None] ** 2) ** horizon).sum(axis=1)
            a.plot(g, jg, "-o", color=COLOR[algo], ms=3, lw=0.8, alpha=0.7,
                   label=f"{algo}, K every 10k steps" if seed == SEEDS[0] else None)
            a.plot(g[-1], jg[-1], "o", color=COLOR[algo], ms=9, mec="black")
    a.plot(0.6, J[np.argmin(abs(ks - 0.6))], "s", color="gray", ms=9, mec="black", label="start, K = 0.6")
    a.set_xlabel("gain K"); a.set_ylabel("J(K)")
    a.set_title("1a. The objective the policy gradient climbs, and where each run went")
    a.legend(fontsize=9, loc="lower right")

    a = ax[0, 1]
    xs = np.linspace(-3, 3, 50)
    a.plot(xs, -0.6 * xs, color="gray", lw=2, label="start: K = 0.6")
    for algo in ("REINFORCE", "REINFORCE-BASELINE"):
        for seed in SEEDS:
            k = histories[(algo, seed)][1][-1]
            a.plot(xs, -k * xs, color=COLOR[algo], lw=1.2, alpha=0.8,
                   label=f"{algo} final (3 seeds)" if seed == SEEDS[0] else None)
    _, k_show, s_show = (h[-1] for h in histories[("REINFORCE-BASELINE", SHOW)])
    a.fill_between(xs, -k_show * xs - s_show, -k_show * xs + s_show, color="tab:blue", alpha=0.12,
                   label=f"± σ of REINFORCE-BASELINE seed {SHOW}: the exploration")
    a.plot(xs, -k_star * xs, "k--", lw=2, label="optimal: u = −K* x")
    a.set_xlabel("state x"); a.set_ylabel("action u (mean)")
    a.set_title("1b. The policy itself: a line through the origin, u = −Kx, plus noise")
    a.legend(fontsize=9)

    a = ax[1, 0]
    for (algo, seed), (steps, gains, stds) in histories.items():
        a.plot(steps / 1000, stds, color=COLOR[algo], lw=1.5, alpha=0.8,
               label=algo if seed == SEEDS[0] else None)
    a.set_xlabel("training steps (thousands)"); a.set_ylabel("σ = exp(log_std)")
    a.set_title("1c. Exploration noise σ is learned too: it is a policy parameter")
    a.legend(fontsize=9)

    a = ax[1, 1]
    agent, _ = load(RUNS / f"ScalarLQR-v0_REINFORCE-BASELINE_base_seed{SHOW}")
    env = gym.make("ScalarLQR-v0")
    torch.manual_seed(0)
    cloud_x, cloud_g = [], []
    for ep in range(40):
        obs, _ = env.reset(seed=2000 + ep)
        xs_ep, rs, done = [], [], False
        while not done:
            xs_ep.append(obs[0])
            obs, r, term, trunc, _ = env.step(agent.select_action(obs))
            rs.append(r); done = term or trunc
        cloud_x += xs_ep; cloud_g += list(reward_to_go(rs, agent.gamma))
    env.close()
    a.scatter(cloud_x, cloud_g, s=4, alpha=0.25, color="black", label="observed G_t (40 episodes)")
    grid = np.linspace(-3, 3, 200)
    with torch.no_grad():
        b = agent.baseline(torch.as_tensor(grid, dtype=torch.float32)[:, None]).squeeze(-1).numpy()
    k, _ = gain_and_std(agent)
    p_k = (lqr.Q + lqr.R * k**2) / (1 - lqr.GAMMA * (lqr.A - lqr.B * k) ** 2)
    a.plot(grid, b, color="tab:blue", lw=2.5, label="learned baseline b_w(x)")
    a.plot(grid, -p_star * grid**2, "k--", lw=1.5, label="optimal value, −p* x²  (Riccati)")
    a.plot(grid, -p_k * grid**2, color="tab:purple", lw=2, ls=":",
           label=f"value of this policy without noise, −p_K x²  (K = {k:.2f})")
    a.set_ylim(min(cloud_g) * 1.1, 1); a.set_xlim(-3, 3)
    a.set_xlabel("state x"); a.set_ylabel("return from x")
    a.set_title(f"1d. The value: the baseline learns E[G | x] of the current policy (seed {SHOW})")
    a.legend(fontsize=9, loc="lower center")

    fig.suptitle("Scalar LQR:  x_{k+1} = 1.1 x_k + u_k,   r = −(x² + 0.1u²),   γ = 0.98,   linear Gaussian policy", fontsize=13)
    fig.tight_layout(); fig.savefig(OUT / "fig1_lqr.png", dpi=105); plt.close(fig)


# --------------------------------------------------------------------------------------------
# The arm: one episode with the trained REINFORCE-BASELINE policy
# --------------------------------------------------------------------------------------------
def arm_episode(agent, stochastic, seed=1000):
    env = gym.make("PlanarArm-v0")                  # hanging start, as in evaluation
    obs, _ = env.reset(seed=seed)
    rec = {k: [] for k in ("x", "u", "mu", "r", "p", "cost", "dist")}
    done = False
    while not done:
        with torch.no_grad():
            dist = agent.policy(torch.as_tensor(obs, dtype=torch.float32))
            mu = dist.mean.numpy()
            u = dist.sample().numpy() if stochastic else mu
        rec["x"].append(obs); rec["u"].append(u); rec["mu"].append(mu)
        obs, r, term, trunc, info = env.step(u)
        rec["r"].append(r); rec["p"].append(info["fingertip"]); rec["cost"].append(info["cost"])
        rec["dist"].append(info["distance"])
        done = term or trunc
    env.close()
    return {k: np.array(v) for k, v in rec.items()}


def draw_arm(a, q, **kw):
    elbow = arm.LINK_LENGTH * np.array([np.sin(q[0]), -np.cos(q[0])])
    tip = arm.fingertip_position(q)
    a.plot([0, elbow[0], tip[0]], [0, elbow[1], tip[1]], "-o", ms=3, **kw)


def figure_arm_episode(agent):
    ep = arm_episode(agent, stochastic=True)
    det = arm_episode(agent, stochastic=False)
    T = len(ep["r"]); t = np.arange(T) * DT
    G = reward_to_go(list(ep["r"]), agent.gamma)
    with torch.no_grad():
        b = agent.baseline(torch.as_tensor(ep["x"], dtype=torch.float32)).squeeze(-1).numpy()
    q_target = arm.target_configuration()

    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    a = ax[0, 0]
    a.add_patch(Circle(arm.OBSTACLE_CENTRE, arm.OBSTACLE_RADIUS, color="red", alpha=0.15, label="obstacle (cost)"))
    a.add_patch(Circle((0, 0), 2 * arm.LINK_LENGTH, fill=False, ls=":", color="gray"))
    for k in (0, 40, 80, 120, 160, T - 1):
        draw_arm(a, ep["x"][k, :2], color="gray", alpha=0.35 + 0.6 * k / T, lw=1.5)
    pts = ep["p"]
    seg = np.stack([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(seg, cmap="viridis", array=t[:-1], lw=2.5)
    a.add_collection(lc)
    fig.colorbar(lc, ax=a, label="time (s)", fraction=0.04)
    a.plot(det["p"][:, 0], det["p"][:, 1], "k--", lw=1, label="without noise (as evaluated)")
    a.plot(*arm.TARGET, "*", color="gold", ms=18, mec="black", label="target")
    a.plot(0, 0, "ks", ms=6)
    a.set_aspect("equal"); a.set_xlim(-1.1, 1.1); a.set_ylim(-1.1, 1.1)
    a.set_xlabel("x (m)"); a.set_ylabel("z (m)")
    a.set_title("2a. Fingertip path of one training-style episode\n(stochastic policy, from hanging; grey: the arm)")
    a.legend(fontsize=8, loc="lower left")

    a = ax[0, 1]
    for j, c in ((0, "tab:red"), (1, "tab:purple")):
        a.plot(t, ep["x"][:, j], color=c, lw=2, label=f"q{j + 1}")
        a.axhline(q_target[j], color=c, ls="--", lw=1, label=f"q{j + 1} at the target")
    a.set_xlabel("time (s)"); a.set_ylabel("angle (rad)")
    a.set_title("2b. State, part 1: joint angles\n(q1 from hanging down, q2 elbow relative to link 1)")
    a.legend(fontsize=8)

    a = ax[0, 2]
    for j, c in ((2, "tab:red"), (3, "tab:purple")):
        a.plot(t, ep["x"][:, j], color=c, lw=1.5, label=f"dq{j - 1}")
    a.set_xlabel("time (s)"); a.set_ylabel("angular velocity (rad/s)")
    a.set_title("2c. State, part 2: joint velocities\n(the swing that builds up energy)")
    a.legend(fontsize=8)

    a = ax[1, 0]
    for j, c in ((0, "tab:red"), (1, "tab:purple")):
        a.plot(t, ep["u"][:, j], color=c, lw=0.6, alpha=0.5, label=f"u{j + 1} sampled")
        a.plot(t, ep["mu"][:, j], color=c, lw=2, label=f"u{j + 1} mean μ_θ(x)")
    a.axhline(1, color="black", ls="--", lw=1); a.axhline(-1, color="black", ls="--", lw=1)
    sig = agent.policy.log_std.exp().detach().numpy()
    a.set_xlabel("time (s)"); a.set_ylabel("action (fraction of torque limit)")
    a.set_title(f"2d. Actions: mean + noise, σ = ({sig[0]:.2f}, {sig[1]:.2f})\nthe arm only gets the part inside ±1")
    a.legend(fontsize=8, ncol=2)

    a = ax[1, 1]
    a.plot(t, ep["r"], color="black", lw=1.5, label="reward r_t = −distance − 0.05‖u‖²")
    a.plot(t, -ep["dist"], color="gray", lw=1, ls="--", label="−distance")
    a.fill_between(t, min(ep["r"]) * 1.05, 0, where=ep["cost"] > 0,
                   color="red", alpha=0.12, step="mid", label="fingertip in obstacle (cost 1)")
    a.set_xlabel("time (s)"); a.set_ylabel("reward")
    a.set_title(f"2e. Reward per step (sum = {ep['r'].sum():.1f});\ncost = {int(ep['cost'].sum())} steps in the obstacle")
    a.legend(fontsize=8, loc="lower right")

    a = ax[1, 2]
    a.plot(t, G, color="black", lw=2, label="G_t, observed return-to-go (γ = 0.99)")
    a.plot(t, b, color="tab:blue", lw=2, label="b_w(x_t), the baseline's guess")
    a.fill_between(t, b, G, where=G >= b, color="tab:green", alpha=0.3, label="δ_t > 0: better than usual → push u_t up")
    a.fill_between(t, b, G, where=G < b, color="tab:red", alpha=0.3, label="δ_t < 0: worse than usual → push u_t down")
    a.set_xlabel("time (s)"); a.set_ylabel("return")
    a.set_title("2f. What the update uses: δ_t = G_t − b_w(x_t)")
    a.legend(fontsize=8, loc="lower right")

    fig.suptitle(f"The arm: anatomy of one episode with the trained REINFORCE-BASELINE policy (seed {SHOW}, 200 steps = 4 s)", fontsize=13)
    fig.tight_layout(); fig.savefig(OUT / "fig2_arm_episode.png", dpi=105); plt.close(fig)
    return det


def figure_arm_maps(agent, det):
    q1 = np.linspace(-4.0, 4.0, 241)
    q2 = np.linspace(arm.JOINT_LOW[1], arm.JOINT_HIGH[1], 121)
    Q1, Q2 = np.meshgrid(q1, q2)
    states = np.stack([Q1.ravel(), Q2.ravel(), np.zeros(Q1.size), np.zeros(Q1.size)], axis=1)
    with torch.no_grad():
        x = torch.as_tensor(states, dtype=torch.float32)
        mu = agent.policy(x).mean.numpy()
        b = agent.baseline(x).squeeze(-1).numpy()
    tips = np.array([arm.fingertip_position(s[:2]) for s in states])
    in_obstacle = (np.linalg.norm(tips - arm.OBSTACLE_CENTRE, axis=1) - arm.OBSTACLE_RADIUS).reshape(Q1.shape)
    grav = np.array([arm.gravity_torque(s[:2]) for s in states]) / arm.TORQUE_LIMITS
    cannot_hold = np.max(np.abs(grav), axis=1).reshape(Q1.shape)
    q_target = arm.target_configuration()

    fig, ax = plt.subplots(1, 3, figsize=(19, 6.2))
    panels = ((np.clip(mu[:, 0], -1, 1), "RdBu_r", (-1, 1), "3a. Policy mean μ_θ: shoulder torque u1"),
              (np.clip(mu[:, 1], -1, 1), "RdBu_r", (-1, 1), "3b. Policy mean μ_θ: elbow torque u2"),
              (b, "viridis", (None, None), "3c. Value: the baseline b_w(x), expected return-to-go"))
    for a, (z, cmap, (lo, hi), title) in zip(ax, panels):
        im = a.pcolormesh(Q1, Q2, z.reshape(Q1.shape), cmap=cmap, vmin=lo, vmax=hi, shading="auto")
        fig.colorbar(im, ax=a, fraction=0.04)
        a.contour(Q1, Q2, in_obstacle, levels=[0], colors="red", linewidths=1.5)
        a.contourf(Q1, Q2, cannot_hold, levels=[1, 1e9], colors="none", hatches=["//"])
        a.contour(Q1, Q2, cannot_hold, levels=[1], colors="white", linewidths=1.2, linestyles="--")
        a.plot(det["x"][:, 0], det["x"][:, 1], color="black", lw=1.5)
        a.plot(0, 0, "o", color="white", mec="black", ms=8)
        a.plot(*q_target, "*", color="gold", ms=16, mec="black")
        a.set_xlabel("q1, shoulder (rad)"); a.set_ylabel("q2, elbow (rad)"); a.set_title(title)
    ax[0].plot([], [], color="red", label="fingertip on the obstacle edge")
    ax[0].fill_between([], [], color="none", hatch="//", edgecolor="black",
                       label="hatched: gravity exceeds the torque limit, cannot hold still")
    ax[0].plot([], [], color="black", label="evaluation path from hanging (○)")
    ax[0].plot([], [], "*", color="gold", mec="black", ms=12, label="target configuration")
    ax[0].legend(fontsize=8, loc="lower left", facecolor="lightgray")
    fig.suptitle(f"The arm: policy and value over joint angles, at zero velocity (REINFORCE-BASELINE seed {SHOW}). "
                 "The real state has 4 numbers; this is a 2-D slice.", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / "fig3_arm_policy_value.png", dpi=105); plt.close(fig)


def figure_arm_learning():
    lqr_ret = load_run(str(RUNS / "PlanarArm-v0_LQR_base_seed0"))["eval_return"][-1]
    groups = (("REINFORCE", "base", "REINFORCE"), ("REINFORCE-BASELINE", "base", "REINFORCE-BASELINE"),
              ("REINFORCE-BASELINE", "plr3e4", "plr3e4"))
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.6), sharey=True)
    for i, (a, (algo, variation, key)) in enumerate(zip(ax, groups)):
        for seed, shade in zip(SEEDS, (1.0, 0.7, 0.45)):
            run = load_run(str(RUNS / f"PlanarArm-v0_{algo}_{variation}_seed{seed}"))
            tr = np.convolve(run["train_return"], np.ones(25) / 25, mode="valid")
            a.plot(run["train_steps"][24:] / 1000, tr, color=COLOR[key], lw=0.8, alpha=0.35 * shade + 0.1)
            a.plot(run["eval_steps"] / 1000, run["eval_return"], "-o", color=COLOR[key], alpha=shade, ms=4,
                   lw=1.8, label=f"seed {seed}: evaluation from hanging, no noise")
        a.plot([], [], color=COLOR[key], lw=0.8, alpha=0.4, label="training episodes (random start, noise), mean of 25")
        a.axhline(lqr_ret, color="black", ls="--", lw=1, label=f"LQR controller ({lqr_ret:.0f})")
        a.set_title(f"4{'abc'[i]}. {algo}" + ("" if variation == "base" else f", {variation}"))
        a.set_xlabel("training steps (thousands)")
        a.legend(fontsize=8, loc="lower right")
    ax[0].set_ylabel("return per episode (200 steps)")
    fig.suptitle("The arm: what learning looks like. Training and evaluation episodes are different experiments.", fontsize=13)
    fig.tight_layout(); fig.savefig(OUT / "fig4_arm_learning.png", dpi=105); plt.close(fig)


if __name__ == "__main__":
    figure_lqr()
    agent, _ = load(RUNS / f"PlanarArm-v0_REINFORCE-BASELINE_base_seed{SHOW}")
    det = figure_arm_episode(agent)
    figure_arm_maps(agent, det)
    figure_arm_learning()
    print("Figures written to", OUT)
