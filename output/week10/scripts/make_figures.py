"""Draw the figures of output/week10/homework1_summary.md from our runs (seeds 80-82).

    python output/week10/scripts/make_figures.py

Writes, each into its part's folder under output/week10/:
    part2_lqr/part2_lqr.png                        the gain K over training, the hill J(K), the variance of the estimators
    part3_arm/part3_arm.png                        both algorithms on the arm: learning curve and final pose per seed
    part4_variation/part4_arm.png                  our variation plr3e4 against base
    part5_own_experiment/part5_arm.png             our own experiment blr3e3 against base
    part5_own_experiment/part5_baseline.png        baseline fit, gradient noise and final return, blr3e3 against base

part5_baseline reads part5_own_experiment/baseline_measurements.csv: scripts/baseline_quality.py at the
checkpoints of the blr3e3 runs, and of a longer copy of the base runs that repeated them exactly and is
no longer kept.
"""
import ast
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle

import envs  # noqa: F401  (registers the project environments)
import envs.planar_arm as arm
import envs.scalar_lqr as lqr
from harness.registry import build_agent
from harness.results import load_run

HERE = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
PART2, PART3, PART4, PART5 = (HERE / name for name in ("part2_lqr", "part3_arm", "part4_variation",
                                                    "part5_own_experiment"))
SEEDS = (80, 81, 82)
SEED_COLOUR = {80: "tab:blue", 81: "tab:orange", 82: "tab:green"}
# The three places an arm run ends in, as evaluation returns (see the summary, part 3).
LEVELS = ((-135, "ends inside the obstacle"), (-172, "stops short of the target"), (-330, "stays hanging"))
torch.set_num_threads(1)


def run_dir(stem, seed):
    return RUNS / f"PlanarArm-v0_{stem}_seed{seed}"


def load_agent(folder):
    cfg = load_run(str(folder))["config"]
    overrides = {k: ast.literal_eval(v) if v[:1] not in "\"'" else v.strip("'\"")
                 for k, v in cfg["hyperparameters"].items()}
    env = gym.make(cfg["env"])
    agent, _ = build_agent(cfg["algo"], env, overrides)
    env.close()
    agent.load(Path(folder) / "final.pt")
    return agent


def evaluation_episode(agent, seed=1000):
    """One episode as the evaluation plays it: from hanging, mean action, no noise."""
    env = gym.make("PlanarArm-v0")
    obs, _ = env.reset(seed=seed)
    tips, states, done = [], [obs], False
    while not done:
        with torch.no_grad():
            u = agent.policy(torch.as_tensor(obs, dtype=torch.float32)).mean.numpy()
        obs, _, term, trunc, info = env.step(u)
        tips.append(info["fingertip"]); states.append(obs)
        done = term or trunc
    env.close()
    return np.array(tips), np.array(states)


def ending(cost, distance):
    if cost > 100:
        return "ends inside the obstacle"
    if distance > 1.4:
        return "stays hanging"
    if distance < 0.4:
        return "gets close to the target"
    return "stops short of the target"


def draw_links(a, q, **kw):
    elbow = arm.LINK_LENGTH * np.array([np.sin(q[0]), -np.cos(q[0])])
    tip = arm.fingertip_position(q)
    a.plot([0, elbow[0], tip[0]], [0, elbow[1], tip[1]], "-o", **kw)


def level_lines(a, x_text):
    for y, text in LEVELS:
        a.axhline(y, color="gray", ls=":", lw=1)
        a.text(x_text, y + 4, text, color="gray", fontsize=7.5, ha="right")


def arm_rows(rows, filename, title):
    """One row per experiment: evaluation return per seed over training, then the final pose per seed."""
    fig, ax = plt.subplots(len(rows), 4, figsize=(17, 3.9 * len(rows)), squeeze=False,
                           gridspec_kw={"width_ratios": [1.6, 1, 1, 1]})
    for i, (stem, label) in enumerate(rows):
        a = ax[i, 0]
        for s in SEEDS:
            r = load_run(str(run_dir(stem, s)))
            a.plot(r["eval_steps"], r["eval_return"], color=SEED_COLOUR[s], lw=2, label=f"seed {s}")
        level_lines(a, a.get_xlim()[1])
        a.set_ylim(-365, -110)
        a.set_title(label, fontsize=10.5, loc="left", fontweight="bold")
        a.set_ylabel("evaluation return")
        if i == len(rows) - 1:
            a.set_xlabel("environment steps of training")
        a.legend(fontsize=8, loc="lower right", ncol=3)
        for j, s in enumerate(SEEDS):
            a = ax[i, j + 1]
            r = load_run(str(run_dir(stem, s)))
            ret, cost, dist = r["eval_return"][-1], r["eval_cost"][-1], r["eval_distance"][-1]
            tips, states = evaluation_episode(load_agent(run_dir(stem, s)))
            a.add_patch(Circle(arm.OBSTACLE_CENTRE, arm.OBSTACLE_RADIUS, color="red", alpha=0.18))
            a.add_patch(Circle((0, 0), 2 * arm.LINK_LENGTH, fill=False, ls=":", color="lightgray"))
            draw_links(a, states[0][:2], color="lightgray", lw=3, ms=4)
            a.plot(tips[:, 0], tips[:, 1], color=SEED_COLOUR[s], lw=1.5)
            draw_links(a, states[-1][:2], color="black", lw=3.5, ms=5)
            a.plot(*arm.TARGET, "*", color="gold", ms=17, mec="black")
            a.set_xlim(-1.1, 1.1); a.set_ylim(-1.1, 1.1); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
            a.set_title(f"seed {s}: {ending(cost, dist)}\nreturn {ret:.0f}, {cost:.0f} steps in obstacle, "
                        f"ends {dist:.2f} m away", fontsize=9, color=SEED_COLOUR[s])
    fig.suptitle(title + "\nLeft: evaluation return during training. Right: the evaluation episode of the final policy "
                 "(grey = start, coloured line = fingertip path, black = final pose, star = target, red = obstacle).",
                 fontsize=11.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / (3.9 * len(rows))))
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print("saved", filename)


def part2_lqr():
    _, k_star = lqr.riccati_solution()
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.4))
    finals = {}
    a = ax[0]
    for algo, colour in (("REINFORCE", "tab:orange"), ("REINFORCE-BASELINE", "tab:blue")):
        for s in SEEDS:
            folder = RUNS / f"ScalarLQR-v0_{algo}_base_seed{s}"
            steps, gains = [0], [0.6]
            for ck in sorted(folder.glob("step_*.pt")):
                steps.append(int(ck.stem.split("_")[1]))
                gains.append(-torch.load(ck, map_location="cpu")["policy"]["mean_net.0.weight"].item())
            a.plot(steps, gains, color=colour, lw=2, label=("with baseline" if "BASE" in algo else "without baseline") if s == 80 else None)
            finals[(algo, s)] = gains[-1]
    a.axhline(k_star, color="black", ls="--")
    a.text(2000, k_star - 0.03, f"K* = {k_star:.3f}, the Riccati answer", fontsize=9)
    a.set_title("a. The gain K during training (3 seeds each)\nEvery run starts at K = 0.6 and climbs towards K*")
    a.set_xlabel("environment steps of training"); a.set_ylabel("gain K"); a.legend(loc="lower right")

    a = ax[1]
    ks = np.linspace(0.45, 1.6, 400)
    horizon = np.arange(lqr.EPISODE_STEPS)
    J = -(lqr.Q + lqr.R * ks**2) * ((lqr.GAMMA * (lqr.A - lqr.B * ks)[:, None] ** 2) ** horizon).sum(axis=1)
    a.plot(ks, J, color="black", lw=2)
    a.axvline(k_star, color="black", ls="--", lw=1)
    hill = lambda k: np.interp(k, ks, J)
    a.plot(0.6, hill(0.6), "s", color="gray", ms=9, label="start, K = 0.6")
    for (algo, s), k in finals.items():
        colour = "tab:blue" if "BASE" in algo else "tab:orange"
        a.plot(k, hill(k), "o", color=colour, ms=9, mec="black",
               label=("final K, with baseline" if "BASE" in algo else "final K, without baseline") if s == 80 else None)
    a.annotate("flat near the top:\na small error in K\ncosts almost nothing", (1.0, hill(1.0)), (0.88, -1.45),
               arrowprops=dict(arrowstyle="->"), fontsize=9)
    a.set_title("b. The hill the runs climb: return J for every gain K\n(the algorithm never sees this curve)")
    a.set_xlabel("gain K"); a.set_ylabel("expected return J(K)"); a.legend(loc="lower center", fontsize=8.5)

    a = ax[2]
    d = np.load(PART2 / "gradient_variance_ScalarLQR-v0.npz")
    values = [float(d["full_return"]), float(d["reward_to_go"]), float(d["baseline"])]
    names = ["full return\n$G_0$ on every step", "reward-to-go\n$G_t$", "with baseline\n$G_t - b(x_t)$"]
    bars = a.bar(names, values, color=["tab:gray", "tab:orange", "tab:blue"])
    for bar, v in zip(bars, values):
        a.text(bar.get_x() + bar.get_width() / 2, v + 5, f"{v:.0f}", ha="center", fontsize=12, fontweight="bold")
    a.set_title("c. How noisy is the gradient from ONE episode?\nRelative variance: about how many episodes you must average\nbefore the noise is as small as the gradient itself")
    a.set_ylabel("relative variance")
    fig.suptitle("Part 2. Scalar LQR: REINFORCE with and without a baseline, against the Riccati answer", fontsize=13)
    fig.tight_layout()
    fig.savefig(PART2 / "part2_lqr.png", dpi=150)
    plt.close(fig)
    print("saved", PART2 / "part2_lqr.png")


def part5_baseline():
    """Our own experiment: did the faster baseline fit better, cut the noise, and help the arm?"""
    with open(PART5 / "baseline_measurements.csv", encoding="utf-8") as f:
        m = [{**r, "seed": int(r["seed"]), "step": int(r["step"]), "r2": float(r["r2_baseline"]),
              "noise": float(r["noise_left_percent"])} for r in csv.DictReader(f)]
    shown = (("base", "REINFORCE-BASELINE_base", "base: baseline_lr = 1e-3", "tab:blue"),
             ("blr3e3", "REINFORCE-BASELINE_blr3e3", "blr3e3: baseline_lr = 3e-3 (3 times faster)", "tab:red"))
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.6))
    for a, key, title, ylabel in (
            (ax[0], "r2", "a. Did the baseline predict the returns better?\n1 = perfect, 0 = no better than one constant number",
             "how well the baseline predicts the return (R²)"),
            (ax[1], "noise", "b. Did the gradient get less noisy?\nNoise left with the baseline, in % of the noise without one",
             "% of the noise without a baseline")):
        for name, _, label, colour in shown:
            rows = [r for r in m if r["experiment"] == name]
            steps = sorted({r["step"] for r in rows})
            a.scatter([r["step"] for r in rows], [r[key] for r in rows], color=colour, s=18, alpha=0.45)
            a.plot(steps, [np.median([r[key] for r in rows if r["step"] == s]) for s in steps], "o-", color=colour,
                   lw=2.8, label=label)
        a.set_title(title); a.set_ylabel(ylabel); a.set_xticks([100_000, 200_000, 300_000])
        a.set_xlabel("environment steps of training (small dots: the 3 seeds, line: their median)")
    ax[0].axhline(0, color="gray", ls=":", lw=1); ax[0].legend(fontsize=9, loc="lower right")
    ax[1].set_ylim(0, None)

    a = ax[2]
    for i, (name, stem, label, colour) in enumerate(shown):
        finals = [load_run(str(run_dir(stem, s)))["eval_return"][-1] for s in SEEDS]
        for s, y in zip(SEEDS, finals):
            a.scatter(i + 0.09 * (s - 81), y, color=SEED_COLOUR[s], s=110, zorder=3, edgecolor="black", label=f"seed {s}" if i == 0 else None)
        a.plot([i - 0.3, i + 0.3], [np.mean(finals)] * 2, color="black", lw=3)
        a.text(i + 0.33, np.mean(finals), f"mean {np.mean(finals):.0f}", va="center", fontsize=10, fontweight="bold")
    level_lines(a, 1.75)
    a.set_xlim(-0.6, 1.8); a.set_ylim(-365, -110)
    a.set_xticks([0, 1], ["base", "blr3e3\nfaster baseline"])
    a.set_title("c. Did the arm end in a better place?\nFinal evaluation return per seed")
    a.set_ylabel("evaluation return"); a.legend(fontsize=9, loc="center left")
    fig.suptitle("Part 5. The faster baseline predicts better (a) and removes more noise (b). "
                 "The arm ends in the same kinds of places (c)", fontsize=13)
    fig.tight_layout()
    fig.savefig(PART5 / "part5_baseline.png", dpi=150)
    plt.close(fig)
    print("saved", PART5 / "part5_baseline.png")


if __name__ == "__main__":
    part2_lqr()
    arm_rows([("REINFORCE_base", "REINFORCE (no baseline)"), ("REINFORCE-BASELINE_base", "REINFORCE-BASELINE")],
             PART3 / "part3_arm.png", "Part 3. The arm, with and without a baseline (seeds 80-82)")
    arm_rows([("REINFORCE-BASELINE_base", "REINFORCE-BASELINE, policy_lr = 1e-3 (base)"),
              ("REINFORCE-BASELINE_plr3e4", "plr3e4: policy_lr = 3e-4 (our variation)")],
             PART4 / "part4_arm.png", "Part 4. Our assigned variation: a policy step 3 times smaller")
    arm_rows([("REINFORCE-BASELINE_base", "base: baseline_lr = 1e-3"),
              ("REINFORCE-BASELINE_blr3e3", "blr3e3: baseline_lr = 3e-3, the baseline learns 3 times faster")],
             PART5 / "part5_arm.png", "Part 5. Our own experiment: a faster baseline")
    part5_baseline()
