"""Figure 5: what one policy-gradient step does to a Gaussian, and why the baseline lowers the variance.

    python output/week10/explain_gradient.py

Panel c measures single-episode estimates of dJ/dK on ScalarLQR-v0 at the starting policy of part 2
(K = 0.6, sigma = e^-0.5), with the functions of experiments/gradient_variance.py.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import envs  # noqa: F401  (registers ScalarLQR-v0)
from experiments.gradient_variance import collect_episode, episode_gradient, fit_baseline
from models.reinforce import reward_to_go
from utils.networks import GaussianPolicy
from utils.seeding import set_seed

OUT = Path(__file__).resolve().parent / "fig5_score_and_variance.png"
GAMMA = 0.98


def normal_pdf(u, mu, sigma):
    return np.exp(-0.5 * ((u - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def gradient_samples(episodes=1500, fit_episodes=200, seed=0):
    """Per-episode estimates of dJ/dK for the three estimators, at u = -0.6 x + noise."""
    torch.set_num_threads(1)
    set_seed(seed)
    env = gym.make("ScalarLQR-v0")
    env.reset(seed=seed)
    policy = GaussianPolicy(1, 1, (), initial_gain=0.6)
    baseline = fit_baseline([collect_episode(env, policy) for _ in range(fit_episodes)], GAMMA, (64, 64))
    out = {"full return  G_0": [], "reward-to-go  G_t": [], "baseline  G_t − b(x_t)": []}
    for _ in range(episodes):
        states, actions, rewards = collect_episode(env, policy)
        G = torch.as_tensor(reward_to_go(rewards, GAMMA))
        with torch.no_grad():
            delta = G - baseline(states)
        for name, w in zip(out, (torch.full_like(G, G[0].item()), G, delta)):
            # Parameter 0 is the weight w = -K, so dJ/dK = -dJ/dw.
            out[name].append(-episode_gradient(policy, states, actions, w, GAMMA)[0])
    env.close()
    return {k: np.array(v) for k, v in out.items()}


fig, ax = plt.subplots(1, 3, figsize=(18, 5.6))

# a. One sample, one update: the bell curve moves.
mu, sigma, u, step = 0.0, 0.6, 0.9, 0.12
d_mu = (u - mu) / sigma**2
d_logsig = (u - mu) ** 2 / sigma**2 - 1
grid = np.linspace(-2.2, 2.6, 400)
a = ax[0]
a.plot(grid, normal_pdf(grid, mu, sigma), color="black", lw=2.5, label=f"before: μ = {mu:g}, σ = {sigma:g}")
for sign, colour, word in ((+1, "tab:green", "δ > 0 (better than expected)"), (-1, "tab:red", "δ < 0 (worse than expected)")):
    mu_new = mu + sign * step * d_mu
    sigma_new = sigma * np.exp(sign * step * d_logsig)
    a.plot(grid, normal_pdf(grid, mu_new, sigma_new), color=colour, lw=2,
           label=f"after, {word}: μ = {mu_new:+.2f}, σ = {sigma_new:.2f}")
a.axvline(u, color="tab:blue", ls="--", lw=1.5)
a.text(u + 0.05, 0.45,f"tried action\nu = {u:g}", color="tab:blue", fontsize=9)
a.set_ylim(0, 0.95)
a.set_title(f"a. One sample, one step:  θ ← θ + α·δ·∇ log π\n"
            f"∂logπ/∂μ = (u−μ)/σ² = {d_mu:.2f},   ∂logπ/∂logσ = (u−μ)²/σ² − 1 = {d_logsig:.2f}")
a.set_xlabel("action u"); a.set_ylabel("probability density π(u | x)")
a.legend(loc="upper left", fontsize=8.5)

# b. The two score functions over all actions: both average to zero under the policy.
a = ax[1]
uu = np.linspace(-3 * sigma, 3 * sigma, 400)
a.fill_between(uu, 0, 4 * normal_pdf(uu, mu, sigma), color="lightgray", alpha=0.6, label="π(u | x), scaled: how often each u is tried")
a.plot(uu, (uu - mu) / sigma**2, color="tab:orange", lw=2.2, label="∂logπ/∂μ = (u−μ)/σ²")
a.plot(uu, (uu - mu) ** 2 / sigma**2 - 1, color="tab:purple", lw=2.2, label="∂logπ/∂logσ = (u−μ)²/σ² − 1")
for s in (-sigma, sigma):
    a.axvline(mu + s, color="gray", ls=":", lw=1)
a.text(mu + sigma + 0.03, -3.6, "±σ", color="gray")
a.axhline(0, color="black", lw=0.6)
a.text(0, 6.3, "inside ±σ: purple < 0 (narrows if δ > 0)\noutside ±σ: purple > 0 (widens if δ > 0)",
       ha="center", va="top", fontsize=9, color="tab:purple",
       bbox=dict(facecolor="white", edgecolor="none", alpha=0.9))
a.set_ylim(-4, 7)
a.set_title("b. The direction each tried action pushes\n(averaged over π, both curves give exactly 0)")
a.set_xlabel("action u  (μ = 0, σ = 0.6)"); a.set_ylabel("score")
a.legend(loc="lower right", fontsize=8.5)

# c. Many episodes: the spread of single-episode gradient estimates.
samples = gradient_samples()
reference = samples["baseline  G_t − b(x_t)"].mean()
lo, hi = np.percentile(np.concatenate(list(samples.values())), [2, 98])
bins = np.linspace(lo, hi, 70)
a = ax[2]
for (name, g), colour in zip(samples.items(), ("tab:gray", "tab:orange", "tab:blue")):
    uphill = np.mean(np.sign(g) == np.sign(reference))
    rel = (np.mean(g**2) - g.mean() ** 2) / g.mean() ** 2
    a.hist(g, bins=bins, histtype="step", lw=2, color=colour, density=True,
           label=f"{name}\n  mean {g.mean():+.2f}, rel. variance {rel:.0f}, uphill in {100 * uphill:.0f}% of episodes")
a.axvline(reference, color="black", lw=1.5, label="their common mean: the true slope")
a.axvline(0, color="black", lw=0.6, ls=":")
a.set_title("c. Scalar LQR, K = 0.6: ∂J/∂K estimated from ONE episode,\n"
            f"repeated {len(samples['full return  G_0'])} times (same policy, new noise each time)")
a.set_xlabel("single-episode estimate of ∂J/∂K"); a.set_ylabel("density")
a.set_ylim(0, a.get_ylim()[1] * 1.5)     # room for the legend above the peak
a.legend(loc="upper right", fontsize=8)

fig.suptitle("Figure 5. The policy-gradient step on a Gaussian policy, and the variance of the estimate", fontsize=13)
fig.tight_layout()
fig.savefig(OUT, dpi=110)
for name, g in samples.items():
    print(f"{name.replace('−', '-'):<26} mean {g.mean():+.3f}  std {g.std():.3f}  uphill {np.mean(np.sign(g) == np.sign(reference)):.2f}")
print("saved", OUT)
