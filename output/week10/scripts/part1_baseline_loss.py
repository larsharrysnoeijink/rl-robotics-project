"""Figure: what baseline_loss measures and what minimising it does, on ScalarLQR-v0.

    python output/week10/scripts/part1_baseline_loss.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1] / "part1_code" / "part1_baseline_loss.png"
sys.path.insert(0, str(ROOT))

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import envs  # noqa: F401  (registers ScalarLQR-v0)
from models.reinforce import reward_to_go
from models.reinforce_baseline import baseline_loss
from utils.networks import GaussianPolicy, mlp

GAMMA = 0.98
torch.manual_seed(0)
env = gym.make("ScalarLQR-v0")
policy = GaussianPolicy(1, 1, (), initial_gain=0.6)          # the frozen policy of part 2
baseline = mlp(1, 1, (64, 64))                                # the default baseline network
opt = torch.optim.Adam(baseline.parameters(), lr=1e-3)        # the default baseline_lr


def episode(seed):
    obs, _ = env.reset(seed=seed)
    xs, rs, done = [], [], False
    while not done:
        with torch.no_grad():
            u = policy(torch.as_tensor(obs, dtype=torch.float32)).sample().numpy()
        xs.append(obs.copy())
        obs, r, term, trunc, _ = env.step(u)
        rs.append(r)
        done = term or trunc
    return np.array(xs, dtype=np.float32), reward_to_go(rs, GAMMA)


def b_of(x):
    with torch.no_grad():
        return baseline(torch.as_tensor(x, dtype=torch.float32).reshape(-1, 1)).squeeze(-1).numpy()


grid = np.linspace(-3, 3, 200)
curves, cloud_x, cloud_g, losses = {}, [], [], []
example = None
SNAPS = (0, 30, 300, 3000)
for ep in range(max(SNAPS) + 1):
    if ep in SNAPS:
        curves[ep] = b_of(grid)
    x, G = episode(seed=ep)
    if ep == 30:
        example = (x.ravel(), G, b_of(x))                     # baseline as it stood before this fit
    if ep >= max(SNAPS) - 300:
        cloud_x.append(x.ravel()); cloud_g.append(G)
    values = baseline(torch.as_tensor(x)).squeeze(-1)
    loss = baseline_loss(torch.as_tensor(G), values)          # YOUR function
    opt.zero_grad(); loss.backward(); opt.step()
    losses.append(loss.item())
env.close()

fig, ax = plt.subplots(1, 2, figsize=(13, 5.4))

# 1. One episode: the residuals the loss squares and averages.
x, G, b = example
t = np.arange(len(G))
a = ax[0]
a.vlines(t, G, b, color="tab:red", alpha=0.6, lw=1.5, label="residual  b(x_t) − G_t")
a.plot(t, G, "o", ms=4, color="black", label="observed return  G_t")
a.plot(t, b, "-", color="tab:blue", lw=2, label="baseline  b_w(x_t)")
mse = np.mean((b - G) ** 2)
a.set_title(f"a. One episode early in training: the baseline's guess is still far\nfrom what really came. loss = average of (red gap)² = {mse:.1f}")
a.set_xlabel("time step t"); a.set_ylabel("return"); a.legend(loc="lower right", fontsize=9)

# 2. Many episodes: the baseline turns into the average return from each state.
a = ax[1]
a.scatter(np.concatenate(cloud_x), np.concatenate(cloud_g), s=3, alpha=0.15, color="black", label="G_t from 300 episodes")
for ep, c in zip(SNAPS, ("lightgray", "tab:cyan", "tab:blue", "navy")):
    a.plot(grid, curves[ep], color=c, lw=2.5, label=f"b_w(x) after {ep} fits")
a.set_xlim(-3, 3)
a.set_title("b. After many episodes the baseline becomes\nthe average return from each state")
a.set_xlabel("state x"); a.set_ylabel("return"); a.legend(loc="lower center", fontsize=9)

fig.suptitle("Part 1. What baseline_loss does (scalar LQR, policy u = −0.6x + noise)", fontsize=13)
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print("saved", OUT, "| loss first/last 50 eps:", round(np.mean(losses[:50]), 1), round(np.mean(losses[-50:]), 1))
