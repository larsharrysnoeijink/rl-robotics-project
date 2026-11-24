"""The idea of a baseline in three pictures, on ScalarLQR-v0 with the start policy of part 2 (K = 0.6).

    python output/week10/scripts/theory_baseline.py

a. the scores G_t that plain REINFORCE gives the actions of 1 500 episodes: never positive
b. the scores G_t - b(x_t) with a baseline, same actions: centred on zero
c. how often the update from ONE episode moves K in the right direction, for the three estimators
"""
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

import envs  # noqa: F401  (registers ScalarLQR-v0)
from experiments.gradient_variance import collect_episode, episode_gradient, fit_baseline
from models.reinforce import reward_to_go
from utils.networks import GaussianPolicy
from utils.seeding import set_seed

OUT = Path(__file__).resolve().parents[1] / "theory" / "baseline_idea.png"
GAMMA = 0.98
torch.set_num_threads(1)
set_seed(0)
env = gym.make("ScalarLQR-v0")
env.reset(seed=0)
policy = GaussianPolicy(1, 1, (), initial_gain=0.6)
baseline = fit_baseline([collect_episode(env, policy) for _ in range(200)], GAMMA, (64, 64))

first_G, first_delta, estimates = [], [], {"full return": [], "reward-to-go": [], "with baseline": []}
for _ in range(1500):
    states, actions, rewards = collect_episode(env, policy)
    G = torch.as_tensor(reward_to_go(rewards, GAMMA))
    with torch.no_grad():
        delta = G - baseline(states)
    first_G.extend(G.tolist()); first_delta.extend(delta.tolist())          # the score of every action
    for name, w in zip(estimates, (torch.full_like(G, G[0].item()), G, delta)):
        estimates[name].append(-episode_gradient(policy, states, actions, w, GAMMA)[0])   # dJ/dK
env.close()
right_way = np.sign(np.mean(estimates["with baseline"]))
uphill = {k: 100 * np.mean(np.sign(v) == right_way) for k, v in estimates.items()}
first_G, first_delta = np.array(first_G), np.array(first_delta)

fig, ax = plt.subplots(1, 3, figsize=(17, 5))
bins = np.linspace(-40, 25, 66)
a = ax[0]
a.hist(first_G, bins=bins, color="tab:red")
a.axvline(0, color="black", lw=1)
a.set_title("a. Without a baseline: the score of an action is $G_t$\n"
            "It is never positive: \"every action was bad\"")
a.set_xlabel("score of an action (all 50 steps of 1 500 episodes)"); a.set_ylabel("number of actions")

a = ax[1]
a.hist(first_delta[first_delta <= 0], bins=bins, color="tab:red", label=f"worse than usual: {100 * np.mean(first_delta <= 0):.0f} %")
a.hist(first_delta[first_delta > 0], bins=bins, color="tab:green", label=f"better than usual: {100 * np.mean(first_delta > 0):.0f} %")
a.axvline(0, color="black", lw=1)
a.set_ylim(ax[0].get_ylim() if ax[0].get_ylim()[1] > a.get_ylim()[1] else a.get_ylim())
a.set_title("b. With a baseline: the score is $G_t - b(x_t)$\n"
            "It is centred on zero: better or worse than usual")
a.set_xlabel("score of an action (same actions, same scale)"); a.legend(fontsize=9)

a = ax[2]
bars = a.bar(["full return", "reward-to-go\n(plain REINFORCE)", "with baseline"], list(uphill.values()),
             color=["tab:gray", "tab:orange", "tab:blue"])
for bar, v in zip(bars, uphill.values()):
    a.text(bar.get_x() + bar.get_width() / 2, v + 1.5, f"{v:.0f} %", ha="center", fontsize=13, fontweight="bold")
a.axhline(50, color="black", ls=":")
a.text(2.45, 51, "coin flip", ha="right", fontsize=9)
a.set_ylim(0, 100)
a.set_title("c. How often does the update from ONE episode\nmove K in the right direction? (1 500 episodes)")
a.set_ylabel("% of episodes")

fig.suptitle("What a baseline does (scalar LQR, start policy K = 0.6)", fontsize=13)
fig.tight_layout()
OUT.parent.mkdir(exist_ok=True)
fig.savefig(OUT, dpi=150)
print("saved", OUT, {k: round(v) for k, v in uphill.items()})
