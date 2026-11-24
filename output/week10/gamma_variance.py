"""Does a larger discount gamma give a noisier policy-gradient estimate? A measurement, no training.

    python output/week10/gamma_variance.py --gammas 0.95 0.99 0.999
    python output/week10/gamma_variance.py --env ScalarLQR-v0 --gammas 0.9 0.95 0.98 0.99 0.999 \
        --episodes 3000 --csv output/week10/gamma_variance_lqr.csv     (the data of Figure 6i)

Builds the untrained REINFORCE-BASELINE policy of seed 80 (the policy every arm run starts from) and
measures, with experiments/gradient_variance.py, the relative variance
(E||g||^2 - ||E g||^2) / ||E g||^2 of the reward-to-go and the baseline estimators for each gamma, on
training-style arm episodes. Relative variance is scale-free, so the different sizes of the returns
for different gamma do not matter. With --env ScalarLQR-v0 it uses the part 2 policy (K = 0.6).

Caution: the formula divides by ||mean of the samples||^2. When N episodes cannot pin down the mean
gradient (the noise in the mean, tr(Var)/N, is larger than the mean itself), the result sits near N
whatever the truth is. On the arm, with about 4 600 parameters, that happens to the estimators
without a baseline; the scalar LQR, with 3 parameters, does not have the problem.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import gymnasium as gym
import torch

import envs  # noqa: F401  (registers the project environments)
from envs import TRAINING_KWARGS
from experiments.gradient_variance import measure, relative_variance
from harness.registry import build_agent
from utils.seeding import set_seed

parser = argparse.ArgumentParser()
parser.add_argument("--env", default="PlanarArm-v0", help="PlanarArm-v0, or ScalarLQR-v0 (the part 2 policy, K = 0.6)")
parser.add_argument("--gammas", type=float, nargs="+", default=[0.95, 0.99, 0.999])
parser.add_argument("--episodes", type=int, default=500)
parser.add_argument("--fit-episodes", type=int, default=200)
parser.add_argument("--seed", type=int, default=80)
parser.add_argument("--csv", default=None, help="append one row per gamma to this file")
args = parser.parse_args()
torch.set_num_threads(1)

for gamma in args.gammas:
    set_seed(args.seed)
    env = gym.make(args.env, **TRAINING_KWARGS.get(args.env, {}))
    extra = {"hidden_sizes": (), "initial_gain": 0.6} if args.env == "ScalarLQR-v0" else {}
    agent, _ = build_agent("REINFORCE-BASELINE", env, {"gamma": gamma, **extra})
    samples = measure(agent.policy, env, args.fit_episodes, args.episodes, gamma, (64, 64), args.seed)
    env.close()
    rtg, base = relative_variance(samples["reward-to-go"]), relative_variance(samples["baseline"])
    print(f"gamma {gamma:<6} relative variance: reward-to-go {rtg:8.1f}   baseline {base:8.1f}", flush=True)
    if args.csv:
        path = Path(args.csv)
        new = not path.exists()
        with open(path, "a", encoding="utf-8") as f:
            if new:
                f.write("gamma,reward_to_go,baseline\n")
            f.write(f"{gamma},{rtg:.1f},{base:.1f}\n")
