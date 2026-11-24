"""How well does a run's own trained baseline do its job, at the run's policy?

    python output/week10/scripts/baseline_quality.py runs/PlanarArm-v0_REINFORCE-BASELINE_base_seed80 [more runs]
    python output/week10/scripts/baseline_quality.py RUN --checkpoints all --csv measurements.csv

experiments/gradient_variance.py fits a fresh (64, 64) baseline of its own, so it cannot show what a
run's baseline settings did. This tool uses the baseline saved with the policy (final.pt, or every
step_*.pt with --checkpoints all). For each it plays training-style episodes (random start on the arm,
with noise) and reports:

    sigma          the policy's exploration noise, one value per joint
    fit R^2        1 - sum (G_t - b(x_t))^2 / sum (G_t - mean G)^2: the share of the spread of the
                   returns that the baseline predicts from the state (1 = perfect, 0 = no better than
                   a constant)
    variance       E||g - E g||^2 of the single-episode policy-gradient estimate, summed over all
                   parameters, for reward-to-go (no baseline), the run's own baseline, and a fresh
                   baseline that experiments/gradient_variance.py fits with 2000 Adam steps on
                   separate episodes. The ratios to reward-to-go say how much each baseline removes:
                   below 1 it lowers the variance, above 1 it raises it.

All three estimators share one mean, so their variances compare directly. The relative variance of
gradient_variance.py divides by ||E g||^2, which is close to 0 at a trained policy and is then only
noise; that is why this tool reports the plain variance.
"""
import argparse
import ast
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import gymnasium as gym
import numpy as np
import torch

import envs  # noqa: F401  (registers the project environments)
from envs import TRAINING_KWARGS
from experiments.gradient_variance import collect_episode, episode_gradient, fit_baseline
from harness.registry import build_agent
from harness.results import load_run
from models.reinforce import reward_to_go
from utils.seeding import set_seed

FIELDS = ["run", "variation", "seed", "checkpoint", "step", "sigma1", "sigma2", "r2_own", "r2_fresh",
          "var_reward_to_go", "var_own", "var_fresh"]


def load(run_dir, checkpoint):
    cfg = load_run(str(run_dir))["config"]
    overrides = {k: ast.literal_eval(v) if v[:1] not in "\"'" else v.strip("'\"")
                 for k, v in cfg["hyperparameters"].items()}
    env = gym.make(cfg["env"])
    agent, _ = build_agent(cfg["algo"], env, overrides)
    env.close()
    agent.load(Path(run_dir) / checkpoint)
    return agent, cfg, overrides


def measure(run_dir, checkpoint, episodes, fit_episodes, seed):
    agent, cfg, hp = load(run_dir, checkpoint)
    if not hasattr(agent, "baseline"):
        raise SystemExit(f"{run_dir}: no baseline in this run")
    gamma, policy, own = hp["gamma"], agent.policy, agent.baseline
    set_seed(seed)
    env = gym.make(cfg["env"], **TRAINING_KWARGS.get(cfg["env"], {}))
    env.reset(seed=seed)
    fresh = fit_baseline([collect_episode(env, policy) for _ in range(fit_episodes)], gamma, (64, 64))

    grads = {"reward-to-go": [], "own": [], "fresh": []}
    all_G, all_own, all_fresh = [], [], []
    for _ in range(episodes):
        states, actions, rewards = collect_episode(env, policy)
        G = torch.as_tensor(reward_to_go(rewards, gamma))
        with torch.no_grad():
            b_own, b_fresh = own(states).squeeze(-1), fresh(states)
        all_G.append(G.numpy()); all_own.append(b_own.numpy()); all_fresh.append(b_fresh.numpy())
        for name, w in (("reward-to-go", G), ("own", G - b_own), ("fresh", G - b_fresh)):
            grads[name].append(episode_gradient(policy, states, actions, w, gamma))
    env.close()

    G, b_own, b_fresh = (np.concatenate(a) for a in (all_G, all_own, all_fresh))
    spread = np.sum((G - G.mean()) ** 2)
    var = {k: float(np.sum(np.var(np.array(v), axis=0, ddof=1))) for k, v in grads.items()}
    sigma = policy.log_std.exp().detach().numpy()
    step = int(checkpoint[5:-3]) if checkpoint.startswith("step_") else -1
    return {"run": Path(run_dir).name, "variation": cfg.get("variation", "base"), "seed": cfg.get("seed"),
            "checkpoint": checkpoint, "step": step, "sigma1": float(sigma[0]), "sigma2": float(sigma[1]),
            "r2_own": float(1 - np.sum((G - b_own) ** 2) / spread),
            "r2_fresh": float(1 - np.sum((G - b_fresh) ** 2) / spread),
            "var_reward_to_go": var["reward-to-go"], "var_own": var["own"], "var_fresh": var["fresh"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--checkpoints", default="final", help="'final' (final.pt) or 'all' (every step_*.pt)")
    parser.add_argument("--episodes", type=int, default=300, help="episodes for the measurement")
    parser.add_argument("--fit-episodes", type=int, default=200, help="episodes for the fresh baseline")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", default=None, help="append one row per measurement to this file")
    args = parser.parse_args()
    torch.set_num_threads(1)

    for run_dir in args.runs:
        if args.checkpoints == "all":
            names = sorted((p.name for p in Path(run_dir).glob("step_*.pt")), key=lambda n: int(n[5:-3]))
        else:
            names = ["final.pt"]
        for checkpoint in names:
            row = measure(run_dir, checkpoint, args.episodes, args.fit_episodes, args.seed)
            print(f"{row['run']:<48} {checkpoint:<16} sigma ({row['sigma1']:.2f}, {row['sigma2']:.2f})  "
                  f"R2 own {row['r2_own']:6.3f}  fresh {row['r2_fresh']:6.3f} | variance: no baseline "
                  f"{row['var_reward_to_go']:9.3g}, own {row['var_own'] / row['var_reward_to_go']:5.2f}x, "
                  f"fresh {row['var_fresh'] / row['var_reward_to_go']:5.2f}x", flush=True)
            if args.csv:
                path = Path(args.csv)
                new = not path.exists()
                with open(path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=FIELDS)
                    if new:
                        writer.writeheader()
                    writer.writerow(row)


if __name__ == "__main__":
    main()
