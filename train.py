"""Train one algorithm with one seed, and write its results to runs/.

Run from the project folder, with the environment activated:

    python train.py --algo REINFORCE --seed 3 --steps 100000
    python train.py --algo REINFORCE --seed 3 --steps 100000 --variation lr3e4 --set lr=3e-4
    python train.py --algo LQR

Each run writes runs/<env>_<algo>_<variation>_seed<N>/ holding results.npz and final.pt.
Use compare.py to plot runs together and watch.py to see a policy.
"""

import argparse
import ast

import torch

import envs  # noqa: F401  (registers the project environments)
import gymnasium as gym
from harness.registry import available_algorithms, build_agent, is_trainable
from harness.results import RunLog
from harness.training import evaluate_only, train
from utils.seeding import set_seed


def parse_overrides(pairs):
    """['lr=3e-4', 'hidden_sizes=(32,32)'] -> {'lr': 0.0003, 'hidden_sizes': (32, 32)}"""
    overrides = {}
    for pair in pairs:
        key, separator, value = pair.partition("=")
        if not separator:
            raise SystemExit(f"--set expects name=value, got '{pair}'")
        try:
            overrides[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            overrides[key] = value
    return overrides


def run_folder(env_id, algo, variation, seed):
    return f"runs/{env_id}_{algo}_{variation}_seed{seed}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algo", required=True, help=f"one of: {', '.join(sorted(available_algorithms()))}")
    parser.add_argument("--env", default="PlanarArm-v0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=100_000, help="training budget, in environment steps")
    parser.add_argument("--eval-every", type=int, default=5_000, help="environment steps between evaluations")
    parser.add_argument("--checkpoint-every", type=int, default=0, help="save a checkpoint every N steps (0 = final only)")
    parser.add_argument("--save-data", action="store_true", help="also write every transition to dataset.npz")
    parser.add_argument("--variation", default="base", help="name of this experiment, used in plots")
    parser.add_argument("--set", nargs="*", default=[], metavar="NAME=VALUE", help="change hyperparameters")
    parser.add_argument("--threads", type=int, default=1, help="CPU threads for torch (1 is usually fastest here)")
    args = parser.parse_args()

    if "_" in args.variation:
        raise SystemExit("--variation cannot contain '_'.")

    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    env = gym.make(args.env)
    agent, params = build_agent(args.algo, env, parse_overrides(args.set))
    env.close()

    trainable = is_trainable(args.algo)
    config = {"env": args.env, "algo": args.algo, "seed": args.seed, "variation": args.variation,
              "steps": args.steps if trainable else 0, "eval_every": args.eval_every,
              "hyperparameters": {key: repr(value) for key, value in params.items()}}
    log = RunLog(run_folder(args.env, args.algo, args.variation, args.seed), config)

    if trainable:
        train(args.env, agent, args.steps, args.eval_every, log, args.seed, args.checkpoint_every, args.save_data)
    else:
        evaluate_only(args.env, agent, log)
    print(f"Results in {log.run_dir}")


if __name__ == "__main__":
    main()
