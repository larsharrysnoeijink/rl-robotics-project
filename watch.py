"""See what a trained policy does, or what a baseline does.

    python watch.py --algo REINFORCE --run runs/PlanarArm-v0_REINFORCE_base_seed0      # opens a window
    python watch.py --algo LQR --path                                                  # saves the fingertip path
    python watch.py --algo SAC --checkpoint reference/w11/bundle/sac_reference.pt --path
"""

import argparse
import ast
from pathlib import Path

import numpy as np
import torch

import envs  # noqa: F401  (registers the project environments)
import gymnasium as gym
from envs.planar_arm import OBSTACLE_CENTRE, OBSTACLE_RADIUS, TARGET
from harness.evaluation import EVAL_SEEDS, rollout_fingertip
from harness.plots import plot_fingertip_path
from harness.registry import build_agent, is_trainable
from harness.results import load_run


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algo", required=True)
    parser.add_argument("--run", default=None, help="run folder holding final.pt (not needed for a baseline)")
    parser.add_argument("--checkpoint", default=None, help="a policy file, with default hyperparameters, instead of --run")
    parser.add_argument("--env", default="PlanarArm-v0")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--path", action="store_true", help="save the fingertip path as an image, with no window")
    args = parser.parse_args()

    torch.set_num_threads(1)
    env = gym.make(args.env, render_mode=None if args.path else "human")
    overrides = None
    if args.run is not None:
        stored = load_run(args.run)["config"]["hyperparameters"]
        overrides = {key: ast.literal_eval(value) if value[:1] not in "\"'" else value.strip("'\"")
                     for key, value in stored.items()}
    agent, _ = build_agent(args.algo, env, overrides)
    if is_trainable(args.algo):
        if args.run is None and args.checkpoint is None:
            raise SystemExit(f"{args.algo} needs --run, the folder of a trained run, or --checkpoint.")
        agent.load(args.checkpoint if args.checkpoint else Path(args.run) / "final.pt")

    if args.path:
        name = f"path_{args.algo}.png"
        path = rollout_fingertip(agent, args.env)
        plot_fingertip_path(path, TARGET, OBSTACLE_CENTRE, OBSTACLE_RADIUS, title=args.algo,
                            save_path=Path("outputs") / name)
        env.close()
        return

    for episode in range(args.episodes):
        obs, _ = env.reset(seed=EVAL_SEEDS[episode])
        if hasattr(agent, "reset"):
            agent.reset()
        done, total = False, 0.0
        while not done:
            obs, reward, terminated, truncated, _ = env.step(agent.select_action(obs, deterministic=True))
            total += float(reward)
            done = terminated or truncated
        print(f"Episode {episode + 1}: return {total:.1f}")
    env.close()


if __name__ == "__main__":
    main()
