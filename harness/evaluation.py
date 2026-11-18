"""Deterministic evaluation on a fixed set of seeds, the same for every algorithm and every group."""

import gymnasium as gym
import numpy as np

EVAL_SEEDS = tuple(range(1000, 1020))


def evaluate(agent, env, seeds=EVAL_SEEDS):
    """Run the agent without exploration noise, one episode per seed. Returns a summary dict."""
    returns, costs, distances = [], [], []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        if hasattr(agent, "reset"):
            agent.reset()
        done, episode_return, episode_cost, info = False, 0.0, 0.0, {}
        while not done:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_return += float(reward)
            episode_cost += float(info.get("cost", 0.0))
        returns.append(episode_return)
        costs.append(episode_cost)
        distances.append(float(info.get("distance", np.nan)))
    return {
        "return": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "cost": float(np.mean(costs)),
        "distance": float(np.mean(distances)),
    }


def rollout_fingertip(agent, env_id, seed=EVAL_SEEDS[0]):
    """One deterministic episode on the arm. Returns the fingertip path, shape (T, 2)."""
    env = gym.make(env_id)
    obs, _ = env.reset(seed=seed)
    if hasattr(agent, "reset"):
        agent.reset()
    path, done = [], False
    while not done:
        obs, _, terminated, truncated, info = env.step(agent.select_action(obs, deterministic=True))
        path.append(info["fingertip"])
        done = terminated or truncated
    env.close()
    return np.array(path)
