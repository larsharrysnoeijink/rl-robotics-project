"""The training loop shared by every algorithm. Budgets are in environment steps."""

import time

import gymnasium as gym
import numpy as np

from envs import TRAINING_KWARGS
from harness.evaluation import evaluate


class TransitionLog:
    """Every transition of a run, in the order it happened, for offline use."""

    def __init__(self, capacity, obs_dim, act_dim):
        self.size = 0
        self.arrays = {"obs": np.zeros((capacity, obs_dim), dtype=np.float32),
                       "action": np.zeros((capacity, act_dim), dtype=np.float32),
                       "reward": np.zeros(capacity, dtype=np.float32),
                       "next_obs": np.zeros((capacity, obs_dim), dtype=np.float32),
                       "terminated": np.zeros(capacity, dtype=bool),
                       "truncated": np.zeros(capacity, dtype=bool)}

    def add(self, obs, action, reward, next_obs, terminated, truncated):
        for name, value in zip(self.arrays, (obs, action, reward, next_obs, terminated, truncated)):
            self.arrays[name][self.size] = value
        self.size += 1

    def save(self, path):
        np.savez_compressed(path, **{name: array[:self.size] for name, array in self.arrays.items()})


def train(env_id, agent, total_steps, eval_every, log, seed, checkpoint_every=0, save_data=False):
    """Train `agent` for `total_steps` environment steps, evaluating every `eval_every`.

    Training episodes use TRAINING_KWARGS; evaluation always uses the task as registered.
    The clock in the results counts training only. Ctrl+C stops the run and keeps what it has.
    With `save_data`, every transition is written to dataset.npz for the later milestones.
    """
    env = gym.make(env_id, **TRAINING_KWARGS.get(env_id, {}))
    eval_env = gym.make(env_id)
    env.action_space.seed(seed)
    low, high = env.action_space.low, env.action_space.high

    if save_data:
        data = TransitionLog(total_steps, env.observation_space.shape[0], env.action_space.shape[0])
    steps, wall_clock, next_eval = 0, 0.0, eval_every
    log.log_evaluation(0, 0.0, evaluate(agent, eval_env))
    log.save()
    obs, _ = env.reset(seed=seed)
    episode_return, episode_cost = 0.0, 0.0

    try:
        while steps < total_steps:
            start = time.perf_counter()
            action = agent.select_action(obs)
            # The environment gets the clipped action; the agent learns from the one it chose.
            next_obs, reward, terminated, truncated, info = env.step(np.clip(action, low, high))
            agent.update(obs, action, reward, next_obs, terminated, truncated)
            wall_clock += time.perf_counter() - start

            if save_data:
                data.add(obs, np.clip(action, low, high), reward, next_obs, terminated, truncated)
            obs = next_obs
            steps += 1
            episode_return += float(reward)
            episode_cost += float(info.get("cost", 0.0))

            if terminated or truncated:
                log.log_episode(steps, episode_return, episode_cost)
                obs, _ = env.reset()
                episode_return, episode_cost = 0.0, 0.0

            if steps >= next_eval or steps == total_steps:
                summary = evaluate(agent, eval_env)
                log.log_evaluation(steps, wall_clock, summary)
                log.save()
                next_eval += eval_every
                print(f"step {steps:>8d} | return {summary['return']:9.1f} | cost {summary['cost']:6.1f} | "
                      f"distance {summary['distance']:.3f} | training time {wall_clock:7.1f} s", flush=True)

            if checkpoint_every > 0 and steps % checkpoint_every == 0:
                agent.save(log.run_dir / f"step_{steps:08d}.pt")

    except KeyboardInterrupt:
        print("\nStopped by the user. Saving what the run has so far.")
    finally:
        agent.save(log.run_dir / "final.pt")
        log.save()
        if save_data:
            data.save(log.run_dir / "dataset.npz")
        env.close()
        eval_env.close()


def evaluate_only(env_id, agent, log):
    """For a baseline with nothing to train: a single evaluation, written like any other run."""
    env = gym.make(env_id)
    start = time.perf_counter()
    summary = evaluate(agent, env)
    log.log_evaluation(0, time.perf_counter() - start, summary)
    log.save()
    env.close()
    print(f"return {summary['return']:.1f} | cost {summary['cost']:.1f} | distance {summary['distance']:.3f}")
