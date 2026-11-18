"""A random agent: the simplest possible implementation of the agent interface.

Every RL model in this folder should expose the same constructor and methods,
so that train.py can swap algorithms with a single command-line argument.
Use this file as the template for new models.
"""

from pathlib import Path

import numpy as np


class RandomAgent:
    def __init__(self, obs_dim, act_dim, act_low, act_high, device="cpu", **kwargs):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_low = np.asarray(act_low, dtype=np.float32)
        self.act_high = np.asarray(act_high, dtype=np.float32)
        self.device = device

    def select_action(self, obs, deterministic=False):
        """Return an action (np.ndarray of shape (act_dim,)) for the observation."""
        if deterministic:
            return np.zeros(self.act_dim, dtype=np.float32)
        return np.random.uniform(self.act_low, self.act_high).astype(np.float32)

    def update(self, obs, action, reward, next_obs, terminated, truncated):
        """Called by the training loop after EVERY environment step.

        Off-policy methods (DDPG, TD3, SAC) store the transition and learn here.
        On-policy methods (REINFORCE, A2C) buffer the transition and learn when
        the episode (or rollout) ends. A random agent learns nothing.
        """
        pass

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("random agent: nothing to save\n")

    def load(self, path):
        pass


ALGO = "RANDOM"
DEFAULTS = {}
AGENT_CLASS = RandomAgent
