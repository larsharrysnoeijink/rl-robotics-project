"""The optimal controller of the scalar LQR problem, from the Riccati equation.

    python train.py --algo RICCATI --env ScalarLQR-v0

writes a run that compare.py draws as a horizontal line, the known target for a learning curve.
"""

import numpy as np

from envs.scalar_lqr import riccati_solution

ALGO = "RICCATI"
TRAINABLE = False
DEFAULTS = {}


class RiccatiController:
    def __init__(self, obs_dim, act_dim, act_low, act_high, device):
        _, self.K = riccati_solution()

    def select_action(self, obs, deterministic=True):
        return np.array([-self.K * float(np.asarray(obs).item())], dtype=np.float32)

    def update(self, obs, action, reward, next_obs, terminated, truncated):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass


AGENT_CLASS = RiccatiController
