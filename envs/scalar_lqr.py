"""Scalar LQR as an environment, with the optimum available in closed form.

    x_{k+1} = a x_k + b u_k,    reward = -(q x^2 + r u^2),    x_0 ~ N(0, 1)

The discounted Riccati equation gives K* and the optimal cost, so a learning curve on this problem
can be read against a known target (exercise 9.7 of the Week 9 notes uses the same numbers).
The action is not bounded.
"""

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

A, B, Q, R, GAMMA = 1.1, 1.0, 1.0, 0.1, 0.98
EPISODE_STEPS = 50


def riccati_solution():
    """(p, K) with cost-to-go p x^2 and optimal law u = -K x, for the discounted problem."""
    p = 1.0
    for _ in range(10_000):
        p = Q + GAMMA * A * A * p - (GAMMA * A * B * p) ** 2 / (R + GAMMA * B * B * p)
    K = GAMMA * A * B * p / (R + GAMMA * B * B * p)
    return p, K


def optimal_return():
    """Mean discounted return of u = -K x over the episode length, with x_0 ~ N(0, 1)."""
    _, K = riccati_solution()
    closed_loop = A - B * K
    per_step = (GAMMA * closed_loop**2) ** np.arange(EPISODE_STEPS)
    return float(-(Q + R * K**2) * per_step.sum())


class ScalarLQREnv(gym.Env):
    def __init__(self):
        self.observation_space = Box(-np.inf, np.inf, shape=(1,), dtype=np.float64)
        self.action_space = Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)
        self.x = 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.x = float(self.np_random.normal(0.0, 1.0))
        return np.array([self.x]), {}

    def step(self, action):
        u = float(np.asarray(action).item())
        reward = -(Q * self.x**2 + R * u**2)
        self.x = A * self.x + B * u
        return np.array([self.x]), reward, False, False, {"cost": 0.0}
