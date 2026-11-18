"""The C1 measurement tools. They use your baseline, so they need the C1 cores to run.

    python -m unittest tests.test_c1_variance
"""

import unittest

import gymnasium as gym
import numpy as np
import torch

import envs  # noqa: F401
from experiments.gradient_variance import constant_baseline_sweep, measure, relative_variance
from utils.networks import GaussianPolicy


class TestArithmetic(unittest.TestCase):
    def test_relative_variance_hand_trace(self):
        # mean (2, 0), E||g||^2 = (1 + 9) / 2 = 5, ||Eg||^2 = 4: (5 - 4) / 4
        self.assertAlmostEqual(relative_variance(np.array([[1.0, 0.0], [3.0, 0.0]])), 0.25)

    def test_sweep_finds_the_best_constant(self):
        # to_go = 2 * score + noise, so subtracting b = 2 removes the part proportional to the score.
        rng = np.random.default_rng(0)
        score = rng.normal(size=(5000, 1))
        to_go = 2.0 * score + 0.1 * rng.normal(size=(5000, 1)) + 0.5
        _, _, best = constant_baseline_sweep(to_go, score)
        self.assertAlmostEqual(best, 2.0, delta=0.1)


class TestMeasurement(unittest.TestCase):
    def test_baseline_reduces_variance_on_lqr(self):
        torch.manual_seed(0)
        env = gym.make("ScalarLQR-v0")
        policy = GaussianPolicy(1, 1, (), initial_gain=0.8)
        results = measure(policy, env, baseline_episodes=100, measure_episodes=400, gamma=0.98,
                          hidden_sizes=(32, 32), seed=0)
        env.close()
        self.assertLess(relative_variance(results["baseline"]), relative_variance(results["reward-to-go"]))


if __name__ == "__main__":
    unittest.main()
