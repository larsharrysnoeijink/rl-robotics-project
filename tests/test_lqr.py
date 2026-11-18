"""The LQR baseline behaves as the course says it does: good near the target, failing from the start."""

import unittest

import gymnasium as gym
import numpy as np

import envs  # noqa: F401
from baselines.lqr import linearise
from envs.planar_arm_dynamics import step
from harness.evaluation import EVAL_SEEDS, evaluate
from harness.registry import build_agent


class TestLQR(unittest.TestCase):
    def setUp(self):
        self.env = gym.make("PlanarArm-v0")
        self.agent, _ = build_agent("LQR", self.env)

    def tearDown(self):
        self.env.close()

    def run_from(self, state):
        unwrapped = self.env.unwrapped
        unwrapped.reset(seed=0)
        unwrapped.set_state(state[:2], state[2:])
        obs, info = state, {}
        for _ in range(200):
            obs, _, _, _, info = unwrapped.step(self.agent.select_action(obs))
        return info["distance"]

    def test_target_is_an_equilibrium_of_the_linearisation_point(self):
        np.testing.assert_allclose(step(self.agent.x_star, self.agent.u_star), self.agent.x_star, atol=1e-9)

    def test_linearisation_predicts_a_small_step(self):
        A, B = linearise(self.agent.x_star, self.agent.u_star)
        dx = np.array([0.002, -0.002, 0.004, 0.002])
        du = np.array([0.002, -0.002])
        linear = self.agent.x_star + A @ dx + B @ du
        np.testing.assert_allclose(step(self.agent.x_star + dx, self.agent.u_star + du), linear, atol=1e-4)

    def test_succeeds_near_the_target(self):
        start = self.agent.x_star + np.array([-0.2, 0.1, 0.0, 0.0])
        self.assertLess(self.run_from(start), 0.01)

    def test_fails_from_the_start_state(self):
        self.assertGreater(evaluate(self.agent, self.env, EVAL_SEEDS[:3])["distance"], 0.5)


if __name__ == "__main__":
    unittest.main()
