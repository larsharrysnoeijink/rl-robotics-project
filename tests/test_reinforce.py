"""REINFORCE, checked against values worked out by hand."""

import unittest

import gymnasium as gym
import numpy as np
import torch

import envs  # noqa: F401
from harness.registry import build_agent
from models.reinforce import policy_loss, reward_to_go


class TestRewardToGo(unittest.TestCase):
    def test_hand_trace(self):
        # G_2 = 3, G_1 = 2 + 0.5 * 3 = 3.5, G_0 = 1 + 0.5 * 3.5 = 2.75
        np.testing.assert_allclose(reward_to_go([1.0, 2.0, 3.0], 0.5), [2.75, 3.5, 3.0])


class TestPolicyLoss(unittest.TestCase):
    def test_hand_trace(self):
        # -(1 * 2 * 0.5 + 0.5 * 4 * 0.25) / 2 = -(1 + 0.5) / 2 = -0.75
        loss = policy_loss(torch.tensor([0.5, 0.25]), torch.tensor([2.0, 4.0]), 0.5)
        self.assertAlmostEqual(loss.item(), -0.75)


class TestAgent(unittest.TestCase):
    def test_learns_only_at_the_end_of_an_episode(self):
        env = gym.make("ScalarLQR-v0")
        agent, _ = build_agent("REINFORCE", env, {"hidden_sizes": ()})
        before = [p.clone() for p in agent.policy.parameters()]
        unchanged = lambda: all(torch.equal(a, b) for a, b in zip(before, agent.policy.parameters()))
        obs, _ = env.reset(seed=0)
        action = agent.select_action(obs)
        next_obs, reward, *_ = env.step(action)
        agent.update(obs, action, reward, next_obs, False, False)
        self.assertTrue(unchanged())
        agent.update(next_obs, action, reward, next_obs, False, True)
        self.assertFalse(unchanged())
        env.close()


if __name__ == "__main__":
    unittest.main()
