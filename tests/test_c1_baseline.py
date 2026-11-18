"""Homework C1: the two blanked functions of models/reinforce_baseline.py.

    python -m unittest tests.test_c1_baseline
"""

import unittest

import gymnasium as gym
import torch

import envs  # noqa: F401
from harness.registry import build_agent
from models.reinforce_baseline import baseline_delta, baseline_loss


class TestDelta(unittest.TestCase):
    def test_hand_trace(self):
        delta = baseline_delta(torch.tensor([2.0, 2.0, 2.0]), torch.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(delta.tolist(), [1.0, 0.0, -1.0])

    def test_no_gradient_reaches_the_baseline(self):
        values = torch.tensor([1.0, 2.0], requires_grad=True)
        delta = baseline_delta(torch.tensor([3.0, 3.0]), values)
        self.assertFalse(delta.requires_grad, "The policy step must not move the baseline: detach it.")


class TestBaselineLoss(unittest.TestCase):
    def test_hand_trace(self):
        # ((1 - 2)^2 + (2 - 2)^2 + (3 - 2)^2) / 3 = 2 / 3
        loss = baseline_loss(torch.tensor([2.0, 2.0, 2.0]), torch.tensor([1.0, 2.0, 3.0]))
        self.assertAlmostEqual(loss.item(), 2.0 / 3.0, places=6)

    def test_gradient_moves_the_baseline_towards_the_return(self):
        values = torch.tensor([1.0], requires_grad=True)
        baseline_loss(torch.tensor([3.0]), values).backward()
        self.assertLess(values.grad.item(), 0.0)


class TestAgent(unittest.TestCase):
    def test_one_episode_moves_both_networks(self):
        env = gym.make("ScalarLQR-v0")
        agent, _ = build_agent("REINFORCE-BASELINE", env, {"hidden_sizes": (8,), "baseline_hidden_sizes": (8,)})
        policy = [p.clone() for p in agent.policy.parameters()]
        baseline = [p.clone() for p in agent.baseline.parameters()]
        obs, _ = env.reset(seed=0)
        done = False
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            agent.update(obs, action, reward, next_obs, terminated, truncated)
            obs, done = next_obs, terminated or truncated
        env.close()
        self.assertFalse(all(torch.equal(a, b) for a, b in zip(policy, agent.policy.parameters())))
        self.assertFalse(all(torch.equal(a, b) for a, b in zip(baseline, agent.baseline.parameters())))


if __name__ == "__main__":
    unittest.main()
