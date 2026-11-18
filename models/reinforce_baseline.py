"""REINFORCE with a baseline fitted to observed returns.

Same as reinforce.py, with G_t replaced by how much better than usual the outcome was:

    delta_t       = G_t - b_w(x_t)
    policy loss   = - mean_t [ gamma^t * delta_t * log pi(u_t | x_t) ]
    baseline loss = mean_t [ (b_w(x_t) - G_t)^2 ]

The baseline is ordinary regression on the return: it reads no successor value, so no Bellman
object appears, and subtracting it reduces the variance of the gradient estimate without changing
its expected value. delta_t uses the baseline as it stood before this episode's fit, because the
identity is exact only for a baseline built from earlier episodes (Week 9 deck).
"""

import numpy as np
import torch

from models.reinforce import policy_loss, reward_to_go
from utils.networks import GaussianPolicy, mlp


def baseline_delta(returns, baseline_values):
    """delta_t = G_t - b_w(x_t). Both inputs have shape (T,); no gradient may flow through delta."""
    # ===== YOUR CODE HERE (≈1 line) ===========================
    # Goal: the difference between the observed return and the baseline, detached from the baseline.
    # Hint: Week 9 deck, line 6 of "REINFORCE with a baseline". The policy step must not move w.
    delta_t = returns - baseline_values.detach()
    return delta_t
    # ==========================================================


def baseline_loss(returns, baseline_values):
    """Regression loss of the baseline on the return, averaged over the episode."""
    # ===== YOUR CODE HERE (≈1 line) ===========================
    # Goal: the mean squared error between b_w(x_t) and G_t.
    # Hint: Week 9 deck, line 7 is one stochastic gradient step on this quantity.
    loss = torch.mean((baseline_values - returns) ** 2)
    return loss
    # ==========================================================


class ReinforceBaselineAgent:
    def __init__(self, obs_dim, act_dim, act_low, act_high, device,
                 *, hidden_sizes, baseline_hidden_sizes, policy_lr, baseline_lr, gamma, initial_gain):
        self.device = device
        self.gamma = gamma
        self.policy = GaussianPolicy(obs_dim, act_dim, hidden_sizes, initial_gain=initial_gain).to(device)
        self.baseline = mlp(obs_dim, 1, baseline_hidden_sizes).to(device)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.baseline_optimizer = torch.optim.Adam(self.baseline.parameters(), lr=baseline_lr)
        self.states, self.actions, self.rewards = [], [], []    # the current episode

    def select_action(self, obs, deterministic=False):
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            dist = self.policy(x)
            u = dist.mean if deterministic else dist.sample()
        return u.cpu().numpy()

    def update(self, obs, action, reward, next_obs, terminated, truncated):
        self.states.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        if terminated or truncated:
            self._learn_from_episode()
            self.states, self.actions, self.rewards = [], [], []

    def _learn_from_episode(self):
        x = torch.as_tensor(np.array(self.states), dtype=torch.float32, device=self.device)
        u = torch.as_tensor(np.array(self.actions), dtype=torch.float32, device=self.device)
        G = torch.as_tensor(reward_to_go(self.rewards, self.gamma), device=self.device)
        values = self.baseline(x).squeeze(-1)
        delta = baseline_delta(G, values)       # uses the baseline before this episode's fit

        self.baseline_optimizer.zero_grad()
        baseline_loss(G, values).backward()
        self.baseline_optimizer.step()

        log_probs = self.policy(x).log_prob(u).sum(dim=-1)
        loss = policy_loss(log_probs, delta, self.gamma)
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()

    def save(self, path):
        torch.save({"policy": self.policy.state_dict(), "baseline": self.baseline.state_dict()}, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy"])
        self.baseline.load_state_dict(checkpoint["baseline"])


ALGO = "REINFORCE-BASELINE"
DEFAULTS = dict(
    hidden_sizes=(64, 64),
    baseline_hidden_sizes=(64, 64),
    policy_lr=1e-3,     # learning rate of the policy network
    baseline_lr=1e-3,   # learning rate of the baseline network
    gamma=0.99,         # discount factor
    initial_gain=None,  # for LQR: a linear policy u = -K x starting at this K
)
AGENT_CLASS = ReinforceBaselineAgent
