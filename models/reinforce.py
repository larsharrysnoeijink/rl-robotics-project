"""REINFORCE: the Monte-Carlo policy gradient, for continuous actions.

Play one whole episode with the current stochastic policy, then raise the probability of each
action in proportion to the reward that followed it:

    G_t  = r_{t+1} + gamma * G_{t+1}                              (reward-to-go)
    loss = - mean_t [ gamma^t * G_t * log pi(u_t | x_t) ]

Same algorithm as the Week 9 deck. There is no baseline here: reinforce_baseline.py adds one.
No Bellman object appears, since G_t is an observed return and nothing is bootstrapped.
"""

import numpy as np
import torch

from utils.networks import GaussianPolicy


def reward_to_go(rewards, gamma):
    """G_t for every t of one episode, computed backwards. rewards: list of T floats -> array (T,)."""
    returns = np.zeros(len(rewards), dtype=np.float32)
    G = 0.0
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        returns[t] = G
    return returns


def policy_loss(log_probs, weights, gamma):
    """Policy-gradient loss. weights is G_t for plain REINFORCE, G_t - b(x_t) with a baseline.

    log_probs and weights have shape (T,). Differentiating this and taking a descent step gives the
    update of line 8 of the algorithm on the deck, averaged over the episode.
    """
    discounts = gamma ** torch.arange(len(weights), dtype=torch.float32, device=weights.device)
    return -(discounts * weights * log_probs).mean()


class ReinforceAgent:
    def __init__(self, obs_dim, act_dim, act_low, act_high, device, *, hidden_sizes, lr, gamma, initial_gain):
        self.device = device
        self.gamma = gamma
        self.policy = GaussianPolicy(obs_dim, act_dim, hidden_sizes, initial_gain=initial_gain).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
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
        if terminated or truncated:     # REINFORCE can only learn once the episode is over
            self._learn_from_episode()
            self.states, self.actions, self.rewards = [], [], []

    def _learn_from_episode(self):
        x = torch.as_tensor(np.array(self.states), dtype=torch.float32, device=self.device)
        u = torch.as_tensor(np.array(self.actions), dtype=torch.float32, device=self.device)
        G = torch.as_tensor(reward_to_go(self.rewards, self.gamma), device=self.device)
        log_probs = self.policy(x).log_prob(u).sum(dim=-1)      # sum over the action dimensions
        loss = policy_loss(log_probs, G, self.gamma)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, path):
        torch.save({"policy": self.policy.state_dict()}, path)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path, map_location=self.device)["policy"])


ALGO = "REINFORCE"
DEFAULTS = dict(
    hidden_sizes=(64, 64),
    lr=1e-3,            # learning rate
    gamma=0.99,         # discount factor
    initial_gain=None,  # for LQR: a linear policy u = -K x starting at this K
)
AGENT_CLASS = ReinforceAgent
