"""Small neural-network helpers shared by the RL models."""

import torch
import torch.nn as nn
from torch.distributions import Normal


def mlp(in_dim, out_dim, hidden_sizes):
    """Fully connected network: in_dim -> hidden layers (ReLU) -> out_dim.

    The output layer is linear (no activation); each model decides what to do
    with the raw output (tanh, softplus, ...).
    """
    layers = []
    last = in_dim
    for size in hidden_sizes:
        layers += [nn.Linear(last, size), nn.ReLU()]
        last = size
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


def soft_update(target_net, source_net, tau):
    """Polyak averaging: target <- tau * source + (1 - tau) * target."""
    with torch.no_grad():
        for t_param, s_param in zip(target_net.parameters(), source_net.parameters()):
            t_param.mul_(1.0 - tau).add_(tau * s_param)


class GaussianPolicy(nn.Module):
    """Stochastic policy for continuous actions (used by REINFORCE and A2C).

    The network outputs the mean of a Gaussian for every action dimension.
    The standard deviation is a learnable parameter that does not depend on
    the state. Calling the module returns a torch Normal distribution, from
    which we can sample actions and compute log-probabilities.
    """

    def __init__(self, obs_dim, act_dim, hidden_sizes, init_log_std=-0.5, initial_gain=None):
        super().__init__()
        self.mean_net = mlp(obs_dim, act_dim, hidden_sizes)
        if initial_gain is not None:    # a linear policy u = -K x that starts from a chosen K
            if hidden_sizes:
                raise ValueError("initial_gain needs a linear policy: use hidden_sizes=().")
            with torch.no_grad():
                self.mean_net[0].weight.fill_(-initial_gain)
                self.mean_net[0].bias.zero_()
        self.log_std = nn.Parameter(torch.full((act_dim,), init_log_std))

    def forward(self, obs):
        mean = self.mean_net(obs)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)