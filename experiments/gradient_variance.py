"""Measure the variance of the policy-gradient estimate at a frozen policy.

    python experiments/gradient_variance.py --run runs/ScalarLQR-v0_REINFORCE_base_seed0

Everything is measured at the policy saved in the run, so all differences come from the estimator and
none from learning. Episodes are split in two: the baseline is fitted on the first half, and the
estimators are measured on the second, because the identity is exact only for a baseline built from
other episodes (Week 9 deck). For each episode this builds:

    full return    sum_t gamma^t G_0 grad log pi(u_t | x_t)
    reward-to-go   sum_t gamma^t G_t grad log pi(u_t | x_t)
    baseline       sum_t gamma^t (G_t - b(x_t)) grad log pi(u_t | x_t)

and reports the relative variance (E||g||^2 - ||E g||^2) / ||E g||^2, with how far each mean lies from
the full-return mean in standard errors. It also sweeps a constant b through the reward-to-go estimator.
The results go to outputs/gradient_variance_<env>.npz.
"""

import argparse
import ast
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import envs  # noqa: E402,F401  (registers the project environments)
import gymnasium as gym  # noqa: E402
from harness.registry import build_agent  # noqa: E402
from harness.results import load_run  # noqa: E402
from models.reinforce import reward_to_go  # noqa: E402
from models.reinforce_baseline import baseline_delta, baseline_loss  # noqa: E402
from utils.networks import mlp  # noqa: E402
from utils.seeding import set_seed  # noqa: E402


def collect_episode(env, policy, seed=None):
    """One episode with the stochastic policy. Returns arrays of states, actions, rewards."""
    obs, _ = env.reset(seed=seed)
    states, actions, rewards = [], [], []
    done = False
    low, high = env.action_space.low, env.action_space.high
    while not done:
        with torch.no_grad():
            u = policy(torch.as_tensor(obs, dtype=torch.float32)).sample().numpy()
        states.append(obs)
        actions.append(u)
        obs, reward, terminated, truncated, _ = env.step(np.clip(u, low, high))
        rewards.append(float(reward))
        done = terminated or truncated
    return (torch.as_tensor(np.array(states), dtype=torch.float32),
            torch.as_tensor(np.array(actions), dtype=torch.float32), rewards)


def fit_baseline(episodes, gamma, hidden_sizes, steps=2000, lr=1e-3, batch=512):
    """Regression of a fresh baseline network on the observed returns, by the C1 baseline loss."""
    x = torch.cat([states for states, _, _ in episodes])
    G = torch.cat([torch.as_tensor(reward_to_go(rewards, gamma)) for _, _, rewards in episodes])
    scale = G.std().clamp(min=1e-6)
    network = mlp(x.shape[1], 1, hidden_sizes)
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    for _ in range(steps):
        index = torch.randint(0, len(x), (min(batch, len(x)),))
        loss = baseline_loss(G[index] / scale, network(x[index]).squeeze(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return lambda states: network(states).squeeze(-1) * scale


def episode_gradient(policy, states, actions, weights, gamma):
    """sum_t gamma^t weights_t grad log pi(u_t | x_t), flattened into one vector."""
    log_probs = policy(states).log_prob(actions).sum(dim=-1)
    discounts = gamma ** torch.arange(len(weights), dtype=torch.float32)
    gradients = torch.autograd.grad((discounts * weights * log_probs).sum(), list(policy.parameters()))
    return torch.cat([g.flatten() for g in gradients]).numpy().astype(np.float64)


def relative_variance(samples):
    """(E||g||^2 - ||E g||^2) / ||E g||^2 from an array of shape (episodes, parameters)."""
    mean = samples.mean(axis=0)
    return float((np.mean(np.sum(samples**2, axis=1)) - np.sum(mean**2)) / np.sum(mean**2))


def measure(policy, env, baseline_episodes, measure_episodes, gamma, hidden_sizes, seed):
    set_seed(seed)
    env.reset(seed=seed)
    fit_set = [collect_episode(env, policy) for _ in range(baseline_episodes)]
    baseline = fit_baseline(fit_set, gamma, hidden_sizes)

    full, to_go, centred, score = [], [], [], []
    for _ in range(measure_episodes):
        states, actions, rewards = collect_episode(env, policy)
        G = torch.as_tensor(reward_to_go(rewards, gamma))
        with torch.no_grad():
            delta = baseline_delta(G, baseline(states))
        full.append(episode_gradient(policy, states, actions, torch.full_like(G, G[0].item()), gamma))
        to_go.append(episode_gradient(policy, states, actions, G, gamma))
        centred.append(episode_gradient(policy, states, actions, delta, gamma))
        score.append(episode_gradient(policy, states, actions, torch.ones_like(G), gamma))
    return {"full return": np.array(full), "reward-to-go": np.array(to_go), "baseline": np.array(centred),
            "score": np.array(score)}


def constant_baseline_sweep(to_go, score, n=21):
    """The estimator with constant b is A - b S, so the whole sweep needs no new episodes."""
    best = np.sum(to_go * score) / np.sum(score * score)
    span = max(abs(best), 1.0)
    values = np.linspace(best - span, best + span, n)
    variances = np.array([relative_variance(to_go - b * score) for b in values])
    return values, variances, best


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="run folder holding final.pt")
    parser.add_argument("--episodes", type=int, default=1000, help="episodes for the measurement")
    parser.add_argument("--baseline-episodes", type=int, default=200, help="episodes for fitting the baseline")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="outputs")
    args = parser.parse_args()

    torch.set_num_threads(1)
    config = load_run(args.run)["config"]
    if config["algo"] not in ("REINFORCE", "REINFORCE-BASELINE"):
        raise SystemExit("This tool measures the two REINFORCE agents.")
    stored = {key: ast.literal_eval(value) for key, value in config["hyperparameters"].items()}
    env = gym.make(config["env"])
    agent, _ = build_agent(config["algo"], env, stored)
    agent.load(Path(args.run) / "final.pt")

    results = measure(agent.policy, env, args.baseline_episodes, args.episodes, stored["gamma"],
                      (64, 64), args.seed)
    env.close()

    reference = results["full return"].mean(axis=0)
    error = results["full return"].std(axis=0, ddof=1) / np.sqrt(args.episodes)
    print(f"{config['env']}, policy of {args.run}, {args.episodes} episodes")
    print(f"{'estimator':<16}{'relative variance':>20}{'parameters beyond 2 s.e.':>28}")
    summary, shifted = {}, {}
    for name in ("full return", "reward-to-go", "baseline"):
        samples = results[name]
        # How many parameters have a mean that differs from the full-return mean by more than two
        # combined standard errors. If the mean is unchanged, this is about 5%.
        combined = np.sqrt(error**2 + samples.var(axis=0, ddof=1) / args.episodes)
        shift = np.abs(samples.mean(axis=0) - reference) / np.maximum(combined, 1e-12)
        summary[name] = relative_variance(samples)
        shifted[name] = float(np.mean(shift > 2.0))
        print(f"{name:<16}{summary[name]:>20.1f}{100 * shifted[name]:>27.0f}%")

    values, variances, best = constant_baseline_sweep(results["reward-to-go"], results["score"])
    print(f"Constant b: best at {best:.2f}, relative variance {variances.min():.1f}")

    # The mean of the estimator at each b, as a distance from the mean at b = 0, in standard errors.
    score_mean = results["score"].mean(axis=0)
    mean_error = np.linalg.norm(results["reward-to-go"].std(axis=0, ddof=1)) / np.sqrt(args.episodes)
    mean_shift = np.abs(values) * np.linalg.norm(score_mean) / mean_error

    path = Path(args.out_dir) / f"gradient_variance_{config['env']}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, b_values=values, b_variances=variances, b_best=best, b_mean_shift=mean_shift,
             full_return_shifted=shifted["full return"], reward_to_go_shifted=shifted["reward-to-go"],
             baseline_shifted=shifted["baseline"],
             full_return=summary["full return"], reward_to_go=summary["reward-to-go"],
             baseline=summary["baseline"], episodes=args.episodes)
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
