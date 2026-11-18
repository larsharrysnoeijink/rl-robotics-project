"""Comparison plots: mean and spread across seeds, against environment steps by default."""

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

METRICS = {"return": ("eval_return", "Evaluation return"),
           "cost": ("eval_cost", "Cost per episode"),
           "distance": ("eval_distance", "Final distance to target")}
AXES = {"steps": ("eval_steps", "Environment steps"), "wall": ("eval_wall_clock", "Training time (s)")}


def run_label(run):
    """Runs with the same label are seeds of the same experiment."""
    config = run["config"]
    variation = config.get("variation", "base")
    return config["algo"] if variation == "base" else f"{config['algo']} ({variation})"


def group_runs(runs):
    groups = defaultdict(list)
    for run in runs:
        groups[run_label(run)].append(run)
    return dict(groups)


def _mean_and_spread(runs, x_field, y_field):
    """Interpolate every seed onto one grid, covering only the range all of them reached."""
    end = min(run[x_field][-1] for run in runs)
    grid = np.linspace(0.0, end, 100) if end > 0 else np.array([0.0])
    curves = np.array([np.interp(grid, run[x_field], run[y_field]) for run in runs])
    return grid, curves.mean(axis=0), curves.std(axis=0)


def plot_comparison(runs, metric="return", x="steps", title=None, save_path=None, show=False):
    """One curve per experiment, with a band of one standard deviation across its seeds.

    A baseline with nothing to train has a single point and is drawn as a horizontal line.
    """
    y_field, y_label = METRICS[metric]
    x_field, x_label = AXES[x]
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (label, group) in enumerate(sorted(group_runs(runs).items())):
        color = f"C{i}"
        legend = f"{label}, {len(group)} seed{'s' if len(group) > 1 else ''}"
        if all(len(run[y_field]) == 1 for run in group):
            ax.axhline(np.mean([run[y_field][0] for run in group]), color=color, linestyle="--", label=legend)
            continue
        grid, mean, spread = _mean_and_spread(group, x_field, y_field)
        ax.plot(grid, mean, color=color, linewidth=2, label=legend)
        if len(group) > 1:
            ax.fill_between(grid, mean - spread, mean + spread, color=color, alpha=0.2)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title or y_label)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_fingertip_path(path, target, obstacle_centre, obstacle_radius, title="Fingertip path",
                        save_path=None, show=False):
    """The fingertip's way through the plane, with the target and the obstacle."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.add_patch(plt.Circle(obstacle_centre, obstacle_radius, color="C3", alpha=0.4, label="obstacle"))
    ax.plot(path[:, 0], path[:, 1], color="C0", label="fingertip")
    ax.plot(*path[0], "o", color="C0")
    ax.plot(*target, "*", color="C2", markersize=14, label="target")
    ax.set_aspect("equal")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    if show:
        plt.show()
    plt.close(fig)
