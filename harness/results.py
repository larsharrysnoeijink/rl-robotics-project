"""The shared results format: one results.npz per run. Plots, compare.py and notebooks read only this.

Evaluation arrays, one entry per evaluation (see harness/evaluation.py):
    eval_steps         environment steps taken so far
    eval_wall_clock    training time so far in seconds, evaluation time excluded
    eval_return        mean return over the fixed seed set
    eval_return_std    spread of that return across the seed set
    eval_cost          mean cost per episode (steps inside the obstacle)
    eval_distance      mean final distance to the target (NaN if the environment has none)
Training arrays, one entry per finished training episode:
    train_steps        environment steps taken when the episode ended
    train_return       its return, with exploration noise
    train_cost         its cost: this is what violations during training means
Plus `config`, a JSON record of how the run was launched.
"""

import json
import os
from pathlib import Path

import numpy as np

EVAL_FIELDS = ("eval_steps", "eval_wall_clock", "eval_return", "eval_return_std", "eval_cost", "eval_distance")
TRAIN_FIELDS = ("train_steps", "train_return", "train_cost")


class RunLog:
    def __init__(self, run_dir, config):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.data = {field: [] for field in EVAL_FIELDS + TRAIN_FIELDS}

    def log_evaluation(self, steps, wall_clock, summary):
        self.data["eval_steps"].append(steps)
        self.data["eval_wall_clock"].append(wall_clock)
        self.data["eval_return"].append(summary["return"])
        self.data["eval_return_std"].append(summary["return_std"])
        self.data["eval_cost"].append(summary["cost"])
        self.data["eval_distance"].append(summary["distance"])

    def log_episode(self, steps, episode_return, episode_cost):
        self.data["train_steps"].append(steps)
        self.data["train_return"].append(episode_return)
        self.data["train_cost"].append(episode_cost)

    def save(self):
        """Write atomically, so stopping a run mid-save cannot corrupt its results."""
        path = self.run_dir / "results.npz"
        temporary = self.run_dir / "results.tmp.npz"
        arrays = {key: np.asarray(values, dtype=float) for key, values in self.data.items()}
        np.savez(temporary, config=np.array(json.dumps(self.config)), **arrays)
        os.replace(temporary, path)


def load_run(path):
    """Load a results.npz, or the run folder holding one. Returns a dict with `config` as a dict."""
    path = Path(path)
    if path.is_dir():
        path = path / "results.npz"
    with np.load(path) as stored:
        run = {key: stored[key] for key in stored.files}
    run["config"] = json.loads(str(run["config"]))
    return run


def find_runs(root):
    """Every run under `root`, at any depth, so a folder of several groups' uploads also works."""
    return [load_run(path) for path in sorted(Path(root).rglob("results.npz"))]
