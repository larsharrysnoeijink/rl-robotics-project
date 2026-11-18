"""Finds the algorithms in models/ and baselines/.

A file provides an algorithm by defining ALGO, DEFAULTS and AGENT_CLASS at module level, so adding
one means adding one file. Modules are imported only when selected.
"""

import importlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_ALGO_LINE = re.compile(r'^ALGO = "([^"]+)"', re.MULTILINE)


def available_algorithms():
    """{algorithm name: module name} for every file that declares an ALGO."""
    found = {}
    for package in ("models", "baselines"):
        for path in sorted((ROOT / package).glob("*.py")):
            match = _ALGO_LINE.search(path.read_text(encoding="utf-8"))
            if match:
                found[match.group(1)] = f"{package}.{path.stem}"
    return found


def load_module(algo):
    algorithms = available_algorithms()
    if algo not in algorithms:
        raise SystemExit(f"Unknown algorithm '{algo}'. Available: {', '.join(sorted(algorithms))}")
    return importlib.import_module(algorithms[algo])


def resolved_hyperparameters(module, overrides=None):
    """The module's DEFAULTS with `overrides` applied. A name it does not know is an error."""
    params = dict(module.DEFAULTS)
    for key, value in (overrides or {}).items():
        if key not in params:
            raise SystemExit(f"{module.ALGO} has no hyperparameter '{key}'. It has: {', '.join(params)}")
        params[key] = value
    return params


def build_agent(algo, env, overrides=None, device="cpu"):
    """Create an agent from the environment's sizes. Returns (agent, the hyperparameters used)."""
    module = load_module(algo)
    params = resolved_hyperparameters(module, overrides)
    agent = module.AGENT_CLASS(
        obs_dim=env.observation_space.shape[0],
        act_dim=env.action_space.shape[0],
        act_low=env.action_space.low,
        act_high=env.action_space.high,
        device=device,
        **params,
    )
    return agent, params


def is_trainable(algo):
    return getattr(load_module(algo), "TRAINABLE", True)
