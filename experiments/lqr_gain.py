"""Read the gain K of a linear policy u = -K x along training, against the Riccati answer K*.

    python train.py --algo REINFORCE --env ScalarLQR-v0 --steps 150000 --checkpoint-every 10000
        --set "hidden_sizes=()" gamma=0.98 initial_gain=0.6
    python experiments/lqr_gain.py runs/ScalarLQR-v0_REINFORCE_base_seed*

Plots to outputs/lqr_gain.png, with one line per run folder.
"""

import argparse
import glob
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from envs.scalar_lqr import riccati_solution  # noqa: E402
from harness.results import load_run  # noqa: E402


def read_gain(checkpoint):
    """K from a saved linear policy u = -K x."""
    state = torch.load(checkpoint, map_location="cpu")["policy"]
    return -state["mean_net.0.weight"].item()


def gain_history(run_dir):
    """(steps, gains) from the checkpoints in a run folder, final one last."""
    run_dir = Path(run_dir)
    pairs = [(int(path.stem.split("_")[1]), read_gain(path)) for path in sorted(run_dir.glob("step_*.pt"))]
    steps = int(load_run(run_dir)["eval_steps"][-1])
    if not pairs or pairs[-1][0] != steps:
        pairs.append((steps, read_gain(run_dir / "final.pt")))
    return [p[0] for p in pairs], [p[1] for p in pairs]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="run folders")
    parser.add_argument("--out", default="outputs/lqr_gain.png")
    args = parser.parse_args()

    _, K_star = riccati_solution()
    folders = [folder for pattern in args.runs for folder in (glob.glob(pattern) or [pattern])]
    fig, ax = plt.subplots(figsize=(8, 5))
    print(f"K* = {K_star:.3f}")
    for folder in folders:
        steps, gains = gain_history(folder)
        print(f"{Path(folder).name:<50} final K = {gains[-1]:.3f}")
        ax.plot(steps, gains, label=Path(folder).name)
    ax.axhline(K_star, color="k", linestyle="--", label="K* (Riccati)")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Gain K")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Plot saved to {args.out}")


if __name__ == "__main__":
    main()
