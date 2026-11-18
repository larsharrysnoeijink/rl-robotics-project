"""Compare runs: a table of final results and a plot with one curve per experiment.

    python compare.py                               # everything in runs/
    python compare.py --env ScalarLQR-v0            # only this environment
    python compare.py --algos REINFORCE LQR         # only these algorithms
    python compare.py --x wall --metric cost        # cost against training time
    python compare.py --runs-dir student_files      # a folder holding many groups' runs

Plots go to outputs/comparison_<env>_<metric>_<x>.png
"""

import argparse
from pathlib import Path

import numpy as np

from harness.plots import AXES, METRICS, group_runs, plot_comparison
from harness.results import find_runs


def print_table(env_id, runs):
    print(f"\n=== {env_id} ===")
    print(f"{'experiment':<36}{'seeds':>6}{'steps':>10}{'time (s)':>10}{'return':>18}{'cost':>8}{'distance':>10}")
    for label, group in sorted(group_runs(runs).items()):
        final = np.array([run["eval_return"][-1] for run in group])
        steps = np.mean([run["eval_steps"][-1] for run in group])
        clock = np.mean([run["eval_wall_clock"][-1] for run in group])
        cost = np.mean([run["eval_cost"][-1] for run in group])
        distance = np.mean([run["eval_distance"][-1] for run in group])
        result = f"{final.mean():.1f} +- {final.std():.1f}"
        print(f"{label:<36}{len(group):>6}{steps:>10.0f}{clock:>10.1f}{result:>18}{cost:>8.1f}{distance:>10.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--env", default=None)
    parser.add_argument("--algos", nargs="*", default=None)
    parser.add_argument("--metric", choices=sorted(METRICS), default="return")
    parser.add_argument("--x", choices=sorted(AXES), default="steps")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--show", action="store_true", help="also open the plot in a window")
    args = parser.parse_args()

    runs = find_runs(args.runs_dir)
    if args.algos is not None:
        runs = [run for run in runs if run["config"]["algo"] in args.algos]
    if args.env is not None:
        runs = [run for run in runs if run["config"]["env"] == args.env]
    if not runs:
        raise SystemExit(f"No matching runs in '{args.runs_dir}'. Train something with train.py first.")

    for env_id in sorted({run["config"]["env"] for run in runs}):
        env_runs = [run for run in runs if run["config"]["env"] == env_id]
        print_table(env_id, env_runs)
        plot_comparison(env_runs, metric=args.metric, x=args.x, title=f"{env_id}: {METRICS[args.metric][1]}",
                        save_path=Path(args.out_dir) / f"comparison_{env_id}_{args.metric}_{args.x}.png",
                        show=args.show)


if __name__ == "__main__":
    main()
