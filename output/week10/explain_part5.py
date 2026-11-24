"""Figure 6: part 5, our own experiment, against base REINFORCE-BASELINE on seeds 80-82.

    python output/week10/explain_part5.py

Reads the run folders in runs/, the baseline measurements in output/week10/part5_measurements.csv
(from baseline_quality.py --checkpoints all --csv ...) and the gamma measurement in
output/week10/gamma_variance_lqr.csv (from gamma_variance.py --env ScalarLQR-v0 --csv ...).
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from harness.results import load_run

HERE = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
SEEDS = (80, 81, 82)
EXPERIMENTS = {                       # run folder stem: (short name, label, colour)
    "REINFORCE-BASELINE_base": ("base", "base: baseline_lr 1e-3, baseline (64, 64)", "tab:blue"),
    "REINFORCE-BASELINE_blr3e3": ("blr3e3", "blr3e3: baseline_lr 3e-3 (H1)", "tab:red"),
    "REINFORCE-BASELINE_blr3e4": ("blr3e4", "blr3e4: baseline_lr 3e-4 (H1)", "tab:purple"),
    "REINFORCE-BASELINE_bhid128": ("bhid128", "bhid128: baseline (128, 128) (H2)", "tab:brown"),
    "REINFORCE-BASELINE_steps600k": ("steps600k", "steps600k: base continued to 600k steps (H3)", "tab:cyan"),
    "REINFORCE_base": ("REINFORCE", "REINFORCE, no baseline (reference)", "tab:gray"),
}
# steps600k repeats the base runs exactly up to 300 000 steps (same seeds, deterministic training), so
# its checkpoints at 100k, 200k and 300k are the base runs' checkpoints too; base itself saved none.
HANGING_DISTANCE = 1.638               # fingertip to target, arm hanging still
HANGING = -HANGING_DISTANCE * 200      # return of an arm that hangs still for 200 steps


def runs_of(stem):
    return [load_run(str(RUNS / f"PlanarArm-v0_{stem}_seed{s}")) for s in SEEDS]


def measurements():
    rows = defaultdict(list)
    with open(HERE / "part5_measurements.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row["variation"]].append({k: (float(v) if k not in ("run", "variation", "checkpoint") else v)
                                           for k, v in row.items()})
    return rows


def per_seed(a, field, title, ylabel, reference=None, reference_text=None):
    """Final evaluation value per seed, one column per experiment, bar at the mean."""
    for i, (stem, (short, _, colour)) in enumerate(EXPERIMENTS.items()):
        finals = [r[field][-1] for r in data[stem]]
        a.scatter([i] * 3, finals, color=colour, s=55, zorder=3)
        for s, y in zip(SEEDS, finals):
            a.annotate(str(s), (i, y), (i + 0.1, y), fontsize=7.5, va="center")
        a.plot([i - 0.28, i + 0.28], [np.mean(finals)] * 2, color=colour, lw=3)
    if reference is not None:
        a.axhline(reference, color="gray", ls=":", lw=1)
        a.text(len(EXPERIMENTS) - 0.5, reference, reference_text, color="gray", fontsize=8, ha="right", va="bottom")
    a.set_xticks(range(len(EXPERIMENTS)), [v[0] for v in EXPERIMENTS.values()], fontsize=8.5)
    a.set_title(title); a.set_ylabel(ylabel)


def over_training(a, value, title, ylabel, log=False):
    """A measured quantity at the checkpoints: dots per seed, line through the median of the seeds."""
    for stem, (short, label, colour) in EXPERIMENTS.items():
        rows = m.get(short, [])
        if not rows:
            continue
        steps = sorted({r["step"] for r in rows})
        a.scatter([r["step"] for r in rows], [value(r) for r in rows], color=colour, s=12, alpha=0.5)
        a.plot(steps, [np.median([value(r) for r in rows if r["step"] == s]) for s in steps],
               color=colour, lw=2, marker="o", ms=4, label=label)
    a.set_title(title); a.set_xlabel("environment steps (checkpoint)"); a.set_ylabel(ylabel)
    if log:
        a.set_yscale("log")


data = {stem: runs_of(stem) for stem in EXPERIMENTS}
m = measurements()
fig, ax = plt.subplots(3, 3, figsize=(20, 16))

# Row 1: what the policy reaches.
a = ax[0, 0]
order = ["REINFORCE-BASELINE_steps600k"] + [s for s in EXPERIMENTS if s != "REINFORCE-BASELINE_steps600k"]
for stem in order:           # steps600k first, so the identical base curve stays visible on top of it
    short, label, colour = EXPERIMENTS[stem]
    steps = data[stem][0]["eval_steps"]
    curves = np.array([r["eval_return"] for r in data[stem]])
    a.plot(steps, curves.mean(axis=0), color=colour, lw=3 if short == "steps600k" else 2, label=label,
           ls="--" if short == "REINFORCE" else "-")
a.axhline(HANGING, color="gray", ls=":", lw=1)
a.text(10_000, HANGING + 6, "hanging still", color="gray", fontsize=8)
a.set_title("a. Evaluation return over training, mean of 3 seeds\n(from hanging, no noise; the seeds are in b to d)")
a.set_xlabel("environment steps"); a.set_ylabel("return"); a.legend(fontsize=7.5, loc="lower right")

per_seed(ax[0, 1], "eval_return", "b. Final evaluation return per seed (bar: mean)", "return", HANGING, "hanging still")
per_seed(ax[0, 2], "eval_distance", "c. Final distance to the target per seed", "distance (m)",
         HANGING_DISTANCE, "hanging pose")

# Row 2: the cost, and how good the baseline is.
per_seed(ax[1, 0], "eval_cost", "d. Steps in the obstacle per seed (out of 200)", "cost")
over_training(ax[1, 1], lambda r: r["r2_own"],
              "e. How well the run's own baseline predicts G_t: R²\n(1 = perfect, 0 = a constant; line: median of 3 seeds)", "R²")
ax[1, 1].legend(fontsize=7.5, loc="lower left")
over_training(ax[1, 2], lambda r: r["var_own"] / r["var_reward_to_go"],
              "f. Gradient variance with the run's baseline,\nas a fraction of the variance without one (lower = baseline helps more)",
              "variance ratio")

# Row 3: the variance over training, the exploration noise, and the gamma check.
a = ax[2, 0]                 # H3: the long runs only, per seed, with and without their baseline
for s, colour in zip(SEEDS, ("tab:cyan", "tab:olive", "tab:pink")):
    rows = sorted((r for r in m["steps600k"] if int(r["seed"]) == s), key=lambda r: r["step"])
    steps = [r["step"] for r in rows]
    a.plot(steps, [r["var_reward_to_go"] for r in rows], "o--", color=colour, lw=1.5, ms=4,
           label=f"seed {s}, no baseline")
    a.plot(steps, [r["var_own"] for r in rows], "o-", color=colour, lw=2.5, ms=4,
           label=f"seed {s}, with the run's baseline")
a.set_yscale("log")
a.set_title("g. H3: gradient variance over training, steps600k\n(one episode, summed over all parameters)")
a.set_xlabel("environment steps (checkpoint)"); a.set_ylabel("variance (log scale)"); a.legend(fontsize=7.5, ncol=2)
over_training(ax[2, 1], lambda r: (r["sigma1"] + r["sigma2"]) / 2,
              "h. Exploration noise σ of the policy\n(mean of the two joints; starts at 0.61)", "σ")
ax[2, 1].axhline(np.exp(-0.5), color="gray", ls=":", lw=1)

a = ax[2, 2]
with open(HERE / "gamma_variance_lqr.csv", encoding="utf-8") as f:
    g = list(csv.DictReader(f))
gammas = [float(r["gamma"]) for r in g]
x = np.arange(len(gammas))
a.plot(x, [float(r["reward_to_go"]) for r in g], "o-", color="tab:orange", lw=2, label="reward-to-go, no baseline")
a.plot(x, [float(r["baseline"]) for r in g], "o-", color="tab:blue", lw=2, label="with a fitted baseline")
a.set_xticks(x, [f"{v:g}" for v in gammas])
a.set_title("i. H4: relative variance against the discount γ\n(scalar LQR, policy K = 0.6, 3 000 episodes each)")
a.set_xlabel("γ"); a.set_ylabel("relative variance"); a.legend(fontsize=8)

fig.suptitle("Figure 6. Part 5 on the arm: a different baseline, or longer training, against base REINFORCE-BASELINE "
             "(seeds 80-82)", fontsize=14)
fig.tight_layout()
fig.savefig(HERE / "fig6_part5.png", dpi=95)
print("saved", HERE / "fig6_part5.png")
