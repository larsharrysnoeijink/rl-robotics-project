"""Figure 6: part 5, our own experiment, against base REINFORCE-BASELINE on seeds 80-82.

    python output/week10/explain_part5.py

Reads the run folders in runs/ and the baseline measurements in output/week10/part5_measurements.csv
(written by baseline_quality.py --checkpoints all --csv ...).
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
EXPERIMENTS = {                       # variation: (label, colour)
    "base": ("base: baseline_lr 1e-3, (64, 64)", "tab:blue"),
    "blr3e3": ("blr3e3: baseline_lr 3e-3", "tab:red"),
    "blr3e4": ("blr3e4: baseline_lr 3e-4", "tab:purple"),
    "bhid128": ("bhid128: baseline (128, 128)", "tab:brown"),
    "steps600k": ("steps600k: base continued to 600 000 steps", "tab:cyan"),
}
# steps600k repeats the base runs exactly up to 300 000 steps (same seeds, deterministic training), so
# its checkpoints at 100k, 200k and 300k are the base runs' checkpoints too; base itself saved none.
HANGING = -1.638 * 200                # return of an arm that hangs still for 200 steps


def runs_of(variation):
    return [load_run(str(RUNS / f"PlanarArm-v0_REINFORCE-BASELINE_{variation}_seed{s}")) for s in SEEDS]


def measurements():
    rows = defaultdict(list)
    with open(HERE / "part5_measurements.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            step = int(row["step"]) if int(row["step"]) > 0 else 300_000      # final.pt of a 300k run
            rows[row["variation"]].append({**row, "step": step})
    return rows


fig, ax = plt.subplots(2, 3, figsize=(19, 10.5))
data = {v: runs_of(v) for v in EXPERIMENTS}

# a. Evaluation return over training.
a = ax[0, 0]
for v, (label, colour) in EXPERIMENTS.items():
    steps = data[v][0]["eval_steps"]
    curves = np.array([r["eval_return"] for r in data[v]])
    a.plot(steps, curves.mean(axis=0), color=colour, lw=2, label=label)
    a.fill_between(steps, curves.min(axis=0), curves.max(axis=0), color=colour, alpha=0.12)
a.axhline(HANGING, color="gray", ls=":", lw=1)
a.text(5000, HANGING + 8, "hanging still", color="gray", fontsize=8)
a.set_title("a. Evaluation return (from hanging, no noise)\nline: mean of 3 seeds, band: min to max")
a.set_xlabel("environment steps"); a.set_ylabel("return"); a.legend(fontsize=8, loc="lower right")

# b. Final evaluation per seed.
a = ax[0, 1]
for i, (v, (label, colour)) in enumerate(EXPERIMENTS.items()):
    finals = [r["eval_return"][-1] for r in data[v]]
    at300 = [r["eval_return"][list(r["eval_steps"]).index(300_000)] for r in data[v]]
    a.scatter([i] * 3, finals, color=colour, s=60, zorder=3)
    for s, y in zip(SEEDS, finals):
        a.annotate(str(s), (i, y), (i + 0.08, y), fontsize=8, va="center")
    a.plot([i - 0.25, i + 0.25], [np.mean(finals)] * 2, color=colour, lw=3)
    if v == "steps600k":
        a.scatter([i] * 3, at300, facecolor="none", edgecolor=colour, s=60, zorder=3, label="steps600k at 300 000 steps")
a.axhline(HANGING, color="gray", ls=":", lw=1)
a.set_xticks(range(len(EXPERIMENTS)), list(EXPERIMENTS), rotation=0)
a.set_title("b. Final evaluation return per seed (bar: mean)")
a.set_ylabel("return"); a.legend(fontsize=8, loc="lower right")

# c. Final distance per seed.
a = ax[0, 2]
for i, (v, (label, colour)) in enumerate(EXPERIMENTS.items()):
    finals = [r["eval_distance"][-1] for r in data[v]]
    a.scatter([i] * 3, finals, color=colour, s=60, zorder=3)
    for s, y in zip(SEEDS, finals):
        a.annotate(str(s), (i, y), (i + 0.08, y), fontsize=8, va="center")
a.axhline(1.638, color="gray", ls=":", lw=1)
a.text(-0.4, 1.66, "hanging pose", color="gray", fontsize=8)
a.set_xticks(range(len(EXPERIMENTS)), list(EXPERIMENTS))
a.set_title("c. Final distance to the target per seed")
a.set_ylabel("distance (m)")

# d-f. The baseline and the variance, measured at checkpoints.
m = measurements()
panels = (
    (ax[1, 0], lambda r: float(r["r2_own"]), "d. How well the run's own baseline fits: R²\n(1 = perfect, 0 = a constant)", "R²", False),
    (ax[1, 1], lambda r: float(r["var_own"]) / float(r["var_reward_to_go"]),
     "e. Gradient variance with the run's baseline,\nas a fraction of the variance without a baseline", "variance ratio", False),
    (ax[1, 2], lambda r: float(r["var_own"]),
     "f. Gradient variance with the run's baseline\n(single episode, summed over all parameters)", "variance (log scale)", True),
)
for a, value, title, ylabel, log in panels:
    for v, (label, colour) in EXPERIMENTS.items():
        rows = sorted(m.get(v, []), key=lambda r: r["step"])
        if not rows:
            continue
        steps = sorted({r["step"] for r in rows})
        a.scatter([r["step"] for r in rows], [value(r) for r in rows], color=colour, s=14, alpha=0.6)
        a.plot(steps, [np.mean([value(r) for r in rows if r["step"] == s]) for s in steps],
               color=colour, lw=2, marker="o", label=label)
    a.set_title(title); a.set_xlabel("environment steps (checkpoint)"); a.set_ylabel(ylabel)
    if log:
        a.set_yscale("log")
ax[1, 0].legend(fontsize=8, loc="lower right")
ax[1, 1].axhline(1, color="gray", ls=":", lw=1)

fig.suptitle("Figure 6. Part 5: REINFORCE-BASELINE on the arm with a different baseline, or for longer (seeds 80-82)", fontsize=13)
fig.tight_layout()
fig.savefig(HERE / "fig6_part5.png", dpi=105)
print("saved", HERE / "fig6_part5.png")
