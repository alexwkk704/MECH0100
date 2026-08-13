"""
plot_results.py — CATEGORY-HOLDOUT charts for the forward model.

ALL settings come from ML_settings.xlsx, sheet "train" (same file train.py uses).

Outputs into charts_dir:
  holdout_<name>.png    — MAPE bars per holdout split (train on all-but-one category,
                          predict the held-out one; density / RF / GPR feature models)

The per-target 4-method comparison (density | RF | GPR | PointNet) is NOT here — it is
drawn by compare_models.py at step 7 (compare_<t>.png, sidebyside_<t>.png, bars_by_target.png),
because PointNet's predictions only exist after step 6. The old parity_<target>.png charts
(3-panel, no PointNet) were removed as redundant.

Run AFTER train.py. Requires: pandas numpy matplotlib openpyxl.
Usage:  python plot_results.py
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from train import load_cfg          # settings loader shared with train.py

PURPLE = "#500778"  # UCL purple
COLORS = {"reference_density": "0.45", "RF": "#2a7f3f", "GPR": PURPLE}
LABELS = {"reference_density": "Density-only reference (not a model)", "RF": "Random Forest", "GPR": "GPR"}


def bars(cfg, res, split, fname, title):
    d = res[res["split"] == split]
    if d.empty:
        print(f"[!] no results for split '{split}' — skipped"); return
    targets = [t for t in d["target"].unique()]
    models = [m for m in ("reference_density", "RF", "GPR") if m in set(d["model"])]
    x = np.arange(len(targets)); w = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(targets)), 4.6))
    for i, m in enumerate(models):
        vals = [d[(d.target == t) & (d.model == m)]["MAPE"].values for t in targets]
        vals = [v[0] if len(v) else np.nan for v in vals]
        b = ax.bar(x + (i - (len(models) - 1) / 2) * w, vals, w,
                   label=LABELS[m], color=COLORS[m])
        ax.bar_label(b, fmt="%.0f", fontsize=8, padding=1)
    ax.set_xticks(x, targets, rotation=15)
    ax.set_ylabel("MAPE (%)  — lower is better")
    ax.set_title(title, color=PURPLE, fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(cfg["charts_dir"], fname), dpi=160)
    plt.close(fig)
    print("[OK]", fname)


def main():
    cfg = load_cfg()
    os.makedirs(cfg["charts_dir"], exist_ok=True)
    df = pd.read_csv(cfg["dataset_csv"])
    res = pd.read_csv(cfg["results_csv"])
    # NOTE: the per-target 4-method comparison (density | RF | GPR | PointNet) is drawn by
    # compare_models.py at step 7 — compare_<t>.png (overlay), sidebyside_<t>.png (panels),
    # and bars_by_target.png (the 4-method R² bar chart). PointNet only exists after step 6,
    # so those must live there, not here. This step now produces ONLY the category-holdout
    # bars (train on all-but-one category, predict the held-out one) — a different question
    # that the density/RF/GPR feature models answer and PointNet does not run.
    for hname, col, val in cfg["holdouts"]:
        bars(cfg, res, hname, f"holdout_{hname}.png",
             f"Holdout: {hname} ({col} == {val} held out)")
    print("[DONE] holdout charts in", cfg["charts_dir"],
          "— per-target method comparison is in compare_/sidebyside_/bars_by_target (step 7)")


if __name__ == "__main__":
    main()
