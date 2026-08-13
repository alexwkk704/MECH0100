#!/usr/bin/env python3
"""holdout_charts.py — holdout R2 charts, one set per run.

The holdout charts the pipeline writes show MAPE only. MAPE has no zero point,
so it cannot tell you whether a model is worse than simply predicting the mean -
which on an extrapolation or unseen-family holdout is the entire question. R2
can, because it goes negative. This builds the R2 versions from data already on
disk. Nothing is retrained.

Reads, both already written by the pipeline:
    model_results.csv        RF / GPR / reference_density, every split
    pointnet_holdouts.csv    PointNet, holdout splits (older runs lack this)

Writes into <run>/charts/ :
    holdoutR2_<split>.png    one chart per holdout split
    holdoutR2_table.csv      the numbers behind them

Holdout R2 can reach -60, so the axis is clipped at -2; a bar below the floor is
drawn to the floor, marked with a red triangle and labelled with its true value.
The clipping is stated on the figure.

Usage:
    python holdout_charts.py --run runs/COMBINED/20260811_0800   # one run
    python holdout_charts.py --profile COMBINED                  # every run
    python holdout_charts.py --profile ADMS
"""
import argparse, os, sys, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)

STYLE = [("reference_density", "Density-only reference (not a model)", "#9a9a9a"),
         ("RF",                "Random Forest",                        "#2e7d32"),
         ("GPR",               "Gaussian Process",                     "#5b2d8e"),
         ("PointNet_v2",       "PointNet v2",                          "#d1451b")]
FLOOR = -2.0


def load_run(run):
    frames = []
    for f in ("model_results.csv", "pointnet_holdouts.csv"):
        p = os.path.join(run, f)
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    if not frames:
        return None
    d = pd.concat(frames, ignore_index=True)
    return d[~d.split.astype(str).str.contains("grouped", case=False, na=False)]


def chart_split(g, split, run_name, out):
    targets = sorted(g.target.unique())
    models = [m for m in STYLE if (g.model == m[0]).any()]
    if not targets or not models:
        return None
    x = np.arange(len(targets)); w = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(max(8, 1.5 * len(targets)), 4.6))
    for i, (key, lab, col) in enumerate(models):
        vals = np.array([g[(g.target == t) & (g.model == key)].R2.mean()
                         for t in targets], float)
        pos = x + (i - (len(models) - 1) / 2) * w
        bars = ax.bar(pos, np.where(np.isfinite(vals), np.maximum(vals, FLOOR), np.nan),
                      w, color=col, label=lab, edgecolor="white", linewidth=.4)
        for rect, real in zip(bars, vals):
            if not np.isfinite(real):
                continue
            clipped = real < FLOOR
            y = max(real, FLOOR)
            ax.text(rect.get_x() + rect.get_width() / 2,
                    y + (0.04 if y >= 0 else -0.04),
                    ("▼ " if clipped else "") + f"{real:.2f}",
                    ha="center", va="bottom" if y >= 0 else "top", fontsize=6.2,
                    color="#b00000" if clipped else "black",
                    weight="bold" if clipped else "normal")
    n = int(g.N_test.max()) if "N_test" in g else 0
    ax.axhline(0, color="#444444", lw=1.0)
    ax.set_xticks(x); ax.set_xticklabels(targets, rotation=20, ha="right")
    ax.set_ylabel(r"$R^2$   (0 = no better than the mean)")
    ax.set_ylim(FLOOR - 0.25, 1.15)
    ax.set_title(f"Holdout R$^2$: {split}   (N = {n})\n{run_name}",
                 fontsize=11, weight="bold")
    ax.grid(axis="y", alpha=.25, lw=.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, ncol=2,
              loc="upper center", bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.text(0.995, 0.008,
             f"axis clipped at {FLOOR:g}; ▼ = bar drawn to the floor, true value printed",
             ha="right", va="bottom", fontsize=6.5, color="#888888")
    f = os.path.join(out, f"holdoutR2_{split}.png")
    fig.savefig(f, dpi=200); plt.close(fig)
    return f


def do_run(run):
    d = load_run(run)
    name = os.path.basename(run.rstrip("/\\"))
    if d is None or d.empty:
        print(f"[skip] {name}  (no holdout results on disk)")
        return
    out = os.path.join(run, "charts")
    os.makedirs(out, exist_ok=True)
    made = [chart_split(d[d.split == s], s, name, out) for s in sorted(d.split.unique())]
    d.to_csv(os.path.join(out, "holdoutR2_table.csv"), index=False)
    have = sorted({m for m, _, _ in STYLE if (d.model == m).any()})
    print(f"[ok] {run}  ->  charts\\")
    for f in made:
        if f:
            print("       ", os.path.basename(f))
    print("        holdoutR2_table.csv")
    if "PointNet_v2" not in have:
        print("        note: no pointnet_holdouts.csv in this run - "
              "charts show RF / GPR / density only")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="a single run folder")
    ap.add_argument("--profile", default="COMBINED",
                    help="do every run under runs/<PROFILE> (ignored if --run given)")
    a = ap.parse_args()

    if a.run:
        do_run(a.run)
        return
    runs = sorted(glob.glob(os.path.join(BASE, "runs", a.profile, "2026*")))
    if not runs:
        sys.exit(f"[E] no runs under runs/{a.profile}")
    print(f"{len(runs)} run(s) under runs/{a.profile}\n")
    for r in runs:
        do_run(r); print()


if __name__ == "__main__":
    main()
