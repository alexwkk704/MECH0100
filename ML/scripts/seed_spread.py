#!/usr/bin/env python3
"""seed_spread.py — how much of the PointNet score is training noise?

Scores every runs/<PROFILE>/* folder that contains pointnet_v2_predictions.csv,
per target and PER FOLD, and reports the spread across runs.

The point: a single cross-validation score is one draw. Weight initialisation,
epoch shuffling, random augmentation and best-epoch selection all vary between
runs, and on a dataset this small the spread can be large. Reporting mean and
range across seeds is the honest version of "the model scores X".

It also writes two figures (2026-08-11):

  seed_spread_by_target.png   PointNet R2 by target, ONE BAR PER SEED. This is
                              the honest picture of a neural-net score: the bar
                              height you quote depends on which draw you took.
  methods_seedmean.png        The four-method comparison, but PointNet is the
                              MEAN over seeds with a min-max error bar, so it is
                              compared against RF / GPR / density on equal terms
                              instead of one draw against three deterministic
                              models.

RF, GPR and the density reference are deterministic given the folds, and the
seed sweep only changes the pointnet sheet, so those three bars are read from
the reference run - the newest run folder holding compare_models_same_split.csv
(or model_results.csv). Nothing is recomputed for them.

Usage:
    python seed_spread.py                       # all runs under runs/ADMS
    python seed_spread.py --profile COMBINED
    python seed_spread.py --runs runs/ADMS/a runs/ADMS/b
    python seed_spread.py --profile COMBINED --out runs/COMBINED/_seed_summary
    python seed_spread.py --no-charts
"""
import argparse, glob, os, re, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ss = ((y - y.mean()) ** 2).sum()
    return np.nan if ss == 0 else 1.0 - ((y - p) ** 2).sum() / ss


def load(run_dir):
    f = os.path.join(run_dir, "pointnet_v2_predictions.csv")
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f)
    folds = os.path.join(run_dir, "cv_folds.csv")
    if os.path.exists(folds):
        d = d.merge(pd.read_csv(folds), on="Run", how="left")
    return d


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
REF_COLS = {"reference_density": ("Density-only reference (not a model)", "#9a9a9a"),
            "RF":                ("Random Forest",                        "#2e7d32"),
            "GPR":               ("Gaussian Process",                     "#5b2d8e")}
PN_COLOUR = "#d1451b"


def _seed_labels(names):
    """Folder name -> "seed N". 20260808_0118_seed3 -> "seed 3"; a run with no
    _seedN suffix is the original seed-1 run."""
    out, used = [], set()
    for nm in names:
        m = re.search(r"seed[_-]?(\d+)", nm, re.I)
        lab = f"seed {int(m.group(1))}" if m else "seed 1"
        if lab in used:
            i = 2
            while f"{lab} ({i})" in used:
                i += 1
            lab = f"{lab} ({i})"
        used.add(lab)
        out.append(lab)
    return out


def _label(ax, bars, fmt="{:.2f}", size=6.5, tops=None):
    """Value labels. `tops` lets a bar that carries an error bar put its label
    above the whisker instead of inside it."""
    for i, b in enumerate(bars):
        h = b.get_height()
        if not np.isfinite(h):
            continue
        y = h if tops is None else max(h, tops[i])
        if h < 0:
            ax.text(b.get_x() + b.get_width() / 2, h - 0.02, fmt.format(h),
                    ha="center", va="top", fontsize=size)
        else:
            ax.text(b.get_x() + b.get_width() / 2, y + 0.015, fmt.format(h),
                    ha="center", va="bottom", fontsize=size)


def _find_reference_run(runs):
    """Newest run folder that has the RF / GPR / density numbers in it."""
    for r in sorted(runs, reverse=True):
        for f in ("compare_models_same_split.csv", "model_results.csv"):
            p = os.path.join(r, f)
            if os.path.exists(p):
                return p
    return None


def _reference_scores(path):
    """{target: {model: R2}} for the grouped-CV split only."""
    d = pd.read_csv(path)
    if "split" in d.columns:
        d = d[d["split"].astype(str).str.contains("grouped", case=False, na=False)]
    out = {}
    for _, r in d.iterrows():
        if r["model"] in REF_COLS:
            out.setdefault(r["target"], {})[r["model"]] = float(r["R2"])
    return out


def make_charts(summary, names, targets, profile, runs, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(targets))

    # ---- 1. one bar per seed -------------------------------------------
    n = len(names)
    w = 0.8 / n
    fig, ax = plt.subplots(figsize=(max(9, 1.5 * len(targets)), 4.8))
    shades = plt.cm.viridis(np.linspace(0.15, 0.8, n))
    labels = _seed_labels(names)
    for i, nm in enumerate(names):
        vals = summary[f"run::{nm}"].values
        b = ax.bar(x + (i - (n - 1) / 2) * w, vals, w, color=shades[i],
                   label=labels[i], edgecolor="white", linewidth=.4)
        _label(ax, b)
    ax.set_xticks(x); ax.set_xticklabels(targets, rotation=20, ha="right")
    ax.set_ylabel(r"$R^2$  (out-of-fold, higher is better)")
    ax.set_ylim(min(0, float(np.nanmin(summary[[f"run::{m}" for m in names]].values)) - .1), 1.05)
    ax.axhline(0, color="#444444", lw=.8)
    ax.set_title(f"PointNet v2 by target — one bar per seed  ({profile}, grouped 5-fold CV)",
                 fontsize=11, weight="bold")
    ax.grid(axis="y", alpha=.25, lw=.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, ncol=min(n, 6),
              loc="upper center", bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(os.path.join(out, "seed_spread_by_target.png"), dpi=200)
    plt.close(fig)

    # ---- 2. four methods, PointNet averaged over seeds -------------------
    refp = _find_reference_run(runs)
    ref = _reference_scores(refp) if refp else {}
    models = [m for m in REF_COLS if any(m in ref.get(t, {}) for t in targets)]
    w2 = 0.8 / (len(models) + 1)
    fig, ax = plt.subplots(figsize=(max(9, 1.5 * len(targets)), 4.8))
    for i, m in enumerate(models):
        vals = [ref.get(t, {}).get(m, np.nan) for t in targets]
        b = ax.bar(x + (i - len(models) / 2) * w2, vals, w2, color=REF_COLS[m][1],
                   label=REF_COLS[m][0], edgecolor="white", linewidth=.4)
        _label(ax, b)
    mean = summary["mean"].values
    lo = mean - summary["lo"].values
    hi = summary["hi"].values - mean
    b = ax.bar(x + (len(models) - len(models) / 2) * w2, mean, w2, color=PN_COLOUR,
               yerr=[lo, hi], capsize=3, ecolor="#333333",
               error_kw=dict(lw=1.0),
               label=f"PointNet v2 — mean of {len(names)} seeds, bar = min–max",
               edgecolor="white", linewidth=.4)
    _label(ax, b, tops=summary["hi"].values)
    ax.set_xticks(x); ax.set_xticklabels(targets, rotation=20, ha="right")
    ax.set_ylabel(r"$R^2$  (out-of-fold, higher is better)")
    ax.set_ylim(min(0, float(np.nanmin(summary["lo"].values)) - .08), 1.12)
    ax.axhline(0, color="#444444", lw=.8)
    ax.set_title(f"All methods by target — PointNet averaged over seeds  "
                 f"({profile}, grouped 5-fold CV)", fontsize=11, weight="bold")
    ax.grid(axis="y", alpha=.25, lw=.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, ncol=2,
              loc="upper center", bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    if refp:
        fig.text(0.995, 0.008,
                 f"RF / GPR / density read from {os.path.basename(os.path.dirname(refp))}",
                 ha="right", va="bottom", fontsize=7, color="#888888")
    fig.savefig(os.path.join(out, "methods_seedmean.png"), dpi=200)
    plt.close(fig)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="ADMS")
    ap.add_argument("--runs", nargs="*")
    ap.add_argument("--out", default=None,
                    help="where the figures and summary CSV go "
                         "(default runs/<PROFILE>/_seed_summary)")
    ap.add_argument("--no-charts", action="store_true")
    a = ap.parse_args()

    runs = a.runs or sorted(glob.glob(os.path.join(BASE, "runs", a.profile, "*")))
    data = {}
    for r in runs:
        d = load(r)
        if d is not None:
            data[os.path.basename(r)] = d
    if len(data) < 2:
        print(f"[!] found {len(data)} run(s) with PointNet predictions under "
              f"runs/{a.profile}. Need at least 2 to show a spread.")
        for k in data:
            print("    ", k)
        sys.exit(1)

    names = list(data)
    targets = sorted(set().union(*[set(d.target.unique()) for d in data.values()]))

    print(f"\n{'='*len('OVERALL grouped-CV R2 per run')}")
    print("OVERALL grouped-CV R2 per run")
    print("=" * 78)
    hdr = f"{'target':14s}" + "".join(f"{n[-12:]:>14s}" for n in names) + f"{'mean':>9s}{'range':>9s}"
    print(hdr)
    rows = []
    for t in targets:
        vals = []
        for n in names:
            g = data[n][data[n].target == t]
            vals.append(r2(g.y_true, g.y_pred) if len(g) > 2 else np.nan)
        v = np.array(vals, float)
        ok = v[~np.isnan(v)]
        print(f"{t:14s}" + "".join(f"{x:>14.3f}" for x in v)
              + f"{ok.mean():>9.3f}{ok.max()-ok.min():>9.3f}")
        rows.append(dict(target=t, mean=ok.mean(), spread=ok.max() - ok.min(),
                         sd=ok.std(ddof=1) if len(ok) > 1 else 0.0,
                         lo=ok.min(), hi=ok.max(),
                         **{f"run::{n}": x for n, x in zip(names, v)}))

    print("\n" + "=" * 78)
    print("PER-FOLD R2  (a single bad fold is what a low overall score usually means)")
    print("=" * 78)
    for t in ["E_over_Es"] + [x for x in targets if x != "E_over_Es"][:2]:
        print(f"\n  target: {t}")
        print(f"    {'run':22s}" + "".join(f"{'fold '+str(i):>10s}" for i in range(5)))
        for n in names:
            d = data[n]
            if "fold" not in d.columns:
                print(f"    {n[:22]:22s}  (no cv_folds.csv)"); continue
            g = d[d.target == t]
            cells = []
            for fo in range(5):
                x = g[g.fold == fo]
                cells.append(f"{r2(x.y_true, x.y_pred):>10.3f}" if len(x) > 2 else f"{'-':>10s}")
            print(f"    {n[:22]:22s}" + "".join(cells))

    print("\n" + "=" * 78)
    worst = max(rows, key=lambda r: r["spread"])
    print(f"Largest spread across runs: {worst['target']} = {worst['spread']:.3f} R2.")
    print("If the spread is large, quote mean +/- range across seeds, not one run.")
    print("If one fold is negative in one run and fine in the others, that run's")
    print("network for that fold failed to train - it is not a property of the data.")

    if not a.no_charts:
        out = a.out or os.path.join(BASE, "runs", a.profile, "_seed_summary")
        os.makedirs(out, exist_ok=True)
        summary = pd.DataFrame(rows)
        summary.to_csv(os.path.join(out, "seed_spread_summary.csv"), index=False)
        make_charts(summary, names, targets, a.profile, runs, out)
        print(f"\n[ok] {out}")
        for f in ("seed_spread_by_target.png", "methods_seedmean.png",
                  "seed_spread_summary.csv"):
            print(f"       {f}")


if __name__ == "__main__":
    main()
