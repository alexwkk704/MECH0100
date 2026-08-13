"""
compare_models.py — ONE parity chart per target, all four methods, SAME rows.

Why this exists
---------------
The charts from plot_results.py and from train_pointnet_v2.py are not directly
comparable, and overlaying them would be misleading:

  * RF / GPR / reference come from 5-fold CROSS-VALIDATION. Every row is
    predicted by a model that did not see it, so all ~115 points appear.
  * PointNet comes from a SINGLE held-out test split (23 rows).

Different protocols, different point counts. So this script re-scores RF, GPR
and the density reference on EXACTLY the rows PointNet held out
(outputs/pointnet_v2_split.csv), trains them on exactly the rows PointNet
trained on, and plots all four together.

The 5-fold numbers remain the better estimate of general accuracy - more test
points, less split luck. This chart answers a different question: "on the same
23 unseen geometries, which method predicted them best?"

Run AFTER train_pointnet_v2.py (it needs the split + prediction files).

    python compare_models.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))


def _find_settings(base, name="ML_settings.xlsx"):
    here = os.path.join(base, name)
    if os.path.exists(here):
        return here
    up = os.path.join(os.path.dirname(base), name)
    return up if os.path.exists(up) else here


SETTINGS_XLSX = _find_settings(HERE)
BASE_DIR = os.path.dirname(SETTINGS_XLSX)
sys.path.insert(0, HERE)
import cv_split                     # same grouped folds as train.py / pointnet
import run_paths                   # clean filing: data/ inputs, runs/<stamp>/ results
from train import make_rf, make_gpr   # ONE definition of RF/GPR — must match train.py exactly

PURPLE = "#500778"
STYLE = {                                   # colour, marker, label
    "reference_density": ("0.55", "s", "Density-only reference (not a model)"),
    "RF":                ("#2a7f3f", "^", "Random Forest"),
    "GPR":               (PURPLE,    "o", "Gaussian Process"),
    "PointNet":          ("#c1442e", "D", "PointNet v2 (merged)"),
}


def load_cfg():
    import openpyxl
    wb = openpyxl.load_workbook(SETTINGS_XLSX, data_only=True)
    c = {str(k).strip(): v for k, v, *_ in
         wb[run_paths.sheet_for(wb, "train")].iter_rows(min_row=2, values_only=True) if k}
    c["features"] = [s.strip() for s in str(c["features"]).split(",") if s.strip()]
    c["targets"] = [s.strip() for s in str(c["targets"]).split(",") if s.strip()]
    c["group_cols"] = [s.strip() for s in str(c.get("group_cols", "")).split(",") if s.strip()]
    c["dataset_csv"] = run_paths.data_path(c["dataset_csv"])   # shared input, ML/data/
    c["charts_dir"] = run_paths.run_path("charts")             # per-run output
    for k in ("rf_trees", "random_seed", "n_folds"):
        c[k] = int(c[k])
    return c


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ss = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - p) ** 2).sum() / ss) if ss > 0 else float("nan")


def mape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = np.abs(y) > 1e-12
    return float(np.mean(np.abs((p[m] - y[m]) / y[m])) * 100)


def power_fit(vf, y):
    m = (vf > 0) & (y > 0)
    if m.sum() < 3:
        raise ValueError("not enough positive rows for the density reference")
    n, lc = np.polyfit(np.log(vf[m]), np.log(y[m]), 1)
    C = np.exp(lc)
    return lambda v: C * np.power(np.clip(v, 1e-12, None), n)


def main():
    cfg = load_cfg()
    pred_csv = run_paths.run_path("pointnet_v2_predictions.csv")
    if not os.path.exists(pred_csv):
        sys.exit("[E] pointnet_v2_predictions.csv not found in this run — run train_pointnet_v2.py first")

    d = pd.read_csv(cfg["dataset_csv"])
    pn = pd.read_csv(pred_csv)

    # SAME grouped folds as train.py / pointnet — recomputed deterministically
    # from the dataset, so no split file needs to be shared.
    fold_of, grouped, stem_col = cv_split.build_fold_map(
        d, cfg["group_cols"], cfg["n_folds"], seed=cfg["random_seed"])
    fold = cv_split.folds_for(d, cfg["group_cols"], fold_of, stem_col)
    split_name = f"grouped{cfg['n_folds']}foldCV" if grouped else f"{cfg['n_folds']}foldCV"
    print(f"comparing all methods on {split_name}: "
          f"{cfg['n_folds']} folds over {len(d)} rows"
          + (f" grouped by {cfg['group_cols']}" if grouped else ""))
    if not grouped:
        print("\n" + "!" * 68)
        print("!!  WARNING: RANDOM (UNGROUPED) SPLIT — results are LEAKY / INFLATED.")
        print("!!  Near-duplicate geometries can sit in both train and test, so")
        print("!!  geometry-sensitive scores (esp. TAI) are optimistic and NOT honest.")
        if not cfg["group_cols"]:
            print("!!  CAUSE: 'group_cols' is empty in ML_settings.xlsx (train sheet).")
            print("!!  FIX  : set  group_cols = adms_type, density_param  and re-run.")
        else:
            print(f"!!  CAUSE: group_cols {cfg['group_cols']} not all present in the dataset.")
        print("!" * 68 + "\n")

    feats = [f for f in cfg["features"] if f in d.columns]
    X = d[feats].apply(pd.to_numeric, errors="coerce").values
    vf = pd.to_numeric(d[cfg.get("vf_column", "VF")], errors="coerce").values

    os.makedirs(cfg["charts_dir"], exist_ok=True)
    summary = []

    for tgt in cfg["targets"]:
        if tgt not in d.columns:
            continue
        y = pd.to_numeric(d[tgt], errors="coerce").values
        ok = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
        if ok.sum() < 10:
            print(f"  {tgt:15} skipped ({int(ok.sum())} labelled)")
            continue

        # out-of-fold RF / GPR / density-reference over ALL labelled rows,
        # on exactly the folds PointNet used.
        preds = {m: np.full(len(d), np.nan) for m in ("RF", "GPR", "reference_density")}
        for f in range(cfg["n_folds"]):
            te = np.flatnonzero(ok & (fold == f))
            tr = np.flatnonzero(ok & (fold != f))
            if len(te) == 0 or len(tr) < 10:
                continue
            rf = make_rf(cfg)
            rf.fit(X[tr], y[tr]); preds["RF"][te] = rf.predict(X[te])
            sc = StandardScaler().fit(X[tr])
            gp = make_gpr(cfg)
            gp.fit(sc.transform(X[tr]), y[tr]); preds["GPR"][te] = gp.predict(sc.transform(X[te]))
            try:
                preds["reference_density"][te] = power_fit(vf[tr], y[tr])(vf[te])
            except ValueError:
                pass

        # PointNet out-of-fold predictions, aligned by Run stem
        sub = pn[pn.target == tgt].set_index("Run")
        ids = d[stem_col].astype(str).values
        preds["PointNet"] = (np.array([sub["y_pred"].get(i, np.nan) for i in ids], float)
                             if len(sub) else np.full(len(d), np.nan))

        te = np.flatnonzero(ok)          # score/plot over all labelled rows

        # ---- chart ----
        fig, ax = plt.subplots(figsize=(6.4, 6.0))
        yt = y[te]
        lo, hi = np.nanmin(yt), np.nanmax(yt)
        pad = 0.08 * (hi - lo if hi > lo else abs(hi) or 1)
        lims = (lo - pad, hi + pad)
        ax.plot(lims, lims, "k--", lw=1, zorder=1, label="perfect prediction")
        for name in ("reference_density", "RF", "GPR", "PointNet"):
            if name not in preds:
                continue
            c, mk, lab = STYLE[name]
            p = preds[name][te]
            m = ~np.isnan(p)
            if m.sum() < 3:
                continue
            ax.scatter(yt[m], p[m], s=46, c=c, marker=mk, alpha=0.82,
                       edgecolors="white", linewidths=0.6, zorder=3,
                       label=f"{lab}   R²={r2(yt[m], p[m]):.3f}")
            summary.append(dict(target=tgt, model=name, N_test=int(m.sum()),
                                R2=round(r2(yt[m], p[m]), 4),
                                MAPE=round(mape(yt[m], p[m]), 2)))
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel(f"FEA (true)  —  {tgt}")
        ax.set_ylabel(f"Predicted  —  {tgt}")
        ax.set_title(f"{tgt} — all methods, {split_name} (N={len(te)})",
                     fontweight="bold")
        ax.grid(alpha=0.25, ls="--")
        ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
        fig.tight_layout()
        out = os.path.join(cfg["charts_dir"], f"compare_{tgt}.png")
        fig.savefig(out, dpi=160); plt.close(fig)

        # ---- side-by-side 4-panel: one method per panel, identical axes ----
        panels = ("reference_density", "RF", "GPR", "PointNet")
        fig2, axes = plt.subplots(1, 4, figsize=(17.5, 4.6), sharex=True, sharey=True)
        for ax2, name in zip(axes, panels):
            c, mk, lab = STYLE[name]
            p = preds.get(name, np.full(len(d), np.nan))[te]
            m = ~np.isnan(p)
            ax2.plot(lims, lims, "k--", lw=1, zorder=1)
            if m.sum() >= 3:
                ax2.scatter(yt[m], p[m], s=40, c=c, marker=mk, alpha=0.82,
                            edgecolors="white", linewidths=0.5, zorder=3)
                title = f"{lab}\nR²={r2(yt[m], p[m]):.3f}  MAPE={mape(yt[m], p[m]):.1f}%"
            else:
                title = f"{lab}\n(no prediction)"
            ax2.set_title(title, fontsize=9, fontweight="bold")
            ax2.set_xlim(lims); ax2.set_ylim(lims)
            ax2.set_xlabel(f"FEA (true) — {tgt}", fontsize=8)
            ax2.grid(alpha=0.25, ls="--")
        axes[0].set_ylabel(f"Predicted — {tgt}", fontsize=9)
        fig2.suptitle(f"{tgt} — side-by-side, {split_name} (N={len(te)})",
                      fontweight="bold", fontsize=12)
        fig2.tight_layout(rect=(0, 0, 1, 0.94))
        out2 = os.path.join(cfg["charts_dir"], f"sidebyside_{tgt}.png")
        fig2.savefig(out2, dpi=160); plt.close(fig2)
        print(f"  {tgt:15} -> {os.path.basename(out)} + {os.path.basename(out2)}")

    s = pd.DataFrame(summary)
    out_csv = run_paths.run_path("compare_models_same_split.csv")
    s.to_csv(out_csv, index=False)
    print(f"\n[OK] wrote {out_csv}")
    if len(s):
        piv = s.pivot_table(index="target", columns="model", values="R2")
        print(f"\nR² on identical {split_name} out-of-fold predictions:")
        print(piv.round(3).to_string())

        # ---- 4-method bar chart: R² by target (all methods, same folds) ----
        # This lives HERE (step 7), not in plot_results (step 5), because PointNet
        # results only exist after step 6 — that is why the old bar chart was missing it.
        order = [m for m in ("reference_density", "RF", "GPR", "PointNet") if m in piv.columns]
        targets = [t for t in cfg["targets"] if t in piv.index]
        fig3, ax3 = plt.subplots(figsize=(max(9, 1.5 * len(targets)), 5.4))
        x = np.arange(len(targets)); w = 0.8 / max(len(order), 1)
        for i, mname in enumerate(order):
            vals = [piv.loc[t, mname] if not np.isnan(piv.loc[t, mname]) else 0 for t in targets]
            c, mk, lab = STYLE[mname]
            bb = ax3.bar(x + (i - (len(order) - 1) / 2) * w, vals, w, label=lab, color=c)
            ax3.bar_label(bb, fmt="%.2f", fontsize=6, padding=1)
        ax3.axhline(0, color="k", lw=0.8)
        ax3.set_xticks(x); ax3.set_xticklabels(targets, rotation=20, ha="right")
        ax3.set_ylabel("R²  (out-of-fold — higher is better)")
        ax3.set_ylim(top=1.06)
        ax3.set_title(f"All four methods by target — {split_name} (N={len(d)})", fontweight="bold")
        ax3.legend(fontsize=8, ncol=4, loc="lower center", framealpha=0.9)
        ax3.grid(axis="y", alpha=0.25, ls="--")
        fig3.tight_layout()
        out3 = os.path.join(cfg["charts_dir"], "bars_by_target.png")
        fig3.savefig(out3, dpi=160); plt.close(fig3)
        print(f"[OK] wrote {out3}  (4-method R² bar chart — includes PointNet)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
