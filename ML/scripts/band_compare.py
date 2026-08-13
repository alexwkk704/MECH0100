#!/usr/bin/env python3
"""band_compare.py - did the new simulations actually help, and where?

Aggregate R2 over a whole dataset hides where a model is strong or weak. When a
batch of simulations is added to fill a specific gap, the honest question is
narrow: did accuracy improve IN THAT REGION? This scores out-of-fold
cross-validation predictions restricted to a volume-fraction band, and puts two
runs side by side.

WHY THIS EXISTS, and the trap it avoids
---------------------------------------
An `extrap_high_density` holdout defined as VF >= 0.38 puts every high-VF row in
the TEST set. Adding more high-VF simulations therefore leaves the TRAINING set
(everything below 0.38) unchanged, so that holdout CANNOT improve - and may look
worse because the test set grew. The improvement shows up in cross-validation,
where the new rows appear in both training and test folds. That is what this
measures.

WHY --family EXISTS  (added 2026-08-07)
---------------------------------------
On a COMBINED run the VF band contains rows from BOTH families. If only one
family gained new simulations, the unchanged family's rows dilute the measured
change and the answer comes out muted for a reason that has nothing to do with
the model. `--family ADMS` restricts scoring to the rows that actually changed.
Run it BOTH ways: the all-rows number is the honest headline for the delivered
model, the per-family number is what explains it. The family column is READ from
the run's own dataset_*.csv - nothing about family names is hardcoded here.

USAGE
    python band_compare.py --before <run_dir> --after <run_dir> [--band LO HI]
    python band_compare.py --after  <run_dir>                  # single run
    python band_compare.py --after  <run_dir> --band 0.38 1.0 --band 0.0 0.11
    python band_compare.py --before <a> --after <b> --family ADMS

Reads, from each run folder:
    cv_oof_predictions.csv    RF / GPR / density baseline (train.py, per row)
    pointnet_v2_predictions.csv + dataset_*.csv   PointNet (VF joined by Run)
    dataset_*.csv             also supplies the family column, when present
"""
import argparse, glob, os, sys
import numpy as np
import pandas as pd

# Candidate names for the column that says which family/dataset a row came from.
# Checked in order; the first one present in the run's dataset_*.csv wins.
FAMILY_COL_CANDIDATES = ("family", "dataset", "source")


def _r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ss = ((y - y.mean()) ** 2).sum()
    return float("nan") if ss == 0 else 1.0 - ((y - p) ** 2).sum() / ss


def _mape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = np.abs(y) > 1e-12
    return float("nan") if not ok.any() else float(np.mean(np.abs((y[ok] - p[ok]) / y[ok])) * 100)


def _run_key(run_dir):
    """Per-Run lookup of VF and family, taken from the run's own dataset_*.csv.

    Returned so BOTH the RF/GPR frame and the PointNet frame can be enriched from
    one source, which guarantees they are filtered on identical definitions.
    """
    ds = sorted(glob.glob(os.path.join(run_dir, "dataset_*.csv")))
    if not ds:
        return None, None
    d = pd.read_csv(ds[0])
    if "Run" not in d.columns:
        return None, None
    fam_col = next((c for c in FAMILY_COL_CANDIDATES if c in d.columns), None)
    cols = ["Run"]
    if "VF" in d.columns:
        cols.append("VF")
    if fam_col:
        cols.append(fam_col)
    key = d[cols].drop_duplicates("Run")
    if fam_col and fam_col != "family":
        key = key.rename(columns={fam_col: "family"})
    return key, fam_col


def load_run(run_dir):
    """Return a tidy frame: Run, target, model, VF, y_true, y_pred [, family]."""
    frames = []
    key, fam_col = _run_key(run_dir)

    def enrich(d):
        """Attach VF and family from the run's dataset, without overwriting VF."""
        if key is None:
            return d
        want = [c for c in ("VF", "family") if c in key.columns and c not in d.columns]
        if not want:
            return d
        return d.merge(key[["Run"] + want], on="Run", how="left")

    oof = os.path.join(run_dir, "cv_oof_predictions.csv")
    if os.path.exists(oof):
        frames.append(enrich(pd.read_csv(oof)))

    pn = os.path.join(run_dir, "pointnet_v2_predictions.csv")
    if os.path.exists(pn):
        d = enrich(pd.read_csv(pn))
        if "VF" not in d.columns:
            print(f"  [!] {os.path.basename(run_dir)}: no dataset_*.csv next to "
                  f"pointnet_v2_predictions.csv - cannot attach VF, PointNet skipped")
        else:
            d["model"] = "PointNet"
            cols = ["Run", "target", "model", "VF", "y_true", "y_pred"]
            if "family" in d.columns:
                cols.append("family")
            frames.append(d[cols])

    if not frames:
        print(f"  [!] {run_dir}: no per-row prediction files found.")
        print( "      Expected cv_oof_predictions.csv (written by train.py from 2026-08-07)")
        print( "      and/or pointnet_v2_predictions.csv. Older runs have only aggregates.")
        return None
    out = pd.concat(frames, ignore_index=True)
    if fam_col is None:
        print(f"  [i] {os.path.basename(run_dir)}: no family column in dataset_*.csv "
              f"(looked for {', '.join(FAMILY_COL_CANDIDATES)}) - --family cannot be applied here")
    return out.dropna(subset=["VF"])


def apply_family(df, fam, label):
    """Filter to one family, and SAY what was dropped rather than dropping quietly."""
    if fam is None:
        return df
    if "family" not in df.columns:
        print(f"  [!] {label}: --family {fam} requested but this run has no family "
              f"column - NOT filtered, the number below covers ALL rows")
        return df
    present = sorted(str(x) for x in df["family"].dropna().unique())
    sel = df[df["family"].astype(str).str.lower() == str(fam).lower()]
    if sel.empty:
        print(f"  [!] {label}: no rows with family == '{fam}'. Present: {present}")
    else:
        dropped = len(df) - len(sel)
        print(f"  [i] {label}: family '{fam}' kept {len(sel)} rows, dropped {dropped} "
              f"(families present: {present})")
    return sel


def score(df, lo, hi):
    sel = df[(df["VF"] >= lo) & (df["VF"] < hi)]
    rows = []
    for (t, m), g in sel.groupby(["target", "model"]):
        if len(g) < 3:
            continue
        rows.append(dict(target=t, model=m, N=len(g),
                         R2=round(_r2(g["y_true"], g["y_pred"]), 4),
                         MAPE=round(_mape(g["y_true"], g["y_pred"]), 2)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before")
    ap.add_argument("--after", required=True)
    ap.add_argument("--band", nargs=2, type=float, action="append", metavar=("LO", "HI"))
    ap.add_argument("--family", default=None,
                    help="restrict scoring to one family/dataset value, e.g. ADMS. "
                         "Read from the run's own dataset_*.csv; not hardcoded.")
    a = ap.parse_args()
    bands = a.band or [[0.0, 1.0], [0.38, 1.0]]

    after = load_run(a.after)
    if after is None:
        sys.exit(1)
    before = load_run(a.before) if a.before else None

    if a.family:
        after = apply_family(after, a.family, "after")
        if before is not None:
            before = apply_family(before, a.family, "before")
        if after.empty:
            print("[E] nothing left to score after the family filter."); sys.exit(1)

    tag = f"   family={a.family}" if a.family else ""
    for lo, hi in bands:
        print(f"\n{'='*78}\nVF band {lo:.3f} <= VF < {hi:.3f}{tag}")
        sa = score(after, lo, hi)
        if sa.empty:
            print("  no rows in this band"); continue
        if before is None:
            print(sa.sort_values(["target", "model"]).to_string(index=False)); continue
        sb = score(before, lo, hi)
        m = sb.merge(sa, on=["target", "model"], how="outer",
                     suffixes=("_before", "_after"))
        m["dR2"] = (m["R2_after"] - m["R2_before"]).round(4)
        m["dMAPE"] = (m["MAPE_after"] - m["MAPE_before"]).round(2)
        cols = ["target", "model", "N_before", "N_after",
                "R2_before", "R2_after", "dR2", "MAPE_before", "MAPE_after", "dMAPE"]
        print(m.sort_values(["target", "model"])[cols].to_string(index=False))
        print("\n  dR2 > 0 and dMAPE < 0 mean the added simulations helped in this band.")


if __name__ == "__main__":
    main()
