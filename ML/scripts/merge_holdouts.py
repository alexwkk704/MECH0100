#!/usr/bin/env python3
"""merge_holdouts.py — bring holdout results from two runs into one folder.

Why this exists
---------------
train_pointnet_v2.py OVERWRITES pointnet_holdouts.csv with whatever holdouts are
enabled for that run, so it cannot be pointed straight at an existing run folder
to add the missing ones - it would delete the ones already there. Instead the
missing holdouts are trained into a scratch folder and this merges them BACK into
the real run folder, which then holds the complete set.

--into  the run folder that ends up complete (its own rows win on any clash)
--from  the scratch folder holding the newly trained holdouts

Each holdout is an independent model trained from scratch on its own split, so
combining them is exactly as valid as having run all eight in one go. The source
run is recorded per row in `source_run` so the provenance is never lost.

It also copies model_results.csv (RF / GPR / density, all splits) across, so
holdout_charts.py can draw all four methods in the merged folder.

Usage:
    python merge_holdouts.py --into runs/COMBINED/20260811_0800 \
                            --from runs/COMBINED/_staging_holdouts_20260811_1200
"""
import argparse, glob, os, shutil, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
NAME = "pointnet_holdouts.csv"
PRED = "pointnet_holdout_predictions.csv"


def newest_other(into):
    parent = os.path.dirname(os.path.abspath(into))
    cands = [d for d in sorted(glob.glob(os.path.join(parent, "*")))
             if os.path.isdir(d)
             and os.path.abspath(d) != os.path.abspath(into)
             and os.path.exists(os.path.join(d, NAME))]
    return cands[-1] if cands else None


def tag(df, run):
    df = df.copy()
    df["source_run"] = os.path.basename(run.rstrip("/\\"))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--into", required=True,
                    help="run folder that ends up with the complete set")
    ap.add_argument("--from", dest="src",
                    help="folder holding the extra holdouts (default: newest sibling)")
    a = ap.parse_args()

    into = a.into
    new_p = os.path.join(into, NAME)
    if not os.path.exists(new_p):
        sys.exit(f"[E] {new_p} not found - did the holdout run finish?")
    src = a.src or newest_other(into)
    if not src:
        sys.exit(f"[E] no other run with {NAME} next to {into}")

    new = tag(pd.read_csv(new_p), into)
    old = tag(pd.read_csv(os.path.join(src, NAME)), src)
    both = pd.concat([new, old], ignore_index=True)
    before = len(both)
    both = both.drop_duplicates(subset=["target", "split", "model"], keep="first")
    both.to_csv(new_p, index=False)
    print(f"[ok] {new_p}")
    print(f"     {len(new)} rows from this run + {len(old)} from {os.path.basename(src)} "
          f"-> {len(both)} ({before - len(both)} duplicate split/target dropped, this run wins)")
    print(f"     splits now: {sorted(both.split.unique())}")

    for f in (PRED, "model_results.csv"):
        s, d = os.path.join(src, f), os.path.join(into, f)
        if os.path.exists(s) and not os.path.exists(d):
            shutil.copy(s, d)
            print(f"     copied {f} from {os.path.basename(src)}")
        elif os.path.exists(s) and f == PRED:
            a_, b_ = pd.read_csv(d), pd.read_csv(s)
            pd.concat([a_, b_], ignore_index=True).drop_duplicates(
                subset=["Run", "split", "target"], keep="first").to_csv(d, index=False)
            print(f"     merged {f}")


if __name__ == "__main__":
    main()
