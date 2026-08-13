"""
coarse_search.py - INVERSE DESIGN PHASE 1: the coarse search.

Pushes every geometry already in the database through the frozen forward model
and ranks them by distance to a target property. Free, seconds, generates nothing.

WHAT THIS IS FOR - read before quoting the output
    Phase 1 LOCALISES the answer. It does NOT produce it. Every row it can
    return is already in the database and already has an FEA result, so a Phase 1
    winner proves nothing on its own - the model would just be reproducing a
    number we already have. GATE 5 exists precisely to stop that: the design
    carried forward must be a NEW geometry from Phase 2.
    What Phase 1 buys is the neighbourhood to build the Phase 2 grid around,
    for no compute and no Spherene tokens.

    Its predictions are also IN-SAMPLE - the model trained on these rows - so the
    agreement here is an upper bound on accuracy, never an accuracy estimate.
    The honest error figure is the grouped-CV MAPE from the results chapter.

WHERE THINGS LIVE
    Forward ML, including predict.py and the exported model, stays in Share\\ML.
    This file is a CONSUMER: it imports predict.py from there rather than
    carrying a copy, so the prediction path can never drift from the one GATE 0
    verified. Nothing here writes into Share\\ML.

GATES APPLIED HERE
    GATE 1  the target must sit in the band where BOTH families have data,
            otherwise the winner is decided by which family the database
            happens to cover rather than by design merit.
    GATE 4  reject any row whose PREDICTED value falls outside the range of the
            training labels. A run in this project once returned 3.14x the
            largest training stiffness when pushed past the sampled range, and a
            search is actively rewarded for finding exactly those regions.

Usage:  python coarse_search.py                 (target 0.10, from the bat)
        python coarse_search.py --target 0.09
        python coarse_search.py --target 0.10 --objective E_over_Es --top 15
"""

from __future__ import annotations
import os, sys, argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ID_ROOT = os.path.dirname(HERE)                      # Share\Inverse design

# Share\ML - overridable so the script can be exercised against a test rig.
ML_DIR = os.environ.get("ML_DIR") or os.path.normpath(
    os.path.join(ID_ROOT, "..", "ML"))
ML_SCRIPTS = os.path.join(ML_DIR, "scripts")
if not os.path.isdir(ML_SCRIPTS):
    sys.exit(f"[E] cannot find the forward ML scripts at {ML_SCRIPTS}\n"
             f"    expected this file to live in Share\\Inverse design\\scripts\\")
sys.path.insert(0, ML_SCRIPTS)

import predict as P                                   # noqa: E402
import run_paths                                      # noqa: E402

RESULTS = os.path.join(ID_ROOT, "search", "results")


def load_rows(bundle, profile):
    """Every dataset row that has a stored cloud, with that cloud loaded.

    Deliberately reuses the STORED clouds rather than re-sampling the STLs:
    that is what the plan specifies, it is what the model trained on, and since
    GATE 0 passed we know the two are the same thing.
    """
    ds = run_paths.dataset_path()
    pc = run_paths.pointclouds_dir()
    if not os.path.exists(ds):
        sys.exit(f"[E] dataset not found: {ds}")
    df = pd.read_csv(ds)
    stem_col = "Run" if "Run" in df.columns else "file"
    n_pts = int(bundle["n_points"])

    X, keep, seeds_seen = [], [], set()
    for _, r in df.iterrows():
        stem = str(r[stem_col])
        f = os.path.join(pc, stem + ".npz")
        if not os.path.exists(f):
            continue
        z = np.load(f)
        pts = z["pts"].astype(np.float32)[:n_pts]
        if len(pts) < n_pts:
            continue
        if "seed" in z.files:
            seeds_seen.add(int(z["seed"]))
        X.append(pts)
        keep.append(r)
    if not X:
        sys.exit("[E] no stored clouds matched the dataset")
    meta = pd.DataFrame(keep).reset_index(drop=True)

    # The clouds must all have been built the same way, and the same way the
    # bundle expects. This is the check that was missing when 130 ADMS clouds
    # silently carried a different sampling seed.
    want = int(bundle["pc_seed"])
    if seeds_seen and seeds_seen != {want}:
        sys.exit(f"[E] stored clouds carry seed(s) {sorted(seeds_seen)} but the "
                 f"model expects {want}. Run 0_AUDIT_POINTCLOUDS.bat.")
    if not seeds_seen:
        print("[!] the stored clouds record no seed - they predate the 12/08 fix. "
              "Re-run the SETUP bats before trusting this.")
    return np.stack(X), meta


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=float, default=0.10)
    ap.add_argument("--objective", default="E_over_Es")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--profile", default=os.environ.get("ML_PROFILE", "COMBINED"))
    a = ap.parse_args()
    os.environ["ML_PROFILE"] = a.profile

    model_path = os.path.join(ML_DIR, "Final Model", a.profile, f"model_{a.profile}.pt")
    b = P.load_bundle(model_path, a.profile)

    print("=" * 72)
    print(f"  INVERSE DESIGN - PHASE 1  coarse search")
    print(f"  target {a.objective} = {a.target}")
    print("=" * 72)
    print(f"  model   {model_path}")
    print(f"  built   {b.get('built_utc')}   seeds {b.get('seeds')}")
    print(f"  trained {b.get('n_train')} rows of {b.get('dataset_name')}")
    if a.objective not in b["targets"]:
        sys.exit(f"[E] '{a.objective}' is not a model target. Available: {b['targets']}")

    X, meta = load_rows(b, a.profile)
    print(f"  scoring {len(X)} stored clouds\n")

    dens = None
    if b["use_density"]:
        col = b.get("density_column") or "Rho_rel"
        dens = pd.to_numeric(meta[col], errors="coerce").values.astype(np.float32)
        if np.isnan(dens).any():
            sys.exit(f"[E] {int(np.isnan(dens).sum())} rows have a blank {col}")

    Pred = P.predict_clouds(X, dens, b)
    j = b["targets"].index(a.objective)
    meta["predicted"] = Pred[:, j]
    for k, t in enumerate(b["targets"]):
        meta[f"pred_{t}"] = Pred[:, k]

    # ---- GATE 1 : the target must sit where both families overlap ----------
    print("-" * 72)
    ok1 = True
    if "family" in meta.columns and a.objective in meta.columns:
        true = pd.to_numeric(meta[a.objective], errors="coerce")
        lo = hi = None
        for f, g in meta.assign(_t=true).groupby("family"):
            print(f"  {f:<6} n={len(g):>3}   {a.objective} "
                  f"{g._t.min():.4f} .. {g._t.max():.4f}")
            lo = g._t.min() if lo is None else max(lo, g._t.min())
            hi = g._t.max() if hi is None else min(hi, g._t.max())
        if lo is not None and hi is not None and lo < hi:
            inside = int(((true >= lo) & (true <= hi)).sum())
            print(f"  OVERLAP (both families present): {lo:.4f} .. {hi:.4f}   "
                  f"{inside}/{len(meta)} rows inside")
            ok1 = bool(lo <= a.target <= hi)
            print(f"  GATE 1 {'PASS' if ok1 else 'FAIL'}: target {a.target} is "
                  f"{'inside' if ok1 else 'OUTSIDE'} the overlap")
            if not ok1:
                print("         Outside the overlap the winning family is decided by "
                      "database coverage,\n         not by design merit, and the "
                      "cross-family comparison says nothing.")
        else:
            ok1 = False
            print(f"  GATE 1 FAIL: the two families do NOT overlap at all on "
                  f"{a.objective}.")
            print("         No target can be compared fairly across families. "
                  "Pick a different\n         objective, or report the two "
                  "families separately and say so.")
    else:
        print("  GATE 1 skipped - no 'family' column")

    # ---- GATE 4 : range guard on the PREDICTED value -----------------------
    lo4, hi4 = b["label_min"][a.objective], b["label_max"][a.objective]
    inrange = (meta.predicted >= lo4) & (meta.predicted <= hi4)
    n_out = int((~inrange).sum())
    print(f"  GATE 4 range guard: training labels {lo4:.6g} .. {hi4:.6g}   "
          f"{n_out} row(s) predicted outside -> rejected")
    meta["gate4_ok"] = inrange

    # ---- rank --------------------------------------------------------------
    meta["abs_err"] = (meta.predicted - a.target).abs()
    cand = meta[meta.gate4_ok].sort_values("abs_err").reset_index(drop=True)

    os.makedirs(RESULTS, exist_ok=True)
    cols = [c for c in ("Run", "family", "topology", "VF", "Rho_rel",
                        a.objective, "predicted", "abs_err", "gate4_ok")
            if c in meta.columns]
    out = os.path.join(RESULTS, f"phase1_ranked_{a.objective}_{a.target:g}.csv")
    meta.sort_values("abs_err")[cols].to_csv(out, index=False)

    def show(title, sub):
        if sub.empty:
            print(f"\n  {title}: none"); return None
        r = sub.iloc[0]
        print(f"\n  {title}")
        print(f"    {r['Run']}")
        print(f"    predicted {r['predicted']:.6f}   |err| {r['abs_err']:.6f}"
              f"   ({r['abs_err']/a.target*100:.2f} % of target)")
        if a.objective in sub.columns:
            print(f"    measured (FEA, already known) {r[a.objective]:.6f}")
        if "VF" in sub.columns:
            print(f"    VF {r['VF']:.4f}" + (f"   topology {r['topology']}"
                                             if "topology" in sub.columns else ""))
        return r

    print("\n" + "=" * 72)
    print("  WINNERS - the neighbourhoods for Phase 2 to search around")
    print("=" * 72)
    w_all = show("GLOBAL BEST", cand)
    w_adms = show("BEST ADMS", cand[cand.family == "ADMS"]) if "family" in cand else None
    w_tpms = show("BEST TPMS", cand[cand.family == "TPMS"]) if "family" in cand else None

    print(f"\n  top {a.top} overall:")
    print(f"    {'Run':<40} {'family':<6} {'pred':>9} {'|err|':>9}")
    for _, r in cand.head(a.top).iterrows():
        print(f"    {str(r['Run'])[:40]:<40} {str(r.get('family','')):<6} "
              f"{r['predicted']:9.6f} {r['abs_err']:9.6f}")

    # ---- honest framing, written into the summary as well ------------------
    summ = os.path.join(RESULTS, f"phase1_summary_{a.objective}_{a.target:g}.txt")
    with open(summ, "w") as f:
        f.write(f"INVERSE DESIGN PHASE 1 - coarse search\n")
        f.write(f"target {a.objective} = {a.target}\n")
        f.write(f"model  {os.path.basename(model_path)}  built {b.get('built_utc')}  "
                f"seeds {b.get('seeds')}\n")
        f.write(f"scored {len(X)} stored clouds\n")
        f.write(f"GATE 1 {'PASS' if ok1 else 'FAIL'}   GATE 4 rejected {n_out}\n\n")
        for name, r in (("GLOBAL BEST", w_all), ("BEST ADMS", w_adms),
                        ("BEST TPMS", w_tpms)):
            if r is not None:
                f.write(f"{name}: {r['Run']}  predicted {r['predicted']:.6f}  "
                        f"|err| {r['abs_err']:.6f}\n")
        f.write("\nEVERY row above is an EXISTING database geometry and already has an\n")
        f.write("FEA result, so none of them is the answer - GATE 5 requires the design\n")
        f.write("carried forward to be a NEW geometry from Phase 2. These are the\n")
        f.write("neighbourhoods to build the Phase 2 grid around.\n")
        f.write("Predictions here are IN-SAMPLE (the model trained on these rows), so\n")
        f.write("the agreement is an upper bound, not an accuracy estimate.\n")

    print("\n" + "-" * 72)
    print(f"  ranked  -> {out}")
    print(f"  summary -> {summ}")
    print("\n  ⚠ Every winner above is an EXISTING database row that already has an")
    print("    FEA result. None of them is the answer. GATE 5 requires the design")
    print("    carried forward to be NEW geometry from Phase 2 - these just say")
    print("    WHERE to look.")
    print("=" * 72)


if __name__ == "__main__":
    main()
