"""
rank_designs.py - INVERSE DESIGN PHASE 3: rank everything, name the finalists.

Brings the two Phase 2 families and the Phase 1 database sweep into ONE table,
applies the selection rule, and names the single ADMS and single TPMS design
that go to the FEA.

THE SELECTION RULE, AND WHY IT IS NOT JUST "SMALLEST ERROR"
    The model cannot resolve the candidates it produced. Measured on this
    project's own runs, its in-sample error on the two reproduction rows was
    +2.27 % (ADMS) and -3.91 % (TPMS), and the honest grouped-CV MAPE is
    5.7-6.6 %. The Phase 2 candidates sit 0.4-4 % from target. **They are a tie.**
    Ranking them by |predicted - target| and reading off the top is ranking noise.

    So: any candidate within --tie-band of the target is treated as equal on
    stiffness, and the tie is broken by the SECOND objective.

    TAI is the Total Anisotropy Index - the Frobenius distance from the nearest
    isotropic tensor (tensor_ops.tai). It is ZERO for a perfectly isotropic
    material, so MORE ISOTROPIC = LOWER TAI. Ranking "highest TAI" would pick
    the most anisotropic design, which is the opposite of what is wanted.

    ⚠ The band must be set BEFORE looking at the answer, and reported. Choosing
    it afterwards is cherry-picking. The default is the grouped-CV MAPE from the
    results chapter, which is the model's honest accuracy - anything inside it is
    genuinely indistinguishable.

    ⚠ TAI is the model's WEAKEST target (grouped R^2 0.369, negative on every
    holdout). The tie-break is therefore reported as a decision, not as a
    measurement, and Phase 4 tests whether it was justified.

GATE 5 IS ENFORCED HERE
    Phase 1 rows are carried in the table for context ONLY. Every one of them
    already has an FEA result, so selecting one would mean the model had merely
    reproduced a number we already had. Only ID2_ candidates - geometry that has
    never been simulated - are eligible to be carried forward.

Usage
    python rank_designs.py
    python rank_designs.py --target 0.10 --tie-band 6.6
    python rank_designs.py --secondary pred_TAI --secondary-goal min
"""

from __future__ import annotations
import os, sys, glob, json, time, argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ID_ROOT = os.path.dirname(HERE)
SHARE = os.path.normpath(os.path.join(ID_ROOT, ".."))
ML_DIR = os.environ.get("ML_DIR") or os.path.join(SHARE, "ML")
ML_SCRIPTS = os.path.join(ML_DIR, "scripts")
if not os.path.isdir(ML_SCRIPTS):
    sys.exit(f"[E] cannot find the forward ML scripts at {ML_SCRIPTS}")
sys.path.insert(0, ML_SCRIPTS)

RESULTS = os.path.join(ID_ROOT, "search", "results")


def die(m):
    print(f"\n[E] {m}")
    sys.exit(1)


def newest(pattern):
    f = sorted(glob.glob(os.path.join(RESULTS, pattern)))
    return f[-1] if f else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=float, default=0.10)
    ap.add_argument("--objective", default="E_over_Es")
    ap.add_argument("--tie-band", type=float, default=6.6,
                    help="per cent of target. Candidates inside it are treated "
                         "as tied on the objective. DEFAULT 6.6 = the upper end "
                         "of the grouped-CV MAPE. State it in the write-up.")
    ap.add_argument("--secondary", default="pred_TAI")
    ap.add_argument("--secondary-goal", choices=["min", "max"], default="min",
                    help="min for TAI: LOWER TAI = MORE ISOTROPIC")
    ap.add_argument("--top", type=int, default=5)
    a = ap.parse_args()

    ocol = f"pred_{a.objective}"
    fa = newest(f"phase2_adms_{a.objective}_*.csv")
    ft = newest(f"phase2_tpms_{a.objective}_*.csv")
    f1 = newest(f"phase1_ranked_{a.objective}_*.csv")
    if not (fa or ft):
        die("no Phase 2 results found. Run ID_2B and ID_3C first.")

    frames = []
    for f, fam in ((fa, "ADMS"), (ft, "TPMS")):
        if not f:
            print(f"  [!] no Phase 2 results for {fam} - it cannot be carried forward")
            continue
        d = pd.read_csv(f)
        d["family"] = fam
        d["source"] = "PHASE2_NEW"
        d["density"] = d.p_rel if "p_rel" in d.columns else d.vf
        d["eligible"] = d.gate2 & d.gate4 & (d.gate3 != False)      # noqa: E712
        frames.append(d)
    new = pd.concat(frames, ignore_index=True)

    print("=" * 78)
    print("  INVERSE DESIGN - PHASE 3   rank everything, name the finalists")
    print("=" * 78)
    print(f"  target        {a.objective} = {a.target}")
    print(f"  tie band      +/-{a.tie_band:g} % of target  ->  "
          f"{a.target * (1 - a.tie_band / 100):.5f} .. {a.target * (1 + a.tie_band / 100):.5f}")
    print(f"                Anything inside this is TIED on {a.objective}: it is the")
    print(f"                model's own accuracy, so the differences are not real.")
    print(f"  tie-break     {a.secondary}, {'LOWEST' if a.secondary_goal == 'min' else 'HIGHEST'}"
          + ("   (TAI is a distance FROM isotropy: lower = more isotropic)"
             if "TAI" in a.secondary else ""))
    print(f"  candidates    {len(new)} new geometries "
          f"({int((new.family == 'ADMS').sum())} ADMS, {int((new.family == 'TPMS').sum())} TPMS)")

    new["off_pct"] = (new[ocol] - a.target).abs() / a.target * 100.0
    new["in_band"] = new.off_pct <= a.tie_band
    ok = new[new.eligible].copy()
    if not len(ok):
        die("no Phase 2 candidate passed its gates")

    # Rank: inside the band first (they are tied on the objective, so the
    # secondary decides), then everything else by distance to target.
    asc = a.secondary_goal == "min"
    ranked = pd.concat([
        ok[ok.in_band].sort_values(a.secondary, ascending=asc),
        ok[~ok.in_band].sort_values("off_pct"),
    ], ignore_index=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))

    print(f"\n  {int(ok.in_band.sum())} of {len(ok)} candidates fall inside the tie "
          f"band and are ranked by {a.secondary}; the rest by distance to target.")

    print("\n" + "=" * 78)
    print(f"  TOP {a.top}")
    print("=" * 78)
    print(f"  {'#':>2} {'family':<5} {'candidate':<40} {'rho':>7} {'pred':>9} "
          f"{'off%':>6} {'TAI':>7}")
    for _, r in ranked.head(a.top).iterrows():
        print(f"  {int(r['rank']):>2} {r.family:<5} {str(r.stem)[:40]:<40} "
              f"{r.density:7.5f} {r[ocol]:9.6f} {r.off_pct:6.2f} "
              f"{r.get('pred_TAI', float('nan')):7.4f}")

    fams = ranked.head(a.top).family.nunique()
    if fams == 1:
        only = ranked.head(a.top).family.iloc[0]
        print(f"\n  [!] The top {a.top} are ALL {only}. That is the tie-break doing its")
        print(f"      job, not a verdict on the families: inside the band they are")
        print(f"      ranked by {a.secondary}, and {only} simply has the lower values.")
        print(f"      Read the per-family finalists and the trade-off table below -")
        print(f"      a single-family top-N is NOT evidence that {only} 'won'.")

    fin = {}
    print("\n" + "=" * 78)
    print("  FINALISTS - one per family, these are what Phase 4 simulates")
    print("=" * 78)
    for fam in ("ADMS", "TPMS"):
        g = ranked[ranked.family == fam]
        if g.empty:
            print(f"\n  {fam}: none eligible")
            continue
        w = g.iloc[0]
        fin[fam] = w
        print(f"\n  {fam}   {w.stem}")
        print(f"      relative density {w.density:.5f}")
        print(f"      predicted {a.objective} {w[ocol]:.6f}   {w.off_pct:.2f} % from target")
        print(f"      predicted TAI     {w.get('pred_TAI', float('nan')):.4f}")
        if fam == "ADMS":
            print(f"      Density {w.D}  Thickness {w.t}  Inner Size {w.inner}  "
                  f"Size Multi {w.size_multi}  Seed {int(w.seed)}")
        else:
            print(f"      t_max {w.t_max:.6f}  t_min {w.t_min:.6f}  "
                  f"Tolerance {w.tolerance_mm:g} mm")
        print(f"      STL {w.stl}")

    # ---- the cross-family comparison, stated honestly ----------------------
    if len(fin) == 2:
        A, T = fin["ADMS"], fin["TPMS"]
        print("\n" + "=" * 78)
        print("  ADMS vs TPMS - a TRADE-OFF, not a winner")
        print("=" * 78)
        dd = (A.density - T.density) / T.density * 100.0
        dt = (A.pred_TAI - T.pred_TAI) / T.pred_TAI * 100.0
        print(f"    {'':22s} {'ADMS':>14s} {'TPMS':>14s}")
        print(f"    {'relative density':22s} {A.density:14.5f} {T.density:14.5f}")
        print(f"    {'predicted ' + a.objective:22s} {A[ocol]:14.6f} {T[ocol]:14.6f}")
        print(f"    {'off target %':22s} {A.off_pct:14.2f} {T.off_pct:14.2f}")
        print(f"    {'predicted TAI':22s} {A.pred_TAI:14.4f} {T.pred_TAI:14.4f}")
        print(f"\n    ADMS needs {dd:+.1f} % relative density for the same stiffness.")
        print(f"    ADMS TAI is {dt:+.1f} % vs TPMS  (negative = MORE isotropic).")
        print(f"\n    Both sit inside the {a.tie_band:g} % tie band, so the difference in")
        print(f"    'off target %' is NOT evidence that one family is more accurate.")

    os.makedirs(RESULTS, exist_ok=True)
    fcsv = os.path.join(RESULTS, f"phase3_ranking_{a.objective}_{a.target:g}.csv")
    keep = ["rank", "family", "source", "stem", "density", ocol, "off_pct",
            "in_band", "pred_TAI", "pred_G_over_Gs", "stl"]
    ranked[[c for c in keep if c in ranked.columns]].to_csv(fcsv, index=False)

    # Phase 1 rows: context only, never selectable.
    n1 = 0
    if f1:
        r1 = pd.read_csv(f1)
        r1["off_pct"] = (r1.predicted - a.target).abs() / a.target * 100.0
        n1 = len(r1)
        r1.sort_values("off_pct").head(20).to_csv(
            os.path.join(RESULTS,
                         f"phase3_context_existing_{a.objective}_{a.target:g}.csv"),
            index=False)

    fjson = os.path.join(RESULTS, f"phase3_finalists_{a.objective}_{a.target:g}.json")
    with open(fjson, "w") as f:
        json.dump({
            "built_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "target": a.target, "objective": a.objective,
            "tie_band_pct": a.tie_band, "tie_break": a.secondary,
            "tie_break_goal": a.secondary_goal,
            "n_candidates": int(len(new)), "n_eligible": int(len(ok)),
            "n_in_band": int(ok.in_band.sum()),
            "n_existing_rows_for_context": n1,
            "finalists": {k: {"stem": v.stem, "density": float(v.density),
                              "predicted": float(v[ocol]),
                              "off_pct": float(v.off_pct),
                              "pred_TAI": float(v.pred_TAI),
                              "stl": v.stl}
                          for k, v in fin.items()}}, f, indent=2)

    print("\n" + "-" * 78)
    print(f"  ranked    -> {fcsv}")
    print(f"  finalists -> {fjson}")
    if f1:
        print(f"  context   -> phase3_context_existing_*.csv  "
              f"({n1} EXISTING rows, for comparison ONLY)")
    print("\n  GATE 5: only the ID2_ geometries above are eligible. Every row in the")
    print("  context file already has an FEA result, so choosing one would mean the")
    print("  model had reproduced a number we already had, not designed anything.")
    print("\n  Next: ID_5_VALIDATE_FEA.bat - one real FEA per family, GATE 7 = 15 %.")
    print("=" * 78)


if __name__ == "__main__":
    main()
