"""
gate0_check.py - GATE 0: does predict.py reproduce the training predictions?

THE FAILURE THIS EXISTS TO CATCH
  The exported network is only half the model. The other half is the recipe that
  turns an STL into its input: sample 2048 surface points, centre on their mean,
  divide by the largest extent, append density as a 4th channel, then undo the
  standardisation and the log10 on the way out. Get any one of those subtly wrong
  - centre on the bounding-box centre instead of the mean, scale by the diagonal
  instead of the largest extent, forget the log - and predict.py still returns
  six confident numbers of a believable magnitude. Nothing downstream can tell.

  So: take rows the model was TRAINED on, start again from their ORIGINAL STL
  file, push them through predict.py's public path, and compare with what the
  export itself recorded. If the recipe matches, the numbers match to rounding.

WHY IT SHOULD BE NEAR-EXACT, NOT MERELY CLOSE
  pointcloud_prep samples with a FIXED seed, so re-sampling the same STL returns
  the same points. Dropout is off and BatchNorm is frozen in eval(). The chain is
  deterministic end to end, so agreement should land near 1e-5, not near 1%.
  The 1% threshold is headroom for a different trimesh build, nothing more. A
  result at 0.5% is a WARNING sign, not a pass to be pleased about.

  Density is taken from the reference row, deliberately. This gate tests the
  geometry-to-prediction chain; where a NEW candidate's density comes from is a
  separate question the model card answers.

Usage:  python gate0_check.py [--n 10] [--tol 0.01] [--profile COMBINED]
Exit code 0 = PASS, 1 = FAIL. The .bat stops on a non-zero exit.
"""

from __future__ import annotations
import os, sys, argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_paths                               # noqa: E402
import predict as P                            # noqa: E402


def find_stl(stem, folders):
    """Locate <stem>.stl under any configured STL folder. ADMS files are flat,
    the TPMS ones sit one folder per topology, so this walks the tree."""
    target = f"{stem}.stl".lower()
    for root_dir in folders:
        if not root_dir or not os.path.isdir(root_dir):
            continue
        for root, _dirs, files in os.walk(root_dir):
            for f in files:
                if f.lower() == target:
                    return os.path.join(root, f)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--tol", type=float, default=0.01, help="max allowed relative error")
    ap.add_argument("--profile")
    ap.add_argument("--model")
    a = ap.parse_args()

    prof = a.profile or run_paths.profile() or "COMBINED"
    out_dir = os.path.join(run_paths.BASE_DIR, "Final Model", prof)
    ref_csv = os.path.join(out_dir, "gate0_reference.csv")
    if not os.path.exists(ref_csv):
        sys.exit(f"[E] {ref_csv} not found - run export_forward_model.py first")

    b = P.load_bundle(a.model, prof)
    ref = pd.read_csv(ref_csv)
    tgts = b["targets"]
    stem_col = "Run" if "Run" in ref.columns else ref.columns[0]

    print("=" * 68)
    print(f"  GATE 0  round-trip   profile={prof}   tolerance={a.tol * 100:.2f}%")
    print("=" * 68)

    # Deterministic, spread across the file so both families are represented
    # even though the dataset is ordered ADMS-then-TPMS.
    n = min(a.n, len(ref))
    idx = np.unique(np.linspace(0, len(ref) - 1, n).round().astype(int))
    folders = b.get("stl_folders") or []
    print(f"  STL folders searched:")
    for f in folders:
        print(f"    {'OK ' if os.path.isdir(f) else 'MISSING'}  {f}")

    rows, worst, missing = [], 0.0, []
    for i in idx:
        r = ref.iloc[int(i)]
        stem = str(r[stem_col])
        stl = find_stl(stem, folders)
        if stl is None:
            missing.append(stem)
            print(f"  [skip] {stem}: no .stl found")
            continue
        dens = float(r["Rho_rel"]) if "Rho_rel" in ref.columns else None
        pred, rep = P.predict_stl(stl, dens, b)
        rec = {stem_col: stem, "family": r.get("family", ""), "stl": stl}
        errs = []
        for t in tgts:
            exp = float(r[f"pred_{t}"])
            got = float(pred[t])
            e = abs(got - exp) / abs(exp) if exp != 0 else abs(got - exp)
            errs.append(e)
            rec[f"expected_{t}"] = exp
            rec[f"roundtrip_{t}"] = got
            rec[f"relerr_{t}"] = e
        m = max(errs)
        rec["max_relerr"] = m
        worst = max(worst, m)
        rows.append(rec)
        print(f"  {stem[:44]:<44} max rel err {m:.3e}  {'ok' if m <= a.tol else 'FAIL'}")

    if not rows:
        print("\n  GATE 0: FAIL - no rows could be checked (no STL was found).")
        print("  The bundle's stl_folders came from ML_settings.xlsx sheets "
              "'settings*'. Fix those paths and re-export.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    out = os.path.join(out_dir, "GATE0_roundtrip.csv")
    df.to_csv(out, index=False)

    ok = worst <= a.tol
    print("-" * 68)
    print(f"  rows checked : {len(rows)}" + (f"   (skipped {len(missing)})" if missing else ""))
    print(f"  worst error  : {worst:.3e}   ({worst * 100:.4f} %)")
    print(f"  written      : {out}")
    if ok and worst > a.tol / 10:
        print("  [!] inside tolerance but larger than expected for a deterministic")
        print("      chain - check that trimesh is the same build used for training.")
    print(f"\n  GATE 0: {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("  predict.py does NOT reproduce the training predictions. Do not use")
        print("  this model for inverse design until the prep chain is reconciled.")
    print("=" * 68)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
