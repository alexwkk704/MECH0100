r"""
compare_cv_vs_8020.py — side-by-side of the two PointNet validation schemes.

Reads, from the current run folder (runs\<stamp>\):
  * pointnet_v2_results.csv     grouped 5-fold CV      (from train_pointnet_v2.py)
  * pointnet_8020_results.csv   stratified 80/20       (from pointnet_split_8020.py)
  * pointnet_timings.csv        wall-clock per scheme  (written by RUN_ALL_ML.bat)

Writes cv_vs_8020_comparison.csv (per-target R2/MAPE for both schemes) and
cv_vs_8020_comparison.png (R2 bars per target + a runtime note). Prints a table.

Both schemes are INTERPOLATION validations (the model sees every topology). The
grouped CV additionally blocks near-duplicate leakage (same adms_type+density_param
kept together), which the type-only 80/20 does not — so expect the 80/20 to be equal
to or slightly rosier. This does NOT measure generalisation to an unseen topology;
that is the separate ADMS->TPMS test.

Run by RUN_ALL_ML.bat Step 9, or standalone:  python compare_cv_vs_8020.py
"""

import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_paths

CV_FILE = "pointnet_v2_results.csv"
S_FILE = "pointnet_8020_results.csv"
T_FILE = "pointnet_timings.csv"


def _read(name):
    p = run_paths.run_path(name)
    if not os.path.exists(p):
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def main():
    cv = _read(CV_FILE)
    sp = _read(S_FILE)
    if cv is None or sp is None:
        miss = CV_FILE if cv is None else S_FILE
        sys.exit(f"[E] missing {miss} in the run folder — run steps 6 and 8 first.")

    cv = cv[["target", "R2", "MAPE", "N_test"]].rename(
        columns={"R2": "R2_groupedCV", "MAPE": "MAPE_groupedCV", "N_test": "N_CV"})
    sp = sp[["target", "R2", "MAPE", "N_test"]].rename(
        columns={"R2": "R2_8020", "MAPE": "MAPE_8020", "N_test": "N_8020"})
    m = cv.merge(sp, on="target", how="outer")
    m["dR2_8020_minus_CV"] = (m["R2_8020"] - m["R2_groupedCV"]).round(4)

    # ---- timings ----
    tim = _read(T_FILE)
    sec_cv = sec_sp = None
    if tim is not None and {"scheme", "seconds"}.issubset(tim.columns):
        d = {str(r["scheme"]): float(r["seconds"]) for _, r in tim.iterrows()}
        sec_cv = d.get("grouped5foldCV"); sec_sp = d.get("stratified8020")

    print("\n=== PointNet: grouped 5-fold CV  vs  stratified 80/20 (per target) ===")
    cols = ["target", "R2_groupedCV", "R2_8020", "dR2_8020_minus_CV",
            "MAPE_groupedCV", "MAPE_8020"]
    print(m[cols].to_string(index=False))

    print("\n--- runtime ---")
    if sec_cv is not None and sec_sp is not None:
        ratio = (sec_cv / sec_sp) if sec_sp else float("nan")
        print(f"  grouped 5-fold CV : {sec_cv:8.0f} s")
        print(f"  stratified 80/20  : {sec_sp:8.0f} s")
        print(f"  CV is {ratio:.1f}x the 80/20 (expected ~n_folds, since CV trains a "
              f"model per fold and 80/20 trains one)")
    else:
        print("  (no pointnet_timings.csv — run via RUN_ALL_ML.bat to capture wall-clock)")

    out = run_paths.run_path("cv_vs_8020_comparison.csv")
    m_out = m[cols].copy()
    if sec_cv is not None:
        m_out["seconds_groupedCV"] = sec_cv
        m_out["seconds_8020"] = sec_sp
    m_out.to_csv(out, index=False)
    print(f"\n[OK] wrote {out}")

    # ---- chart: R2 bars per target for both schemes ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        mm = m.dropna(subset=["R2_groupedCV", "R2_8020"], how="all").copy()
        t = list(mm["target"]); x = np.arange(len(t)); w = 0.38
        fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(t)), 4.6))
        ax.bar(x - w / 2, mm["R2_groupedCV"].fillna(0), w, label="grouped 5-fold CV", color="#500778")
        ax.bar(x + w / 2, mm["R2_8020"].fillna(0), w, label="stratified 80/20", color="#00b8a9")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(x); ax.set_xticklabels(t, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("R2"); ax.set_ylim(top=1.0)
        title = "PointNet accuracy: grouped 5-fold CV vs stratified 80/20"
        if sec_cv is not None and sec_sp is not None:
            title += f"   (runtime: CV {sec_cv:.0f}s vs 80/20 {sec_sp:.0f}s)"
        ax.set_title(title, color="#500778", fontweight="bold", fontsize=10)
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        cp = run_paths.run_path("cv_vs_8020_comparison.png")
        fig.savefig(cp, dpi=150); plt.close(fig)
        print(f"[OK] wrote {cp}")
    except Exception as ex:
        print("[!] chart skipped:", ex)


if __name__ == "__main__":
    main()
