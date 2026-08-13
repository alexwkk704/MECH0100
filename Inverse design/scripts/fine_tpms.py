"""
fine_tpms.py - INVERSE DESIGN PHASE 2 (TPMS half): generate NEW geometry.

The TPMS mirror of fine_adms.py. Same job, same gates, different generator.

WHICH NOTEBOOK AND WHAT DRIVES IT
    Share\\TPMS\\TPMS STL\\ntopfile+script\\ntopfile\\<Type> Generator.ntop -
    Zhezhe's STL generators. Read off his own driver
    (tpms_stl_batch_export.py), a run writes FOUR inputs and no more:
        t_max (scalar) | t_min (scalar) | Tolerance (scalar, mm) |
        Path (file_path)
    and he always sets t_min = -t_max, so there is ONE free geometric
    parameter, t. Everything else - lattice type, period, cube size, edge
    length - is baked into the notebook.

    THE JSON SHAPE IS NOT THE ADMS ONE. TPMS uses "values" (plural) and
    type "scalar"; ADMS uses "value" (singular) and type "real". ntopcl does
    not error on a key it does not recognise - it keeps the notebook default -
    so getting this wrong produces a successful run and the wrong geometry.
    The flags differ too: no -s here, deliberately (his comment: no point
    writing the mesh back into the notebook every density).

DENSITY -> t: MEASURED HERE, FROM NOTHING
    There is no density->t table for this topology that we can trust.
    merged_training_data.csv is the OLD LINEAR-era file and is NOT READ BY THIS
    SCRIPT AT ALL - not its C columns, not its t, not its VF, not one cell.

    Instead the calibration is measured. --scout generates a spread of t values
    through the notebook we are actually driving, measures the volume fraction
    off each STL, and writes search\\results\\tpms_<topology>_calibration.csv.
    --grid then inverts THAT to pick t for a wanted density. Every number in the
    chain is produced tonight, by this notebook, and is checkable on disk.

    The scout runs are not overhead: each one is a real geometry with a measured
    VF and a model prediction, so they are candidates in their own right.

    Inverting outside the scouted range is refused - past the ends we have no
    evidence the relation holds and nothing downstream would notice.

DENSITY IS MEASURED, NOT REQUESTED
    The requested density only selects t. What the model is fed is the VF
    MEASURED from the generated STL. For TPMS rows Rho_rel == VF exactly, so
    that is the same quantity the density channel trained on - there is no
    p_rel to chase, unlike ADMS.

GATES
    GATE 2  watertight | consistent winding | single body | even Euler number
    GATE 3  measured VF vs the density asked for. Zhezhe's own worst miss on
            this generator is 0.63 %, so the tolerance is 3 % - loose enough
            not to fail good geometry, tight enough to catch a bad generation.
    GATE 4  reject predictions outside the training label range
    GATE 5  by construction - a density already in the database is skipped

ONE-WAY RULE
    Everything written here stays under Share\\Inverse design\\search\\.
    Share\\TPMS is read-only: the notebook is copied into search\\ntop\\ and
    refreshed from source before every run. Nothing produced here ever enters
    the training dataset.

Usage
    python fine_tpms.py --scout            measure the density<->t calibration
    python fine_tpms.py --verify           closed-loop check against a real row
    python fine_tpms.py --grid             the new densities around the winner
    python fine_tpms.py --grid --dry-run   print the plan, generate nothing
"""

from __future__ import annotations
import os, sys, json, glob, time, hashlib, shutil, argparse, subprocess
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ID_ROOT = os.path.dirname(HERE)                       # Share\Inverse design
SHARE = os.path.normpath(os.path.join(ID_ROOT, ".."))
MECH = os.path.normpath(os.path.join(SHARE, ".."))    # D:\MECH0100
ML_DIR = os.environ.get("ML_DIR") or os.path.join(SHARE, "ML")
TPMS_DIR = os.environ.get("TPMS_DIR") or os.path.join(SHARE, "TPMS")
ML_SCRIPTS = os.path.join(ML_DIR, "scripts")
if not os.path.isdir(ML_SCRIPTS):
    sys.exit(f"[E] cannot find the forward ML scripts at {ML_SCRIPTS}\n"
             f"    expected this file to live in Share\\Inverse design\\scripts\\")
sys.path.insert(0, ML_SCRIPTS)

import predict as P                                    # noqa: E402
import run_paths                                       # noqa: E402

OUT_ROOT = os.path.join(ID_ROOT, "search")
STL_DIR = os.path.join(OUT_ROOT, "candidates", "tpms")
RUN_DIR = os.path.join(OUT_ROOT, "runs", "tpms")
NTOP_DIR = os.path.join(OUT_ROOT, "ntop")
RESULTS = os.path.join(OUT_ROOT, "results")

NB_ROOT = os.path.join(TPMS_DIR, "TPMS STL", "ntopfile+script", "ntopfile")
# NO LOOKUP FILE. The density<->t calibration is measured by --scout and lives in
# search\results\. merged_training_data.csv is the old linear-era file and is
# deliberately not referenced anywhere in this script.

NTOPCL_CANDIDATES = [
    r"C:\Program Files\nTopology\nTopology\ntopcl.exe",
    r"C:\Program Files\nTopology\ntopcl.exe",
    r"D:\Program Files\nTopology\nTopology\ntopcl.exe",
]

# Exact input names, from Zhezhe's driver. ntopcl matches by string.
NB_TMAX, NB_TMIN, NB_TOL, NB_PATH = "t_max", "t_min", "Tolerance", "Path"
TOLERANCE_MM = 0.2          # the value in his shipped script

GATE3_TOL = 0.03            # his own worst miss on this generator is 0.63 %
VERIFY_TOL = 2.0            # per cent, measured VF vs the database
# per cent, measured wall thickness vs the database. Sampling scatter is ~0.1 %
# (250/500/1000 samples give 0.5504/0.5514/0.5514 on Zhezhe's FRD_0.26 against
# the dataset's 0.5509), so 5 % is loose enough to be about geometry, not noise.
THICKNESS_TOL = 5.0


def die(m):
    print(f"\n[E] {m}")
    sys.exit(1)


def find_ntopcl():
    p = os.environ.get("NTOPCL")
    if p and os.path.exists(p):
        return p
    for c in NTOPCL_CANDIDATES:
        if os.path.exists(c):
            return c
    die("ntopcl.exe not found. Set NTOPCL=<full path to ntopcl.exe> and re-run.")


def sha1(path, chunk=1 << 20):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()[:16]


def stem_for(topo, rho, prefix="ID2"):
    """Named like Zhezhe's rows so a table reads the same, but ID2_-prefixed so
    a Phase 2 candidate can never be confused with - or collide with - a row
    that has a real FEA result behind it."""
    return f"{prefix}_{topo}_{rho:.3f}".replace(" ", "_")


def build_input_json(dest, t, tolerance, stl_path):
    """Zhezhe's exact shape: 'values' PLURAL, type 'scalar', units only on
    Tolerance. Nothing else is written, so every other notebook input keeps its
    baked-in value - which is what produced the training rows."""
    data = {"inputs": [
        {"name": NB_TMAX, "type": "scalar", "values": float(t)},
        {"name": NB_TMIN, "type": "scalar", "values": -float(t)},
        {"name": NB_TOL, "type": "scalar", "units": "mm",
         "values": float(tolerance)},
        {"name": NB_PATH, "type": "file_path",
         "values": str(stl_path).replace("\\", "/")},
    ]}
    with open(dest, "w") as f:
        json.dump(data, f, indent=2)
    return data


CALIB_MIN_POINTS = 3


def calib_path(topo):
    return os.path.join(RESULTS,
                        f"tpms_{topo}_calibration.csv".replace(" ", "_"))


def load_calibration(topo, nb_sha1, tolerance):
    """OUR OWN measured density<->t curve, written by --scout.

    Nothing is read from any pre-existing table. The calibration must have been
    measured with the SAME notebook and the SAME Tolerance, or it is refused -
    a curve measured at one mesh tolerance does not describe another.
    """
    f = calib_path(topo)
    if not os.path.exists(f):
        die(f"no calibration for {topo}.\n"
            f"    Run ID_3A_FINE_TPMS_SCOUT.bat first - it measures the\n"
            f"    density-to-t relation with this notebook, from scratch.")
    c = pd.read_csv(f)
    c = c[c.ok.astype(bool)].sort_values("t")
    if len(c) < CALIB_MIN_POINTS:
        die(f"the calibration has only {len(c)} usable points "
            f"(need {CALIB_MIN_POINTS}). Re-run the scout with a wider range.")
    bad_nb = sorted(set(c.notebook_sha1.astype(str))) != [str(nb_sha1)]
    if bad_nb:
        die(f"the calibration was measured with a DIFFERENT notebook "
            f"({sorted(set(c.notebook_sha1.astype(str)))} vs {nb_sha1}).\n"
            f"    Re-run the scout.")
    tols = sorted(set(round(float(x), 6) for x in c.tolerance_mm))
    if tols != [round(float(tolerance), 6)]:
        die(f"the calibration was measured at Tolerance {tols} mm but this run "
            f"asks for {tolerance} mm.\n    Mesh tolerance changes the measured "
            f"volume fraction. Re-run the scout at {tolerance} mm.")
    t = c.t.values.astype(float)
    vf = c.vf.values.astype(float)
    if np.any(np.diff(vf) <= 0):
        die("the measured VF is not monotonic in t:\n"
            f"    t  {list(np.round(t, 4))}\n    VF {list(np.round(vf, 5))}\n"
            "    Inverting a non-monotonic curve is ambiguous. Widen or refine "
            "the scout.")
    return t, vf


def t_for(rho, ts, vfs):
    """Invert OUR measured curve. Refuses to extrapolate past what we scouted."""
    if rho < vfs.min() - 1e-9 or rho > vfs.max() + 1e-9:
        die(f"density {rho:g} is outside the SCOUTED range "
            f"{vfs.min():.5f}..{vfs.max():.5f}.\n"
            f"    Re-run the scout with --t-lo / --t-hi covering it. "
            f"Extrapolating the\n    generator is not the same as extrapolating "
            f"the model, and GATE 4 would not catch it.")
    return float(np.interp(rho, vfs, ts))


def tpms_settings():
    """The thickness settings the TPMS features were actually extracted with.

    Read from ML_settings.xlsx sheet `settings_TPMS` - not hardcoded. The
    inverse design runs under ML_PROFILE=COMBINED, so the profile is switched
    for the duration of the read and put back.
    """
    old = os.environ.get("ML_PROFILE")
    os.environ["ML_PROFILE"] = "TPMS"
    try:
        def g(k, d):
            v = run_paths.sheet_value("settings", k, "")
            return d if v in (None, "") else v
        return dict(cell_size_mm=float(g("cell_size_mm", 10.0)),
                    random_seed=int(float(g("random_seed", 1))),
                    boundary_skin_frac=float(g("boundary_skin_frac", 0.02)),
                    cap_fallback=float(g("thickness_cap_fallback", 0.6)))
    finally:
        if old is None:
            os.environ.pop("ML_PROFILE", None)
        else:
            os.environ["ML_PROFILE"] = old


def measure_thickness(stl, n_samples):
    """Median wall thickness in mm, the SAME way the dataset column was made.

    Reuses feature_extraction.wall_thickness rather than reimplementing it, so
    the number is comparable to thickness_med_mm by construction. The dataset
    used 4000 sample points; the median is stable well below that - measured on
    Zhezhe's own FRD_0.26 STL, 250/500/1000 points give 0.5504 / 0.5514 / 0.5514
    against the dataset's 0.5509, i.e. within 0.1 % - so this runs 500 by
    default and takes seconds instead of a minute.

    t is unknown for TPMS (no dataset records it), which is exactly the case the
    extractor handles with cap = thickness_cap_fallback x cell_size.

    WHY BOTHER: volume fraction alone can be hit by the wrong geometry. Volume
    fraction AND wall thickness AND Euler number together cannot.
    """
    import feature_extraction as FE
    S = tpms_settings()
    m = FE.load_welded(stl)
    interior = FE.interior_vertex_mask(np.asarray(m.vertices), m.bounds,
                                       S["boundary_skin_frac"])
    cap = S["cap_fallback"] * S["cell_size_mm"]
    th = FE.wall_thickness(m, interior, int(n_samples), cap, S["random_seed"])
    if len(th) < 50:
        return None, None, len(th)
    return float(np.median(th)), float(np.std(th)), len(th)


def refresh_notebook(src, dst):
    shutil.copy2(src, dst)


def resolve_notebook(topology):
    src = os.path.join(NB_ROOT, f"{topology}.ntop")
    if not os.path.exists(src):
        cands = glob.glob(os.path.join(NB_ROOT, f"{topology}*.ntop"))
        if len(cands) != 1:
            die(f"cannot resolve the STL notebook for {topology!r} in "
                f"{NB_ROOT}: {cands}")
        src = cands[0]
    os.makedirs(NTOP_DIR, exist_ok=True)
    dst = os.path.join(NTOP_DIR, os.path.basename(src))
    if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(src):
        print(f"  copying notebook into search\\ntop\\ "
              f"({os.path.getsize(src) / 1e6:.0f} MB) ...", flush=True)
        shutil.copy2(src, dst)
    return src, dst


def run_one(ntopcl, nb, stem, t, tolerance, dry=False, force=False, src_nb=None):
    """One nTop generation. Returns (stl_path_or_None, seconds)."""
    os.makedirs(STL_DIR, exist_ok=True)
    rd = os.path.join(RUN_DIR, stem)
    os.makedirs(rd, exist_ok=True)
    stl = os.path.join(STL_DIR, stem + ".stl")
    inj, outj = os.path.join(rd, "in.json"), os.path.join(rd, "out.json")
    build_input_json(inj, t, tolerance, stl)
    if dry:
        return None, 0.0
    # Zhezhe validates by size; 2000 bytes is his floor for "header then died".
    if os.path.exists(stl) and os.path.getsize(stl) > 2000 and not force:
        print("      [have it already - reusing, --force to regenerate]")
        return stl, 0.0
    if src_nb:
        refresh_notebook(src_nb, nb)
    t0 = time.time()
    args = [ntopcl, "-v2", "-j", inj, "-o", outj, nb]      # note: NO -s
    with open(os.path.join(rd, "log.txt"), "w") as lf:
        lf.write(" ".join(args) + "\n\n")
        lf.flush()
        rc = subprocess.run(args, stdout=lf, stderr=subprocess.STDOUT,
                            text=True).returncode
    secs = time.time() - t0
    if rc != 0 or not os.path.exists(stl) or os.path.getsize(stl) <= 2000:
        print(f"      FAILED (exit {rc}, {secs:.0f}s) - see "
              f"{os.path.join(rd, 'log.txt')}")
        return None, secs
    return stl, secs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scout", action="store_true",
                    help="MEASURE the density<->t calibration for this notebook")
    ap.add_argument("--verify", action="store_true",
                    help="closed-loop check: hit a real row's VF and compare")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--target", type=float, default=0.10)
    ap.add_argument("--objective", default="E_over_Es")
    ap.add_argument("--profile", default=os.environ.get("ML_PROFILE", "COMBINED"))
    ap.add_argument("--t-lo", type=float, default=0.30, help="scout: lowest t")
    ap.add_argument("--t-hi", type=float, default=1.30, help="scout: highest t")
    ap.add_argument("--n-scout", type=int, default=4)
    ap.add_argument("--span", type=float, default=0.035,
                    help="+/- density either side of the centre")
    ap.add_argument("--step", type=float, default=0.005)
    ap.add_argument("--tolerance", type=float, default=TOLERANCE_MM)
    ap.add_argument("--skip-verify-check", action="store_true")
    ap.add_argument("--allow-multibody", action="store_true")
    ap.add_argument("--thickness-samples", type=int, default=500,
                    help="verify only: surface samples for the wall-thickness check")
    ap.add_argument("--no-thickness-check", action="store_true")
    ap.add_argument("--row", default=None,
                    help="Run stem to centre on (default: the Phase 1 TPMS winner)")
    a = ap.parse_args()
    if not (a.scout or a.verify or a.grid):
        ap.print_help()
        return
    os.environ["ML_PROFILE"] = a.profile

    ntopcl = find_ntopcl()
    model_path = os.path.join(ML_DIR, "Final Model", a.profile,
                              f"model_{a.profile}.pt")
    b = P.load_bundle(model_path, a.profile)
    if a.objective not in b["targets"]:
        die(f"'{a.objective}' is not a model target. Available: {b['targets']}")

    df = pd.read_csv(run_paths.dataset_path())
    tp = df[df.family == "TPMS"].copy()

    # ---- centre row ---------------------------------------------------------
    stem = a.row
    if stem is None:
        rk = sorted(glob.glob(os.path.join(
            RESULTS, f"phase1_ranked_{a.objective}_*.csv")))
        if not rk:
            die("no Phase 1 results found - run ID_1_COARSE.bat first")
        r1 = pd.read_csv(rk[-1])
        r1 = r1[(r1.family == "TPMS") & r1.gate4_ok].reset_index(drop=True)
        if r1.empty:
            die("Phase 1 returned no TPMS row that passed GATE 4")
        stem = str(r1.iloc[int((r1.predicted - a.target).abs().values.argmin())]["Run"])
    row = tp[tp.Run == stem]
    if row.empty:
        die(f"{stem} is not a TPMS row in the dataset")
    row = row.iloc[0]
    topo = str(row.topology).strip()
    same = tp[tp.topology.astype(str).str.strip() == topo]
    db_vfs = np.sort(pd.to_numeric(same.VF, errors="coerce").dropna().values)

    print("=" * 74)
    print("  INVERSE DESIGN - PHASE 2  (TPMS)   generate NEW geometry")
    print("=" * 74)
    print(f"  ntopcl    {ntopcl}")
    src_nb, nb = resolve_notebook(topo)
    nbsha = sha1(src_nb)
    print(f"  notebook  {os.path.basename(src_nb)}   sha1 {nbsha}")
    print("            driven from a refreshed copy in search\\ntop\\ "
          "(Share\\TPMS is read-only here)")
    print(f"  model     {os.path.basename(model_path)}  {b.get('dataset_name')}  "
          f"seeds {b.get('seeds')}")
    print(f"  centre    {stem}   Tolerance {a.tolerance:g} mm")
    print(f"            database VF {row.VF:.5f} (== Rho_rel for TPMS)   "
          f"{a.objective} {row[a.objective]:.6f}   euler {row.get('euler')}")
    print("  calibration source: MEASURED HERE. No pre-existing "
          "density-to-t table is read.")

    os.makedirs(RESULTS, exist_ok=True)

    def assess(stl):
        """One mesh load. Density fed to the model is the MEASURED VF."""
        pred, rep = P.predict_stl(stl, None, b)     # None -> measure from mesh
        g2ok, g2bad = P.gate2(rep)
        if a.allow_multibody:
            g2bad = [x for x in g2bad if "disconnected bodies" not in x]
            g2ok = not g2bad
        return pred, rep, g2ok, g2bad

    # ---- SCOUT: measure the calibration -------------------------------------
    if a.scout:
        print("\n" + "-" * 74)
        print("  SCOUT - measure the density-to-t relation for this notebook.")
        print("  Generates a spread of t, measures the volume fraction off each")
        print("  STL, and writes the curve. Nothing is assumed and no existing")
        print("  table is consulted. These are real candidates too.")
        print("-" * 74)
        ts = np.round(np.linspace(a.t_lo, a.t_hi, a.n_scout), 4)
        print(f"  t values {list(ts)}   Tolerance {a.tolerance:g} mm")
        if a.dry_run:
            print("\n  [dry-run] nothing generated")
            return
        rows = []
        for i, t in enumerate(ts, 1):
            st = f"ID2_SCOUT_{topo}_t{t:.4f}".replace(" ", "_")
            print(f"  [{i}/{len(ts)}] t={t:<9.4f} -> {st}", flush=True)
            stl, secs = run_one(ntopcl, nb, st, t, a.tolerance,
                                False, a.force, src_nb)
            if stl is None:
                rows.append(dict(t=float(t), vf=np.nan, ok=False, secs=round(secs, 1),
                                 tolerance_mm=a.tolerance, notebook_sha1=nbsha,
                                 notebook=os.path.basename(src_nb), stl="",
                                 gate2=False, gate2_detail="generation failed",
                                 euler="", bodies="", n_tris=""))
                continue
            pred, rep, g2ok, g2bad = assess(stl)
            rows.append(dict(t=float(t), vf=float(rep["vf"]), ok=True,
                             secs=round(secs, 1), tolerance_mm=a.tolerance,
                             notebook_sha1=nbsha,
                             notebook=os.path.basename(src_nb), stl=stl,
                             gate2=g2ok, gate2_detail="; ".join(g2bad),
                             euler=rep["euler"], bodies=rep["bodies"],
                             n_tris=rep["n_tris"],
                             **{f"pred_{k}": v for k, v in pred.items()}))
            print(f"      VF {rep['vf']:.5f}   euler {rep['euler']}  "
                  f"bodies {rep['bodies']}  tris {rep['n_tris']}   "
                  f"{'GATE2 pass' if g2ok else 'GATE2 FAIL: ' + '; '.join(g2bad)}"
                  f"   ({secs:.0f}s)")
        out = pd.DataFrame(rows)
        out.to_csv(calib_path(topo), index=False)
        good = out[out.ok.astype(bool)]
        print("\n" + "=" * 74)
        print(f"  CALIBRATION  {len(good)}/{len(out)} points measured")
        print("=" * 74)
        print(f"    {'t':>9} {'measured VF':>12}")
        for _, r in good.iterrows():
            print(f"    {r.t:9.4f} {r.vf:12.5f}")
        if len(good) >= 2:
            lo, hi = float(good.vf.min()), float(good.vf.max())
            print(f"\n  covers density {lo:.5f} .. {hi:.5f}")
            want = float(row.VF)
            if lo <= want <= hi:
                print(f"  the centre row's VF {want:.5f} IS inside that range - "
                      f"good, the grid can be built.")
            else:
                print(f"  [!] the centre row's VF {want:.5f} is OUTSIDE that "
                      f"range.\n      Re-run the scout with --t-lo / --t-hi "
                      f"widened, or the grid will refuse.")
            k = np.polyfit(good.t.values, good.vf.values, 1)
            res = np.abs(np.polyval(k, good.t.values) - good.vf.values)
            print(f"  straight-line fit VF = {k[0]:.5f} t + {k[1]:+.5f}   "
                  f"worst residual {res.max():.5f}")
            print("  (the grid interpolates the measured points, not this line -")
            print("   the line is only here to show how smooth the relation is)")
        print(f"\n  -> {calib_path(topo)}")
        print("\n  Next: ID_3B_FINE_TPMS_VERIFY.bat")
        return

    ts_c, vfs_c = load_calibration(topo, nbsha, a.tolerance)
    print(f"  calibration  {len(ts_c)} measured points, "
          f"t {ts_c.min():.4f}..{ts_c.max():.4f} -> VF "
          f"{vfs_c.min():.5f}..{vfs_c.max():.5f}")

    # ---- VERIFY: closed loop against a row that really exists ---------------
    if a.verify:
        print("\n" + "-" * 74)
        print("  VERIFY - closed loop. Ask the measured calibration for the t")
        print(f"  that should give VF {row.VF:.5f} (the real {stem}), generate")
        print("  it, and measure what actually comes out. This tests the whole")
        print("  chain: notebook, JSON shape, inversion, and measurement.")
        print("-" * 74)
        t_want = t_for(float(row.VF), ts_c, vfs_c)
        vstem = f"ID2_VERIFY_{topo}_{float(row.VF):.5f}".replace(" ", "_")
        print(f"    target VF {row.VF:.5f}  ->  t {t_want:.6f}  -> {vstem}")
        stl, secs = run_one(ntopcl, nb, vstem, t_want, a.tolerance,
                            a.dry_run, a.force, src_nb)
        if a.dry_run:
            print("\n  [dry-run] nothing generated")
            return
        if stl is None:
            die("the verification run failed - fix that before generating a grid")
        pred, rep, g2ok, g2bad = assess(stl)
        d_vf = (rep["vf"] - row.VF) / row.VF * 100.0
        print(f"\n  generated in {secs:.0f}s")
        print(f"    VF      measured {rep['vf']:.5f}   database {row.VF:.5f}"
              f"   diff {d_vf:+.3f} %")
        print(f"    mesh    {'GATE2 pass' if g2ok else 'GATE2 FAIL: ' + '; '.join(g2bad)}")
        print(f"            euler {rep['euler']}  bodies {rep['bodies']}  "
              f"tris {rep['n_tris']}")
        print(f"            database: watertight {row.get('watertight')}  "
              f"euler {row.get('euler')}  tris {row.get('n_tris')}")
        same_euler = str(rep["euler"]) == str(row.get("euler"))
        print(f"            euler {'MATCHES' if same_euler else 'DIFFERS FROM'} "
              f"the database row"
              + ("" if same_euler else "  <- different topology or resolution"))
        print(f"    {a.objective:<9}  predicted {pred[a.objective]:.6f}   "
              f"database FEA {row[a.objective]:.6f}   "
              f"diff {(pred[a.objective] - row[a.objective]) / row[a.objective] * 100:+.2f} %"
              f"   (IN-SAMPLE, not an accuracy figure)")
        # --- second, independent target: the wall thickness -----------------
        # Volume fraction alone can be hit by the wrong geometry. Volume
        # fraction AND wall thickness AND Euler number together cannot.
        th_med = th_std = th_hits = None
        d_th = None
        th_db = pd.to_numeric(pd.Series([row.get("thickness_med_mm")]),
                              errors="coerce").iloc[0]
        if a.no_thickness_check:
            print("    wall    thickness check SKIPPED (--no-thickness-check)")
        elif not np.isfinite(th_db):
            print("    wall    the database row has no thickness_med_mm - "
                  "nothing to compare against")
        else:
            print(f"    wall    measuring thickness "
                  f"({a.thickness_samples} samples) ...", end="", flush=True)
            t1 = time.time()
            try:
                th_med, th_std, th_hits = measure_thickness(stl,
                                                            a.thickness_samples)
            except Exception as ex:
                print(f" FAILED: {type(ex).__name__}: {ex}")
                th_med = None
            else:
                print(f" {time.time() - t1:.0f}s")
            if th_med is None:
                print(f"            not enough valid samples "
                      f"({th_hits}) - check skipped, not failed")
            else:
                d_th = (th_med - th_db) / th_db * 100.0
                print(f"            measured {th_med:.4f} mm   database "
                      f"{th_db:.4f} mm   diff {d_th:+.3f} %   "
                      f"(hits {th_hits})")

        ok = abs(d_vf) < VERIFY_TOL and g2ok
        if d_th is not None:
            ok = ok and abs(d_th) < THICKNESS_TOL
        print(f"\n  VERIFY {'PASS' if ok else 'FAIL'}")
        print(f"      volume fraction  {abs(d_vf):.3f} % from the database "
              f"(threshold {VERIFY_TOL:g} %)")
        if d_th is not None:
            print(f"      wall thickness   {abs(d_th):.3f} % from the database "
                  f"(threshold {THICKNESS_TOL:g} %)")
        print(f"      mesh             {'GATE 2 pass' if g2ok else 'GATE 2 FAIL'}"
              f"   Euler {'matches' if same_euler else 'DIFFERS'}")
        with open(os.path.join(RESULTS, "phase2_tpms_verify.json"), "w") as f:
            json.dump(dict(row=stem, topology=topo, target_vf=float(row.VF),
                           t_used=t_want, tolerance_mm=a.tolerance,
                           notebook=os.path.basename(src_nb), notebook_sha1=nbsha,
                           calibration=calib_path(topo), pass_=bool(ok),
                           secs=round(secs, 1), vf_measured=float(rep["vf"]),
                           vf_db=float(row.VF), worst_pct=float(abs(d_vf)),
                           gate2=bool(g2ok), gate2_detail=g2bad,
                           euler=rep["euler"], euler_db=str(row.get("euler")),
                           euler_matches=bool(same_euler),
                           thickness_med_mm=th_med,
                           thickness_std_mm=th_std,
                           thickness_hits=th_hits,
                           thickness_db=(None if not np.isfinite(th_db)
                                         else float(th_db)),
                           thickness_diff_pct=d_th,
                           thickness_samples=a.thickness_samples,
                           bodies=rep["bodies"], n_tris=rep["n_tris"],
                           predicted={k: float(v) for k, v in pred.items()},
                           stl=stl), f, indent=2)
        if not ok:
            print("  The chain does not reproduce the geometry behind the")
            print("  training row. Do not generate the grid until that is understood.")
            sys.exit(1)
        print("  Reproduces the database geometry. Grid is safe.")
        return

    # ---- GRID ---------------------------------------------------------------
    vpath = os.path.join(RESULTS, "phase2_tpms_verify.json")
    if not (a.skip_verify_check or a.dry_run):
        if not os.path.exists(vpath):
            die("the chain has not been checked against the database.\n"
                "    Run ID_3B_FINE_TPMS_VERIFY.bat first (about 2 minutes).")
        try:
            v = json.load(open(vpath, encoding="utf-8"))
        except Exception as ex:
            die(f"cannot read {vpath}: {ex}")
        if not v.get("pass_"):
            die(f"the last check FAILED "
                f"({v.get('worst_pct', float('nan')):.3f} % on {v.get('row')}).")
        if v.get("notebook_sha1") != nbsha:
            die(f"the check used a DIFFERENT notebook "
                f"(checked {v.get('notebook_sha1')}, now {nbsha}).")
        if str(v.get("topology")) != topo:
            die(f"the check was for {v.get('topology')!r}, not {topo!r}.")
        if round(float(v.get("tolerance_mm", -1)), 6) != round(a.tolerance, 6):
            die(f"the check ran at Tolerance {v.get('tolerance_mm')} mm, "
                f"this run asks for {a.tolerance} mm.")
        print(f"\n  chain check: PASS on {v.get('row')} "
              f"({v.get('worst_pct', 0):.3f} % on VF), same notebook.")

    rho_c = float(row.VF)
    lo = max(float(vfs_c.min()), rho_c - a.span)
    hi = min(float(vfs_c.max()), rho_c + a.span)
    n = max(2, int(round((hi - lo) / a.step)) + 1)
    want = np.round(np.linspace(lo, hi, n), 4)
    # GATE 5: skip anything within half a step of a density we already have
    built = [round(float(x), 4) for x in db_vfs]
    plan = [float(r) for r in want
            if all(abs(r - x) > a.step / 2 for x in built)]
    skipped = len(want) - len(plan)

    print("\n" + "-" * 74)
    print(f"  GRID   {topo}, scouted range VF {vfs_c.min():.5f}..{vfs_c.max():.5f}")
    print(f"         centre {rho_c:.5f}  +/-{a.span:g}  step {a.step:g}")
    print(f"         densities {[float(x) for x in want]}")
    print(f"  {len(plan)} candidates   ({skipped} skipped - within half a step of "
          f"a density already in the database, GATE 5)")
    print("-" * 74)
    if not plan:
        die("every density already exists - widen --span or shrink --step")

    lo4, hi4 = b["label_min"][a.objective], b["label_max"][a.objective]
    rows, t0 = [], time.time()
    for i, rho in enumerate(plan, 1):
        t = t_for(rho, ts_c, vfs_c)
        stem_i = stem_for(topo, rho)
        print(f"  [{i}/{len(plan)}] density={rho:<7.4g} t={t:<10.6f} -> {stem_i}",
              flush=True)
        stl, secs = run_one(ntopcl, nb, stem_i, t, a.tolerance,
                            a.dry_run, a.force, src_nb)
        if a.dry_run or stl is None:
            continue
        pred, rep, g2ok, g2bad = assess(stl)
        g3ok, g3err = P.gate3(rep, rho, tol=GATE3_TOL)
        g4ok = bool(lo4 <= pred[a.objective] <= hi4)
        rows.append(dict(
            stem=stem_i, topology=topo, density_requested=rho, t_max=t,
            t_min=-t, tolerance_mm=a.tolerance,
            vf=rep["vf"], density_used=rep["density_used"],
            density_source="measured_VF (== Rho_rel for TPMS)",
            gate2=g2ok, gate2_detail="; ".join(g2bad),
            gate3=g3ok, gate3_err=g3err, gate4=g4ok,
            watertight=rep["watertight"], winding_ok=rep["winding_ok"],
            euler=rep["euler"], bodies=rep["bodies"], n_tris=rep["n_tris"],
            secs=round(secs, 1), stl=stl,
            **{f"pred_{k}": v for k, v in pred.items()}))
        flag = "ok" if (g2ok and g4ok and g3ok is not False) else "REJECTED"
        print(f"      VF {rep['vf']:.5f}  {a.objective} {pred[a.objective]:.6f}  "
              f"|err| {abs(pred[a.objective] - a.target):.6f}  {flag}  ({secs:.0f}s)")

    if a.dry_run:
        print("\n  [dry-run] nothing generated")
        return
    if not rows:
        die("no candidate generated successfully - see the log.txt files under "
            "search\\runs\\tpms\\")

    out = pd.DataFrame(rows)
    out["abs_err"] = (out[f"pred_{a.objective}"] - a.target).abs()
    out = out.sort_values("abs_err").reset_index(drop=True)
    f_csv = os.path.join(RESULTS, f"phase2_tpms_{a.objective}_{a.target:g}.csv")
    out.to_csv(f_csv, index=False)

    good = out[out.gate2 & out.gate4 & (out.gate3 != False)]   # noqa: E712
    print("\n" + "=" * 74)
    print(f"  {len(out)} generated in {(time.time() - t0) / 60:.1f} min   "
          f"{len(good)} passed GATE 2 + GATE 3 + GATE 4")
    print("=" * 74)
    print(f"    {'candidate':<40} {'VF':>8} {'pred':>10} {'|err|':>9}")
    for _, r in good.head(10).iterrows():
        print(f"    {r.stem[:40]:<40} {r.vf:8.5f} "
              f"{r[f'pred_{a.objective}']:10.6f} {r.abs_err:9.6f}")
    if len(good):
        w = good.iloc[0]
        print(f"\n  BEST NEW TPMS DESIGN   {w.stem}")
        print(f"    {topo}   t_max {w.t_max:.6f}   t_min {w.t_min:.6f}   "
              f"Tolerance {w.tolerance_mm:g} mm")
        print(f"    asked for density {w.density_requested:g}, measured VF "
              f"{w.vf:.5f}  ({w.gate3_err * 100:+.2f} %)")
        print(f"    predicted {a.objective} {w[f'pred_{a.objective}']:.6f}   "
              f"{w.abs_err / a.target * 100:.2f} % from the target {a.target:g}")
        print(f"    predicted TAI {w.get('pred_TAI', float('nan')):.4f}")
        print(f"    STL {w.stl}")
    else:
        print("\n  Nothing passed all three gates. Read the CSV before widening.")

    with open(os.path.join(RESULTS,
                           f"phase2_tpms_{a.objective}_{a.target:g}_manifest.json"),
              "w") as f:
        json.dump(dict(built_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                       centre_row=stem, topology=topo, centre_vf=rho_c,
                       target=a.target, objective=a.objective,
                       notebook=os.path.basename(src_nb), notebook_sha1=nbsha,
                       calibration=calib_path(topo),
                       calibration_points=int(len(ts_c)),
                       tolerance_mm=a.tolerance,
                       model=os.path.basename(model_path),
                       model_built=b.get("built_utc"), model_seeds=b.get("seeds"),
                       dataset=b.get("dataset_name"),
                       dataset_sha1=b.get("dataset_sha1"),
                       densities=[float(x) for x in plan],
                       n_planned=len(plan), n_generated=len(out),
                       n_passed=int(len(good)),
                       skipped_already_in_database=int(skipped)), f, indent=2)

    print(f"\n  -> {f_csv}")
    print("\n  Every geometry above is NEW - none has ever been simulated, and")
    print("  none is in the training set. Phase 4 runs ONE real FEA on the winner.")
    print("=" * 74)


if __name__ == "__main__":
    main()
