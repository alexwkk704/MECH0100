"""
fine_adms.py - INVERSE DESIGN PHASE 2 (ADMS half): generate NEW geometry.

Phase 1 found the neighbourhood. This builds a small grid of parameter
combinations that have NEVER been simulated, generates each as an STL through
the trimmed nTop notebook, and scores it with the frozen forward model.

WHY THIS STEP IS THE POINT OF THE WHOLE METHOD
    Every row Phase 1 could return already had an FEA result, so the model was
    only reproducing numbers we already had. Here the model is the ONLY source
    of properties - none of these geometries has ever been simulated. GATE 5
    requires the design carried forward to come from this step.

WHICH NOTEBOOK
    ADMS_<Variant>_only_STL.ntop - Alex's trimmed copy: geometry and STL export
    only, FE and curvature blocks removed, so a candidate costs ~2 min instead
    of a full homogenisation. Its six inputs, read off the notebook:
        Density (real) | Thickness (real, mm) | Seed (integer) |
        Inner Size (real, mm) | Size Multi (real) | STL Path (file path)
    and it outputs p_rel - the MEASURED relative density, which is what the
    model's density channel needs. There is no Cell Size input and that is
    correct: cell size was only the length scale for the curvature features,
    which this notebook no longer computes.

    Nothing here writes to Share\\ADMS. `ntopcl -s` SAVES ITS RESULT BACK INTO
    THE NOTEBOOK - that is why ntop_batch.py copies the notebook per run instead
    of driving the original, and why every database row was generated from a
    pristine file. This does the same: the notebook is copied into search\\ntop\\
    and REFRESHED FROM SOURCE BEFORE EVERY RUN. The generic notebooks,
    ntop_batch.py, input_template_*.json and Results_summary.xlsx are untouched.
    The SHA1 recorded in the manifest is the SOURCE notebook's, so it stays
    meaningful after a run has mutated the copy.

ORDER FOR EVERY CANDIDATE - density is MEASURED, never assumed
    generate STL -> read p_rel out of the run -> mesh checks -> predict

    p_rel is the notebook's own relative density and is what the density
    channel was trained on for ADMS rows. Measured STL VF is recorded too, but
    only as a cross-check: across the 163 ADMS rows the two differ by a median
    0.33 % and up to 6.59 %.

GATES
    GATE 2  watertight | consistent winding | single body | even Euler number
    GATE 3  measured STL VF must agree with the notebook's own p_rel, to 7 %
            (the database's own worst disagreement is 6.59 %)
    GATE 4  reject predictions outside the training label range
    GATE 5  enforced by construction - every point here is a parameter pair
            that is NOT in the database, and the run refuses to include one
            that is.

ONE-WAY RULE
    Everything written here stays under Share\\Inverse design\\search\\. No STL,
    CSV or row produced by this script ever enters the training dataset.

Usage
    python fine_adms.py --verify              reproduce ONE existing row, compare
    python fine_adms.py --grid                the 4 x 4 around the Phase 1 winner
    python fine_adms.py --grid --dry-run      print the plan, generate nothing
"""

from __future__ import annotations
import os, sys, json, glob, time, hashlib, shutil, argparse, subprocess
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ID_ROOT = os.path.dirname(HERE)                       # Share\Inverse design
SHARE = os.path.normpath(os.path.join(ID_ROOT, ".."))
ML_DIR = os.environ.get("ML_DIR") or os.path.join(SHARE, "ML")
ADMS_DIR = os.environ.get("ADMS_DIR") or os.path.join(SHARE, "ADMS")
ML_SCRIPTS = os.path.join(ML_DIR, "scripts")
if not os.path.isdir(ML_SCRIPTS):
    sys.exit(f"[E] cannot find the forward ML scripts at {ML_SCRIPTS}\n"
             f"    expected this file to live in Share\\Inverse design\\scripts\\")
sys.path.insert(0, ML_SCRIPTS)

import predict as P                                    # noqa: E402
import run_paths                                       # noqa: E402

OUT_ROOT = os.path.join(ID_ROOT, "search")
STL_DIR = os.path.join(OUT_ROOT, "candidates", "adms")
RUN_DIR = os.path.join(OUT_ROOT, "runs", "adms")
NTOP_DIR = os.path.join(OUT_ROOT, "ntop")
RESULTS = os.path.join(OUT_ROOT, "results")

NTOPCL_CANDIDATES = [
    r"C:\Program Files\nTopology\nTopology\ntopcl.exe",
    r"C:\Program Files\nTopology\ntopcl.exe",
    r"D:\Program Files\nTopology\nTopology\ntopcl.exe",
]

# Exact input names, read off the notebooks. ntopcl matches by string: a name
# that does not match is not an error, the notebook silently keeps its default.
IN_DENSITY, IN_THICK, IN_SEED = "Density", "Thickness", "Seed"
IN_INNER, IN_SIZEM, IN_STL = "Inner Size", "Size Multi", "STL Path"

# topology column -> notebook filename. Explicit, because the Run stems spell
# the variant three different ways (ADMS_DF_, ADMS_flow_, ADMS_raw_).
NB_FOR_TOPOLOGY = {"df": "ADMS_DF_only_STL.ntop",
                   "flow": "ADMS_Flow_only_STL.ntop",
                   "raw": "ADMS_Raw_only_STL.ntop"}

GATE3_TOL = 0.07          # see the module docstring: database worst is 6.59 %
VERIFY_TOL = 2.0          # per cent, on p_rel and on VF


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


def tag(x, dp):
    """Project filename convention: 22.85 -> '22p85', 15.0 -> '15p0'."""
    return f"{float(x):.{dp}f}".replace(".", "p")


def stem_for(topo, inner, D, t, seed, sizem, prefix="ID2"):
    """Named like the database rows so they read the same in a table, but with
    an ID2_ prefix so a Phase 2 candidate can NEVER be mistaken for - or
    collide with - a row that has a real FEA result behind it."""
    return (f"{prefix}_ADMS_{topo}_i{tag(inner,1)}_d{tag(D,2)}"
            f"_t{tag(t,2)}_s{int(seed)}_m{tag(sizem,1)}")


def build_input_json(dest, density, thick, seed, inner, sizem, stl_path, title):
    """Same shape as Share\\ADMS\\input_template_*.json - the format that built
    the database. 'value' singular, 'units' only on the length quantities."""
    data = {"description": "", "title": title, "inputs": [
        {"description": "", "name": IN_DENSITY, "type": "real",
         "value": float(density)},
        {"description": "", "name": IN_THICK, "type": "real", "units": "mm",
         "value": float(thick)},
        {"description": "", "name": IN_SEED, "type": "integer",
         "value": int(seed)},
        {"description": "", "name": IN_INNER, "type": "real", "units": "mm",
         "value": float(inner)},
        {"description": "", "name": IN_SIZEM, "type": "real",
         "value": float(sizem)},
        {"description": "", "name": IN_STL, "type": "file_path",
         "value": str(stl_path).replace("\\", "/")},
    ]}
    with open(dest, "w") as f:
        json.dump(data, f, indent=2)
    return data


def _norm(name):
    """Fold an nTop output name to a comparable key.

    The trimmed notebooks name the output with a GREEK SMALL LETTER RHO
    (U+03C1): the string is 'rho_rel', not 'p_rel'. The generic notebooks'
    output_template.json spells it 'p_rel' in Latin. Both must match.
    """
    return ("".join(ch for ch in str(name).lower()
                    .replace("\u03c1", "rho")            # Greek rho -> rho
                    if ch.isalnum()))


WANT_DENSITY = {"prel", "rhorel"}


def _scalar(v, depth=0):
    """Dig a float out of whatever nTop wrapped the value in.

    Seen in the wild, all from this project:
        {"value": {"isFinite": true, "units": {}, "val": 0.295}}   <- only_STL
        {"name": "p_rel", "rows": ["0.10154739"]}                  <- generic table
        {"value": 0.295}
    """
    if depth > 6:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return None
    if isinstance(v, (list, tuple)):
        for x in v:
            r = _scalar(x, depth + 1)
            if r is not None:
                return r
        return None
    if isinstance(v, dict):
        for k in ("val", "value", "values", "rows"):
            if k in v:
                r = _scalar(v[k], depth + 1)
                if r is not None:
                    return r
    return None


def read_outputs(out_json):
    """Parse ntopcl's output JSON with THE PARSER THAT BUILT THE DATABASE.

    ntop_batch.read_output_values already handles every shape these notebooks
    emit, and has for months:
        name.replace("rho", "rho").replace("sigma", "sigma")  <- Greek U+03C1 / U+03C3
        the curvature Table block, columns stored as strings
        plain named scalars  {"value": {"val": 0.295}}
        a last-resort "first number in the document"
    Reimplementing it here is what cost a failed verification run on 13/08: the
    trimmed only_STL notebooks name the output with a GREEK SMALL LETTER RHO, so
    the string is "rho_rel", and my copy matched the literal "p_rel" and found
    nothing. So it is IMPORTED, not copied, and cannot drift - the same rule
    predict.py follows for pointcloud_prep.process.

    Share\\ADMS is read-only to this script. Importing is reading.
    Returns the dict; the density lives under "rho_rel".
    """
    if ADMS_DIR not in sys.path:
        sys.path.append(ADMS_DIR)
    import ntop_batch                                   # noqa: E402
    return ntop_batch.read_output_values(out_json)


def find_number(obj_or_path, *_keys):
    """Relative density out of a finished run. Accepts the out.json PATH."""
    try:
        out = read_outputs(obj_or_path)
    except Exception as ex:
        print(f"      [!] could not use ntop_batch.read_output_values "
              f"({type(ex).__name__}: {ex}) - falling back to a local read")
        return _local_density(obj_or_path)
    v = out.get("rho_rel")
    return None if v is None else float(v)


def _local_density(path):
    """Fallback only, if Share\\ADMS is not reachable. Same folding rules."""
    try:
        obj = json.load(open(path, encoding="utf-8"))
    except Exception:
        return None
    hits = []

    def norm(n):
        n = str(n).replace("\u03c1", "rho").lower()
        return "".join(c for c in n if c.isalnum())

    def scalar(v, d=0):
        if d > 6 or isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                return None
        if isinstance(v, (list, tuple)):
            for x in v:
                r = scalar(x, d + 1)
                if r is not None:
                    return r
        if isinstance(v, dict):
            for k in ("val", "value", "values", "rows"):
                if k in v:
                    r = scalar(v[k], d + 1)
                    if r is not None:
                        return r
        return None

    def walk(o):
        if isinstance(o, dict):
            if norm(o.get("name", "")) in ("prel", "rhorel"):
                r = scalar(o)
                if r is not None:
                    hits.append(r)
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)
    walk(obj)
    return hits[0] if hits else None


def refresh_notebook(src, dst):
    """Restore the working copy from Share\\ADMS before EVERY run.

    ntop_batch.py passes `-s`, which makes ntopcl SAVE THE RESULT BACK INTO THE
    NOTEBOOK - that is why production copies the notebook per run rather than
    driving the original, so every database row came from a pristine file.
    Refreshing the copy each time reproduces that exactly, and keeps the copy's
    SHA1 stable. Costs a fraction of a second against a ~30 s generation.
    """
    shutil.copy2(src, dst)


def run_one(ntopcl, nb, stem, density, thick, seed, inner, sizem, title,
            dry=False, force=False, src_nb=None):
    """One nTop generation. Returns (stl_path_or_None, p_rel_or_None, seconds)."""
    os.makedirs(STL_DIR, exist_ok=True)
    rd = os.path.join(RUN_DIR, stem)
    os.makedirs(rd, exist_ok=True)
    stl = os.path.join(STL_DIR, stem + ".stl")
    inj, outj = os.path.join(rd, "in.json"), os.path.join(rd, "out.json")
    build_input_json(inj, density, thick, seed, inner, sizem, stl, title)
    if dry:
        return None, None, 0.0

    # resume: a 16-candidate grid is ~30 min, and an interrupted run should not
    # start from zero.
    if os.path.exists(stl) and os.path.getsize(stl) > 1000 and not force:
        p_rel = None
        if os.path.exists(outj):
            try:
                p_rel = find_number(outj)
            except Exception:
                pass
        print("      [have it already - reusing, --force to regenerate]")
        return stl, p_rel, 0.0

    if src_nb:
        refresh_notebook(src_nb, nb)
    t0 = time.time()
    # exactly the invocation ntop_batch.py uses for the production database
    args = [ntopcl, "-v2", "-s", "-j", inj, "-o", outj, nb]
    with open(os.path.join(rd, "log.txt"), "w") as lf:
        lf.write(" ".join(args) + "\n\n")
        lf.flush()
        rc = subprocess.run(args, stdout=lf, stderr=subprocess.STDOUT,
                            text=True).returncode
    secs = time.time() - t0
    if rc != 0 or not os.path.exists(stl) or os.path.getsize(stl) < 1000:
        print(f"      FAILED (exit {rc}, {secs:.0f}s) - see "
              f"{os.path.join(rd, 'log.txt')}")
        return None, None, secs
    p_rel = None
    if os.path.exists(outj):
        try:
            p_rel = find_number(outj)
        except Exception:
            pass
    return stl, p_rel, secs


def assess(stl, p_rel, bundle):
    """Mesh checks + measured VF + prediction, loading the mesh ONCE.

    Density fed to the model is nTop's p_rel, per the model card. Measured VF
    is a cross-check only. If p_rel is missing the fallback is measured VF and
    that is recorded in density_source, never left silent.
    """
    src = "ntop_p_rel"
    if p_rel is None:
        src = "measured_VF_FALLBACK"
    dens = p_rel if p_rel is not None else None
    pred, rep = P.predict_stl(stl, dens, bundle)      # density=None -> measures
    g2ok, g2bad = P.gate2(rep)
    return rep, g2ok, g2bad, rep["density_used"], src, pred


def resolve_notebook(topology):
    name = NB_FOR_TOPOLOGY.get(str(topology).lower())
    src = os.path.join(ADMS_DIR, name) if name else None
    if not (src and os.path.exists(src)):
        cands = [p for p in glob.glob(os.path.join(ADMS_DIR, "ADMS_*_only_STL.ntop"))
                 if str(topology).lower() in os.path.basename(p).lower()]
        if len(cands) != 1:
            die(f"cannot resolve the only_STL notebook for topology "
                f"{topology!r} in {ADMS_DIR}: {cands}")
        src = cands[0]
    # Copy ONCE into our own folder and drive that copy: Share\ADMS is read-only
    # to this script, and the copy is the provenance record.
    os.makedirs(NTOP_DIR, exist_ok=True)
    dst = os.path.join(NTOP_DIR, os.path.basename(src))
    stale = (not os.path.exists(dst)
             or os.path.getsize(dst) != os.path.getsize(src)
             or abs(os.path.getmtime(dst) - os.path.getmtime(src)) > 2)
    if stale:
        print(f"  copying notebook into search\\ntop\\ "
              f"({os.path.getsize(src) / 1e6:.0f} MB) ...", flush=True)
        shutil.copy2(src, dst)
    return src, dst


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true",
                    help="regenerate ONE existing database row and compare")
    ap.add_argument("--grid", action="store_true", help="the 4 x 4 fine grid")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="regenerate even if the STL is already there")
    ap.add_argument("--target", type=float, default=0.10)
    ap.add_argument("--objective", default="E_over_Es")
    ap.add_argument("--profile", default=os.environ.get("ML_PROFILE", "COMBINED"))
    ap.add_argument("--nd", type=int, default=4, help="grid points in Density")
    ap.add_argument("--nt", type=int, default=4, help="grid points in Thickness")
    ap.add_argument("--d-span", type=float, default=0.10, help="+/- fraction on D")
    ap.add_argument("--t-span", type=float, default=0.15, help="+/- fraction on t")
    ap.add_argument("--skip-verify-check", action="store_true",
                    help="run the grid without a passing reproduction check "
                         "(say why in PROJECT_LOG.md if you use this)")
    ap.add_argument("--allow-multibody", action="store_true",
                    help="do not let a body count != 1 fail GATE 2")
    ap.add_argument("--row", default=None,
                    help="Run stem to centre on (default: the Phase 1 ADMS winner)")
    a = ap.parse_args()
    if not (a.verify or a.grid):
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
    if "family" not in df.columns:
        die("the dataset has no 'family' column")
    adms = df[df.family == "ADMS"].copy()
    for c in ("density_param", "t_input", "inner", "size_multi", "seed"):
        adms[c] = pd.to_numeric(adms[c], errors="coerce")

    # ---- which row is the centre -------------------------------------------
    stem = a.row
    if stem is None:
        rk = sorted(glob.glob(os.path.join(
            RESULTS, f"phase1_ranked_{a.objective}_*.csv")))
        if not rk:
            die("no Phase 1 results found - run ID_1_COARSE.bat first")
        r1 = pd.read_csv(rk[-1])
        r1 = r1[(r1.family == "ADMS") & r1.gate4_ok].reset_index(drop=True)
        if r1.empty:
            die("Phase 1 returned no ADMS row that passed GATE 4")
        stem = str(r1.iloc[int((r1.predicted - a.target).abs().values.argmin())]["Run"])
    row = adms[adms.Run == stem]
    if row.empty:
        die(f"{stem} is not an ADMS row in the dataset")
    row = row.iloc[0]
    need = ("density_param", "t_input", "inner", "size_multi", "seed")
    missing = [c for c in need if pd.isna(row[c])]
    if missing:
        die(f"{stem} has no recorded {missing} - it cannot be regenerated. "
            f"Pass --row with a fully parameterised run.")
    topo = str(row.topology).lower()
    title = f"ADMS {topo}"

    print("=" * 74)
    print("  INVERSE DESIGN - PHASE 2  (ADMS)   generate NEW geometry")
    print("=" * 74)
    print(f"  ntopcl    {ntopcl}")
    src_nb, nb = resolve_notebook(topo)      # may print a one-off copy line
    print(f"  notebook  {os.path.basename(src_nb)}   (trimmed: geometry + STL only)")
    print(f"            sha1 {sha1(src_nb)}  driven from a refreshed copy in "
          f"search\\ntop\\ (Share\\ADMS is read-only here)")
    print(f"  model     {os.path.basename(model_path)}  {b.get('dataset_name')}  "
          f"seeds {b.get('seeds')}")
    print(f"  centre    {stem}   topology {topo}")
    print(f"            D {row.density_param}  t {row.t_input}  "
          f"inner {row.inner}  size_multi {row.size_multi}  seed {int(row.seed)}")
    print(f"            database p_rel {row.Rho_rel:.5f}   VF {row.VF:.5f}   "
          f"{a.objective} {row[a.objective]:.6f}")

    os.makedirs(RESULTS, exist_ok=True)

    # ---- VERIFY: does the trimmed notebook reproduce the database row? ------
    if a.verify:
        print("\n" + "-" * 74)
        print("  VERIFY - regenerate an EXISTING row and compare to the database.")
        print("  The trimmed notebook has never produced a row in the dataset. If")
        print("  the trimming changed the geometry, this is where it shows - two")
        print("  minutes, instead of after sixteen generations.")
        print("-" * 74)
        vstem = stem_for(topo, row.inner, row.density_param, row.t_input,
                         row.seed, row.size_multi, prefix="ID2_VERIFY")
        print(f"    D={row.density_param:<7.4g} t={row.t_input:<6.4g} -> {vstem}")
        stl, p_rel, secs = run_one(ntopcl, nb, vstem, row.density_param,
                                   row.t_input, int(row.seed), row.inner,
                                   row.size_multi, title, a.dry_run, a.force, src_nb)
        if a.dry_run:
            print("\n  [dry-run] nothing generated")
            return
        if stl is None:
            die("the verification run failed - fix that before generating a grid")
        rep, g2ok, g2bad, dens, src, pred = assess(stl, p_rel, b)
        d_vf = (rep["vf"] - row.VF) / row.VF * 100.0
        print(f"\n  generated in {secs:.0f}s")
        if p_rel is not None:
            d_pr = (p_rel - row.Rho_rel) / row.Rho_rel * 100.0
            print(f"    p_rel   notebook {p_rel:.5f}   database {row.Rho_rel:.5f}"
                  f"   diff {d_pr:+.3f} %")
        else:
            d_pr = None
            print(f"    p_rel   NOT FOUND in out.json   database {row.Rho_rel:.5f}")
            print("            -> the density channel would fall back to measured VF.")
            print(f"            Check {os.path.join(RUN_DIR, vstem, 'out.json')}.")
        print(f"    VF      measured {rep['vf']:.5f}   database {row.VF:.5f}"
              f"   diff {d_vf:+.3f} %")
        print(f"    mesh    {'GATE2 pass' if g2ok else 'GATE2 FAIL: ' + '; '.join(g2bad)}")
        print(f"            tris {rep['n_tris']}  euler {rep['euler']}  "
              f"bodies {rep['bodies']}   database: watertight "
              f"{row.get('watertight')}  euler {row.get('euler')}")
        print(f"    {a.objective:<8}  predicted {pred[a.objective]:.6f}   "
              f"database FEA {row[a.objective]:.6f}   "
              f"diff {(pred[a.objective] - row[a.objective]) / row[a.objective] * 100:+.2f} %"
              f"   (IN-SAMPLE, not an accuracy figure)")

        # Two separate questions. Reporting them as one number produced a
        # "geometry agrees to 0.011 % ... VERIFY FAIL" on 13/08, which is
        # nonsense to read.
        #   1 GEOMETRY  - did the trimmed notebook rebuild the same shape?
        #   2 DENSITY   - can we read the value the model's density channel
        #                 needs? For ADMS that is the notebook's own rho_rel;
        #                 measured VF is NOT a substitute (they differ by a
        #                 median 0.33 % and up to 6.59 % across the 163 rows).
        geom_worst = max(abs(d_vf), abs(d_pr) if d_pr is not None else 0.0)
        geom_ok = geom_worst < VERIFY_TOL and g2ok
        dens_ok = p_rel is not None
        ok = geom_ok and dens_ok
        print(f"\n  VERIFY {'PASS' if ok else 'FAIL'}")
        print(f"      geometry   {'OK' if geom_ok else 'FAILED'}   worst "
              f"{geom_worst:.3f} % (threshold {VERIFY_TOL:g} %)"
              f"   mesh {'GATE 2 pass' if g2ok else 'GATE 2 FAIL'}"
              f"   Euler {'matches' if str(rep['euler']) == str(row.get('euler')) else 'DIFFERS'}")
        print(f"      density    {'OK' if dens_ok else 'NOT READABLE'}"
              + ("" if dens_ok else
                 "   <- rho_rel was not found in out.json, so the model would\n"
                 "                          be fed measured VF instead. Fix that "
                 "before the grid."))
        with open(os.path.join(RESULTS, "phase2_adms_verify.json"), "w") as f:
            json.dump(dict(row=stem, topology=topo, notebook=os.path.basename(src_nb),
                           notebook_sha1=sha1(src_nb), pass_=bool(ok), secs=round(secs, 1),
                           p_rel_ntop=p_rel, p_rel_db=float(row.Rho_rel),
                           vf_measured=float(rep["vf"]), vf_db=float(row.VF),
                           worst_pct=float(geom_worst),
                           geometry_ok=bool(geom_ok), density_ok=bool(dens_ok),
                           euler_db=str(row.get("euler")), gate2=bool(g2ok),
                           gate2_detail=g2bad, euler=rep["euler"],
                           bodies=rep["bodies"], n_tris=rep["n_tris"],
                           predicted={k: float(v) for k, v in pred.items()},
                           stl=stl), f, indent=2)
        if not ok:
            print("  The trimmed notebook does NOT reproduce the database geometry.")
            print("  Do not generate the grid until that is understood.")
            sys.exit(1)
        print("  The trimmed notebook reproduces the database geometry. Grid is safe.")
        return

    # ---- GRID ---------------------------------------------------------------
    # The trimmed notebook produced NO row in the training database - every ADMS
    # row came from the generic_v1 notebooks. Generating sixteen candidates with
    # an unchecked generator is exactly how a wrong design reaches Phase 4, so
    # the reproduction check is a precondition, not a suggestion.
    vpath = os.path.join(RESULTS, "phase2_adms_verify.json")
    if not (a.skip_verify_check or a.dry_run):
        if not os.path.exists(vpath):
            die("the trimmed notebook has not been checked against the database.\n"
                "    Run ID_2A_FINE_ADMS_VERIFY.bat first (about 2 minutes).")
        try:
            v = json.load(open(vpath, encoding="utf-8"))
        except Exception as ex:
            die(f"cannot read {vpath}: {ex}")
        if not v.get("pass_"):
            die(f"the last reproduction check FAILED "
                f"({v.get('worst_pct', float('nan')):.3f} % on {v.get('row')}).\n"
                f"    The trimmed notebook is not building the geometry the model\n"
                f"    was trained on, and every candidate would inherit that.\n"
                f"    Fix it, or re-run ID_2A_FINE_ADMS_VERIFY.bat.")
        if v.get("notebook_sha1") != sha1(src_nb):
            die(f"the reproduction check was run against a DIFFERENT notebook\n"
                f"    (checked {v.get('notebook_sha1')}, now {sha1(src_nb)}).\n"
                f"    Re-run ID_2A_FINE_ADMS_VERIFY.bat.")
        print(f"\n  reproduction check: PASS on {v.get('row')} "
              f"({v.get('worst_pct', 0):.3f} %), same notebook.")

    # D and t TOGETHER set relative density, so a grid over both spans a band of
    # densities around the winner rather than a single line through it.
    #
    # Both axes are clipped to the range this topology was actually sampled over.
    # Beyond it we would be extrapolating the GENERATOR, not just the model, and
    # nothing downstream would flag that - GATE 4 only guards the model.
    same = adms[adms.topology.str.lower() == topo]
    dlo, dhi = float(same.density_param.min()), float(same.density_param.max())
    tlo, thi = float(same.t_input.min()), float(same.t_input.max())

    def span(c, frac, lo, hi, n):
        a_, b_ = np.clip([c * (1 - frac), c * (1 + frac)], lo, hi)
        return np.unique(np.round(np.linspace(a_, b_, n), 2))

    Ds = span(float(row.density_param), a.d_span, dlo, dhi, a.nd)
    ts = span(float(row.t_input), a.t_span, tlo, thi, a.nt)

    built = {(round(float(r.density_param), 2), round(float(r.t_input), 2))
             for _, r in same.iterrows()
             if not (pd.isna(r.density_param) or pd.isna(r.t_input))}
    plan = [(float(D), float(t)) for D in Ds for t in ts
            if (round(float(D), 2), round(float(t), 2)) not in built]
    skipped = len(Ds) * len(ts) - len(plan)

    print("\n" + "-" * 74)
    print(f"  GRID   topology {topo} sampled over D {dlo:g}..{dhi:g}  "
          f"t {tlo:g}..{thi:g}")
    print(f"         D +/-{a.d_span * 100:g} %  ->  {[float(x) for x in Ds]}")
    print(f"         t +/-{a.t_span * 100:g} %  ->  {[float(x) for x in ts]}")
    print(f"  {len(plan)} candidates" +
          (f"   ({skipped} skipped - already in the database, GATE 5)"
           if skipped else "   (none already in the database)"))
    print("-" * 74)
    if not plan:
        die("every grid point already exists - widen --d-span / --t-span")

    lo4 = b["label_min"][a.objective]
    hi4 = b["label_max"][a.objective]
    rows, t0 = [], time.time()
    for i, (D, t) in enumerate(plan, 1):
        stem_i = stem_for(topo, row.inner, D, t, row.seed, row.size_multi)
        print(f"  [{i}/{len(plan)}] D={D:<7.4g} t={t:<6.4g} -> {stem_i}", flush=True)
        stl, p_rel, secs = run_one(ntopcl, nb, stem_i, D, t, int(row.seed),
                                   row.inner, row.size_multi, title,
                                   a.dry_run, a.force, src_nb)
        if a.dry_run or stl is None:
            continue
        rep, g2ok, g2bad, dens, src, pred = assess(stl, p_rel, b)
        if a.allow_multibody:
            g2bad = [x for x in g2bad if "disconnected bodies" not in x]
            g2ok = not g2bad
        g3ok, g3err = P.gate3(rep, p_rel, tol=GATE3_TOL) if p_rel is not None \
            else (None, None)
        g4ok = bool(lo4 <= pred[a.objective] <= hi4)
        rows.append(dict(
            stem=stem_i, topology=topo, D=D, t=t, seed=int(row.seed),
            inner=float(row.inner), size_multi=float(row.size_multi),
            p_rel=p_rel, vf=rep["vf"], density_used=dens, density_source=src,
            gate2=g2ok, gate2_detail="; ".join(g2bad),
            gate3=g3ok, gate3_err=g3err, gate4=g4ok,
            watertight=rep["watertight"], winding_ok=rep["winding_ok"],
            euler=rep["euler"], bodies=rep["bodies"], n_tris=rep["n_tris"],
            secs=round(secs, 1), stl=stl,
            **{f"pred_{k}": v for k, v in pred.items()}))
        flag = "ok" if (g2ok and g4ok and g3ok is not False) else "REJECTED"
        print(f"      p_rel {dens:.5f}  {a.objective} {pred[a.objective]:.6f}  "
              f"|err| {abs(pred[a.objective] - a.target):.6f}  {flag}  ({secs:.0f}s)")

    if a.dry_run:
        print("\n  [dry-run] nothing generated")
        return
    if not rows:
        die("no candidate generated successfully - see the log.txt files under "
            "search\\runs\\adms\\")

    out = pd.DataFrame(rows)
    out["abs_err"] = (out[f"pred_{a.objective}"] - a.target).abs()
    out = out.sort_values("abs_err").reset_index(drop=True)
    f_csv = os.path.join(RESULTS, f"phase2_adms_{a.objective}_{a.target:g}.csv")
    out.to_csv(f_csv, index=False)

    good = out[out.gate2 & out.gate4 & (out.gate3 != False)]  # noqa: E712
    print("\n" + "=" * 74)
    print(f"  {len(out)} generated in {(time.time() - t0) / 60:.1f} min   "
          f"{len(good)} passed GATE 2 + GATE 3 + GATE 4")
    print("=" * 74)
    print(f"    {'candidate':<44} {'p_rel':>8} {'pred':>10} {'|err|':>9}")
    for _, r in good.head(10).iterrows():
        print(f"    {r.stem[:44]:<44} {r.density_used:8.5f} "
              f"{r[f'pred_{a.objective}']:10.6f} {r.abs_err:9.6f}")
    if len(good):
        w = good.iloc[0]
        print(f"\n  BEST NEW ADMS DESIGN   {w.stem}")
        print(f"    Density {w.D}   Thickness {w.t}   Inner Size {w.inner}   "
              f"Size Multi {w.size_multi}   Seed {w.seed}")
        print(f"    measured p_rel {w.density_used:.5f} ({w.density_source})   "
              f"measured VF {w.vf:.5f}")
        print(f"    predicted {a.objective} {w[f'pred_{a.objective}']:.6f}   "
              f"{w.abs_err / a.target * 100:.2f} % from the target {a.target:g}")
        print(f"    predicted TAI {w.get('pred_TAI', float('nan')):.4f}")
        print(f"    STL {w.stl}")
    else:
        print("\n  Nothing passed all three gates. Read the CSV before widening "
              "the grid.")

    with open(os.path.join(RESULTS,
                           f"phase2_adms_{a.objective}_{a.target:g}_manifest.json"),
              "w") as f:
        json.dump(dict(built_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                       centre_row=stem, topology=topo, target=a.target,
                       objective=a.objective, notebook=os.path.basename(src_nb),
                       notebook_sha1=sha1(src_nb), model=os.path.basename(model_path),
                       model_built=b.get("built_utc"), model_seeds=b.get("seeds"),
                       dataset=b.get("dataset_name"),
                       dataset_sha1=b.get("dataset_sha1"),
                       D_grid=[float(x) for x in Ds],
                       t_grid=[float(x) for x in ts],
                       n_planned=len(plan), n_generated=len(out),
                       n_passed=int(len(good)),
                       skipped_already_in_database=int(skipped)), f, indent=2)

    print(f"\n  -> {f_csv}")
    print("\n  Every geometry above is NEW - none has ever been simulated, and")
    print("  none of them is in the training set. Their properties come from the")
    print("  model alone. Phase 4 runs ONE real FEA on the winner to test that.")
    print("=" * 74)


if __name__ == "__main__":
    main()
