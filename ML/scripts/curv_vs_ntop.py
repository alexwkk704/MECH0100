"""
curv_vs_ntop.py  -  does the PYTHON extraction read an STL the same way nTop does?

THE ONE QUESTION THIS ANSWERS
-----------------------------
Farooq: "volume fraction and wall thickness are validated - what about curvature?"

Relative density and wall thickness were cross-checked STL-vs-nTop already
(0.9965 and 1.005). Curvature was not, because the two sides were never measuring
the same thing. This script closes that, and NOTHING ELSE.

    nTop side   Curvature_Check_v1.ntop, Import Mesh <- the exported STL
                  Max Evenly Spaced Points(mesh, spacing, relax, seed)
                  Filter Points by Volume: Offset Body(Box, -d), Region Inside
                  Evaluate Field(Gaussian Curvature / Curvature[Mean], ON the mesh)
                  Average  x Cell Size^2   (Gaussian)   or   x Cell Size   (mean)
                -> curvature_ntop.csv : avg_Mn_K, avg_Mn_H

    THIS SCRIPT the SAME STL, the SAME domain, the SAME weighting, in Python
                -> curv_vs_ntop.csv   : py_K, py_H  and the ratio to nTop

If the ratios sit near 1.0 with a TIGHT SPREAD, the extraction is validated.
That is the deliverable. This script does not touch the design body, the FE mesh,
the export tolerance, features_ADMS.csv or anything else. It is READ-ONLY.

WHY THE DOMAIN AND WEIGHTING HAVE TO MATCH
------------------------------------------
1. DOMAIN.  nTop keeps only points at least `offset` mm inside the bounding cube,
   via Offset Body(Box, -offset). That removes the flat cut-caps and the sharp
   creases the Boolean intersect leaves behind - which carry a large share of the
   curvature and are not part of the lattice. feature_extraction.py instead drops a
   2% bounding-box SKIN, which removes a geometry-dependent 11-17% of vertices and
   is not the same set. Comparing across those two domains is meaningless, so this
   script reproduces nTop's rule exactly: |x| <= Inner/2 - offset on every axis.

2. WEIGHTING.  `Max Evenly Spaced Points` distributes points uniformly BY AREA, so
   nTop's plain `Average` over them is already an area-weighted average. The exact
   limit of that as the point count grows is  sum(v*A) / sum(A)  over the domain,
   which is what the *_area columns report. There is deliberately no point-sampled
   column: see the note above self-test() for why the one that used to be here was
   removed.

SIGN
----
The mean-curvature sign convention has been a live problem (nTop reports a convex
sphere as NEGATIVE, -1/R). This script reports BOTH signs and does not assume one.
Whichever matches nTop across all geometries is the correct convention, and that is
then a measured result rather than a guess. Nothing is "corrected" here.

SELF-TEST
---------
Run with --selftest and it builds a sphere and a torus and checks the estimators
against the analytic answers before touching any lattice:
    sphere R:   K = 1/R^2 everywhere,  |H| = 1/R everywhere
    torus:      area-average of K is EXACTLY 0  (Gauss-Bonnet, chi = 0)
If the self-test fails, no lattice number from this script can be trusted.

USAGE  (from ML\\scripts\\)
--------------------------
    python curv_vs_ntop.py --selftest          # analytic check, no data needed
    python curv_vs_ntop.py                     # the 11 pilot geometries
    python curv_vs_ntop.py --only <stem>
    python curv_vs_ntop.py --all
Options: --offset (default 0.1 mm, matches Offset Body), --cell-size (default 3.0,
must equal cell_size_mm in ML_settings.xlsx), --spacing (default 0.3 mm).
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent            # .../ML/scripts
ML_DIR = SCRIPT_DIR.parent
SHARE = ML_DIR.parent
STL_DIR = SHARE / "ADMS" / "ADMS_STL"
NTOP_CSV = ML_DIR / "Curvature_Check" / "curvature_ntop.csv"
OUT_CSV = SCRIPT_DIR / "curv_vs_ntop.csv"

PILOT = [
    "ADMS_DF_i12p0_d20p83_t0p4_s2_m1p6",
    "ADMS_DF_i15p0_d20p83_t0p6_s2_m1p6",
    "ADMS_DF_i15p0_d26p89_t0p17_s2_m1p6",
    "ADMS_DF_i15p0_d30p93_t0p8_s2_m1p6",
    "ADMS_DF_i15p0_d36p0_t0p4_s2_m1p6",
    "ADMS_flow_i15p0_d14p77_t0p8_s2_m1p6",
    "ADMS_flow_i15p0_d26p89_t0p16_s2_m1p6",
    "ADMS_flow_i15p0_d50p0_t0p4_s2_m1p6",
    "ADMS_raw_i15p0_d30p93_t0p14_s2_m1p6",
    "ADMS_raw_i15p0_d42p0_t0p18_s2_m1p6",
    "ADMS_raw_i15p0_d42p0_t0p5_s2_m1p6",
]

COLS = ["Run", "inner_mm", "inner_src", "n_tris", "n_vert_in_domain", "domain_vert_frac",
        "py_K_area", "py_H_area", "py_Hcot_area",
        "py_absH_area", "H_cancel", "py_H_iqr",
        "nTop_K", "nTop_H", "K_ratio", "H_ratio", "Hcot_ratio", "H_ratio_flipped",
        "seconds"]


# ------------------------------------------------------------------
# curvature estimators - identical maths to feature_extraction.py
# ------------------------------------------------------------------
def per_vertex_curvature(mesh):
    """Gaussian and mean curvature PER VERTEX, plus the vertex dual area.

    Gaussian: angle defect / dual area. This is the standard discrete estimator and
    it is exact in the integrated sense (sum of defects = 2*pi*chi at any resolution).

    Mean: the per-EDGE dihedral estimator  H_e = 0.5 * theta * len / A_e  is scattered
    onto vertices weighted by area, so that both quantities live on the same support
    and can share one domain mask. Sign is left POSITIVE for convex here; the caller
    decides the convention (see the module docstring - it is not assumed).
    """
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.faces)
    Af = np.asarray(mesh.area_faces)

    A_dual = np.zeros(len(V))
    np.add.at(A_dual, F.ravel(), np.repeat(Af / 3.0, 3))

    K = np.zeros(len(V))
    ok = A_dual > 1e-12
    K[ok] = np.asarray(mesh.vertex_defects)[ok] / A_dual[ok]

    theta = np.asarray(mesh.face_adjacency_angles)
    edges = np.asarray(mesh.face_adjacency_edges)
    pairs = np.asarray(mesh.face_adjacency)
    elen = np.linalg.norm(V[edges[:, 0]] - V[edges[:, 1]], axis=1)
    A_e = (Af[pairs[:, 0]] + Af[pairs[:, 1]]) / 3.0
    sign = np.where(np.asarray(mesh.face_adjacency_convex), 1.0, -1.0)
    good = A_e > 1e-12
    Hn = np.zeros(len(edges))
    Hn[good] = sign[good] * 0.5 * theta[good] * elen[good] / A_e[good]

    # scatter each edge value onto its two endpoints, area-weighted
    num = np.zeros(len(V))
    den = np.zeros(len(V))
    for c in (0, 1):
        np.add.at(num, edges[good, c], Hn[good] * A_e[good])
        np.add.at(den, edges[good, c], A_e[good])
    H = np.zeros(len(V))
    okh = den > 1e-12
    H[okh] = num[okh] / den[okh]

    return K, H, A_dual


def mean_curvature_cotan(mesh, A_dual):
    """The OTHER standard discrete mean curvature: the cotangent Laplacian.

        dx_i = 1/(2 A_i) * sum_j (cot a_ij + cot b_ij) (x_j - x_i)
        H_i  = -0.5 * <dx_i, n_i>

    WHY THIS EXISTS
    ---------------
    Gaussian curvature has a unique discrete form (angle defect = integrated K,
    exact at any resolution by discrete Gauss-Bonnet), so any correct code gives the
    same number - which is why Python and nTop agree on it to within 10% on all 11
    geometries. Mean curvature has NO such theorem. The dihedral-edge estimator and
    the cotangent Laplacian both converge to the true H on a SMOOTH surface, but
    they weight facet angles differently and diverge on a rough one. Measured: on
    the lattice STLs the dihedral estimator reads 0.77x nTop, while both hit 1/R on
    a sphere to better than 0.05%.

    So this is a HYPOTHESIS TEST, not a fix: if the cotangent value lands on nTop's,
    nTop uses this estimator and mean curvature is validated too. If it does not,
    nTop uses a third method and the 0.77 stands as an estimator difference.

    SIGN: calibrated on a sphere so that CONVEX IS POSITIVE, matching the dihedral
    estimator in per_vertex_curvature() and nTop's mesh convention (nTop returned
    +0.19993 for a sphere of R=5). Note nTop uses the OPPOSITE sign for implicit
    bodies - that inconsistency is nTop's, not ours.
    """
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.faces)
    N = np.asarray(mesh.vertex_normals)

    dx = np.zeros((len(V), 3))
    for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        # cotangent of the angle at vertex `a`, applied to the opposite edge (b, c)
        va, vb, vc = V[F[:, a]], V[F[:, b]], V[F[:, c]]
        u, w = vb - va, vc - va
        cr = np.linalg.norm(np.cross(u, w), axis=1)
        cot = np.einsum("ij,ij->i", u, w) / np.where(cr > 1e-20, cr, 1e-20)
        np.add.at(dx, F[:, b], (cot[:, None]) * (vc - vb))
        np.add.at(dx, F[:, c], (cot[:, None]) * (vb - vc))
    ok = A_dual > 1e-12
    dx[ok] /= (2.0 * A_dual[ok])[:, None]
    H = np.zeros(len(V))
    H[ok] = -0.5 * np.einsum("ij,ij->i", dx[ok], N[ok])
    return H


def domain_mask(V, inner_mm, offset_mm):
    """nTop's Filter Points by Volume: Offset Body(Box, -offset), Region Inside.

    The Box is the analytic cube of side `inner_mm` centred on the origin - NOT the
    mesh bounding box. That distinction matters: the production STL overshoots the
    design cube by up to 0.0086 mm because the FE mesher smooths nodes, so a bbox
    test and an analytic-cube test are not the same set.
    """
    if inner_mm is None:
        return np.ones(len(V), dtype=bool)
    half = inner_mm / 2.0 - offset_mm
    return np.all(np.abs(V) <= half, axis=1)


def area_mean(vals, weights, mask):
    w = weights[mask]
    if w.sum() <= 0:
        return float("nan")
    return float(np.sum(vals[mask] * w) / w.sum())


# NOTE - REMOVED 2026-07-30: there used to be a sampled_mean() here that repeated
# the average by drawing random surface points and taking each one's NEAREST VERTEX
# value, as a cross-check on the area-weighted number. It agreed to six decimals on
# the smooth cut-sphere self-test, so it looked sound - but on a real lattice it was
# badly wrong (ratio to the area-weighted value ranged -2.46 to 0.91, sign flips
# included). Nearest-vertex assignment is a poor interpolator on a mesh whose triangle
# areas span orders of magnitude, and the domain mask was applied to the samples'
# nearest vertices rather than to the samples themselves. It was a broken diagnostic,
# never a result. Deleted rather than left in with a caveat, because a plausible-looking
# extra column in a csv is exactly the sort of thing that ends up quoted by mistake.
# The reported values are the *_area columns: exact area-weighted averages over the
# domain, which are the mathematical limit of nTop's area-uniform point sampling.


# ------------------------------------------------------------------
# self-test on shapes with known answers
# ------------------------------------------------------------------
def selftest():
    import trimesh
    print("\n" + "=" * 70)
    print("  SELF-TEST - estimators against analytic answers")
    print("=" * 70)
    ok = True

    R = 5.0
    s = trimesh.creation.icosphere(subdivisions=5, radius=R)
    K, H, A = per_vertex_curvature(s)
    Hc = mean_curvature_cotan(s, A)
    m = np.ones(len(K), dtype=bool)
    kK, kH = area_mean(K, A, m), area_mean(H, A, m)
    kHc = area_mean(Hc, A, m)
    eK, eH = 1.0 / R**2, 1.0 / R
    print(f"  sphere R={R}")
    print(f"    Gaussian  {kK:+.6f}   expected {eK:+.6f}   err {100*(kK/eK-1):+.3f} %")
    print(f"    mean      {kH:+.6f}   expected {eH:+.6f} (magnitude)   "
          f"err {100*(abs(kH)/eH-1):+.3f} %")
    if abs(kK / eK - 1) > 0.02:
        ok = False; print("    [FAIL] Gaussian off by more than 2%")
    if abs(abs(kH) / eH - 1) > 0.02:
        ok = False; print("    [FAIL] mean off by more than 2%")
    print(f"    mean, cotan  {kHc:+.6f}   expected {eH:+.6f}   "
          f"err {100*(abs(kHc)/eH-1):+.3f} %")
    if abs(abs(kHc) / eH - 1) > 0.02:
        ok = False; print("    [FAIL] cotan mean off by more than 2%")
    print(f"    convex sign, both estimators: "
          f"{'POSITIVE' if kH > 0 else 'NEGATIVE'} / "
          f"{'POSITIVE' if kHc > 0 else 'NEGATIVE'}"
          f"   (nTop on a MESH returned +0.19993 for R=5, i.e. POSITIVE)")

    t = trimesh.creation.torus(major_radius=10.0, minor_radius=3.0,
                               major_sections=200, minor_sections=100)
    K, H, A = per_vertex_curvature(t)
    m = np.ones(len(K), dtype=bool)
    kK = area_mean(K, A, m)
    # Gauss-Bonnet: chi(torus) = 0 so the integral of K dA is exactly 0
    scale = 1.0 / (3.0 ** 2)          # compare against a natural curvature scale
    print(f"  torus R=10 r=3")
    print(f"    area-avg Gaussian {kK:+.6f}   expected {0.0:+.6f} exactly "
          f"(Gauss-Bonnet, chi=0)")
    if abs(kK) > 0.02 * scale:
        ok = False; print("    [FAIL] should be ~0")

    # ---- THE test that matters: a smooth surface CUT BY A BOX ---------------
    # This is the lattice situation exactly - curved surface, flat cut-caps, sharp
    # creases where they meet - and it is the only case that exercises the domain
    # filter. If the filter works, the caps and creases are excluded and what is
    # left must return the ANALYTIC sphere values.
    try:
        cut = trimesh.boolean.intersection(
            [trimesh.creation.icosphere(subdivisions=5, radius=R),
             trimesh.creation.box(extents=[8.0, 8.0, 8.0])])
    except Exception as ex:
        print(f"  cut sphere: SKIPPED (no boolean backend: {type(ex).__name__})")
        print("    pip install manifold3d  to enable the domain-filter test")
        cut = None
    if cut is not None:
        cut.merge_vertices()
        K, H, A = per_vertex_curvature(cut)
        Hc = mean_curvature_cotan(cut, A)
        msk = domain_mask(np.asarray(cut.vertices), 8.0, 0.1)
        cK, cH = area_mean(K, A, msk), area_mean(H, A, msk)
        cHc = area_mean(Hc, A, msk)
        print(f"  sphere R={R} CUT BY an 8 mm box   "
              f"(domain keeps {100*msk.mean():.1f}% of vertices)")
        print(f"    Gaussian  {cK:+.6f}   expected {eK:+.6f}   "
              f"err {100*(cK/eK-1):+.3f} %")
        print(f"    mean      {cH:+.6f}   expected {eH:+.6f}   "
              f"err {100*(abs(cH)/eH-1):+.3f} %")
        print(f"    mean cotan{cHc:+.6f}   expected {eH:+.6f}   "
              f"err {100*(abs(cHc)/eH-1):+.3f} %")
        if abs(cK / eK - 1) > 0.02 or abs(abs(cH) / eH - 1) > 0.02:
            ok = False
            print("    [FAIL] the domain filter is NOT removing the caps/creases "
                  "cleanly - every lattice number would be contaminated")

    print("=" * 70)
    print("  SELF-TEST " + ("PASSED" if ok else "FAILED"))
    print("=" * 70 + "\n")
    return ok


# ------------------------------------------------------------------
def load_ntop(path=None):
    """nTop's reading of the SAME STL: avg_Mn_K / avg_Mn_H from curvature_ntop.csv."""
    src = Path(path) if path else NTOP_CSV
    if not src.is_absolute():
        src = NTOP_CSV.parent / src
    if not src.exists():
        print(f"[!] {src} not found - nTop columns will be blank")
        return {}
    out = {}
    with src.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            run = str(r.get("Run", "")).strip()
            if not run:
                continue
            rec = {}
            for col, dst in (("avg_Mn_K", "K"), ("avg_Mn_H", "H")):
                try:
                    rec[dst] = float(r[col])
                except (TypeError, ValueError, KeyError):
                    pass
            if rec:
                out[run] = rec
    print(f"[ok] nTop STL reference loaded for {len(out)} runs  ({src.name})")
    return out


FEATURES_CSV = ML_DIR / "data" / "features_ADMS.csv"


def load_bbox_inner():
    """{stem: box size} from features_ADMS.csv 'bbox_mm'.

    MUST match curv_backfill.py's resolve_inner() exactly, or the two sides filter
    on different boxes and the ratio is meaningless. Legacy stems (ADMS_DF_5cube_*)
    carry no _iXXpX token, and nTop falls back to the measured bounding box rounded
    to the nearest 0.5 mm (faceting makes a 15 mm cube measure ~15.015). Same rule
    here.
    """
    if not FEATURES_CSV.exists():
        return {}
    out = {}
    try:
        with FEATURES_CSV.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                stem = str(r.get("file", "")).replace(".stl", "").replace(".STL", "")
                try:
                    b = float(r.get("bbox_mm"))
                except (TypeError, ValueError):
                    continue
                if stem and b > 0:
                    out[stem] = round(b * 2.0) / 2.0
    except OSError:
        return {}
    return out


def inner_from_stem(stem, bbox_map=None):
    import re
    m = re.search(r"_i(\d+)p(\d+)", stem)
    if m:
        return float(f"{m.group(1)}.{m.group(2)}"), "filename"
    if bbox_map and stem in bbox_map:
        return bbox_map[stem], "bbox"
    return None, "none"


def one(path, stem, args, bbox_map=None):
    import trimesh
    t0 = time.time()
    mesh = trimesh.load(str(path), process=False)
    if hasattr(mesh, "geometry"):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    mesh.merge_vertices()          # nTop welds on import too ("duplicate ... merged")

    V = np.asarray(mesh.vertices)
    inner, inner_src = inner_from_stem(stem, bbox_map)
    if args.no_filter:
        inner, inner_src = None, "no-filter"
    mask = domain_mask(V, inner, args.offset)

    K, H, A = per_vertex_curvature(mesh)
    Hc = mean_curvature_cotan(mesh, A)
    L = args.cell_size
    row = {
        "Run": stem,
        "inner_mm": inner if inner is not None else "",
        "inner_src": inner_src,
        "n_tris": len(mesh.faces),
        "n_vert_in_domain": int(mask.sum()),
        "domain_vert_frac": round(float(mask.mean()), 4),
        "py_K_area": round(area_mean(K, A, mask) * L * L, 6),
        "py_H_area": round(area_mean(H, A, mask) * L, 6),
        "py_Hcot_area": round(area_mean(Hc, A, mask) * L, 6),
    }
    # CANCELLATION DIAGNOSTIC.  H_cancel = |mean(H)| / mean(|H|)  over the domain.
    #   ~1.0  -> H has one sign, the average is well conditioned (a torus gives this)
    #   ~0.1  -> the average is a small RESIDUAL of large opposing contributions, so a
    #            few-percent difference between two estimators becomes a tens-of-percent
    #            difference in the mean. That is the suspected cause of the 0.77 factor
    #            between Python and nTop on the lattices, given that BOTH tools are
    #            exact on a smooth torus AND on a 400-triangle one (nTop +0.36%,
    #            Python +0.02% vs the analytic 1/(2r)), which rules out faceting.
    # py_H_iqr is the SPREAD of H, which does not cancel and is the statistic
    # feature_extraction.py already prefers (validated 20/07: IQR drifts -0.5% under
    # mesh refinement where std drifts +27%).
    absmean = area_mean(np.abs(H), A, mask)
    row["py_absH_area"] = round(absmean * L, 6)
    row["H_cancel"] = (round(abs(row["py_H_area"] / L) / absmean, 4)
                       if absmean > 1e-12 else "")
    Hm = H[mask]
    row["py_H_iqr"] = round(float(np.subtract(*np.percentile(Hm, [75, 25]))) * L, 6) \
        if Hm.size else ""
    row["seconds"] = round(time.time() - t0, 1)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--stl-dir", default=str(STL_DIR))
    ap.add_argument("--only", action="append", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--match-ntop", action="store_true",
                    help="process ONLY the geometries that already have a row in the "
                         "nTop csv. Use this to validate a partial nTop run before "
                         "committing to all 131 - there is no point extracting an STL "
                         "that has nothing to be compared against.")
    ap.add_argument("--offset", type=float, default=0.1,
                    help="matches Offset Body Distance in nTop (default 0.1 mm)")
    ap.add_argument("--cell-size", type=float, default=3.0,
                    help="must equal cell_size_mm in ML_settings.xlsx AND the "
                         "Cell Size input in nTop (default 3.0)")
    ap.add_argument("--spacing", type=float, default=0.3,
                    help="matches Point spacing in nTop (default 0.3 mm)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-filter", action="store_true",
                    help="measure the WHOLE STL surface, no box exclusion. Use this "
                         "when the nTop notebook has no Filter Points by Volume block. "
                         "The Boolean cut-caps and creases are then included on BOTH "
                         "sides and treated identically, which is all that a "
                         "Python-vs-nTop comparison requires.")
    ap.add_argument("--ntop-csv", default="curvature_ntop_stl.csv",
                    help="nTop's reading of the SAME STL. Default is the STL-only run "
                         "(points on the exported mesh, Zhezhe's setup). Pass "
                         "curvature_ntop.csv to compare against the design-body run "
                         "instead - but note those points sit on the design surface, "
                         "not on the STL, so it is not a like-for-like comparison.")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)

    if not selftest():
        sys.exit("[FATAL] self-test failed - not running on lattice data")

    stl_dir = Path(args.stl_dir)
    if not stl_dir.is_dir():
        sys.exit(f"[FATAL] STL folder not found: {stl_dir}")
    ntop = load_ntop(args.ntop_csv)
    bbox_map = load_bbox_inner()
    print(f"[ok] bbox fallback available for {len(bbox_map)} stems "
          f"(used when the filename has no _iXXpX token)")
    if args.only:
        stems = [s.replace(".stl", "") for s in args.only]
    elif args.match_ntop:
        have = {p.stem for p in stl_dir.glob("*.stl")}
        stems = sorted(k for k in ntop if k in have)
        missing = sorted(k for k in ntop if k not in have)
        if missing:
            print(f"[!] {len(missing)} nTop rows have no STL on disk: {missing[:3]}...")
        if not stems:
            sys.exit("[FATAL] --match-ntop: no overlap between the nTop csv and the "
                     "STL folder. Has the nTop run written any rows yet?")
        print(f"[ok] --match-ntop: {len(stems)} geometries to check")
    elif args.all:
        stems = sorted(p.stem for p in stl_dir.glob("*.stl"))
    else:
        stems = PILOT
    rows = []
    print(f"\ndomain: " + ("WHOLE SURFACE (no box filter)" if args.no_filter
          else f"box minus {args.offset} mm")
          + f"   cell {args.cell_size} mm   spacing {args.spacing} mm\n")
    hdr = (f"{'run':<34}{'py_K':>10}{'nTop_K':>10}{'K rat':>7}  "
           f"{'py_H':>9}{'py_Hcot':>9}{'nTop_H':>9}{'H rat':>7}{'cot rat':>8}")
    print(hdr); print("-" * len(hdr))

    for stem in stems:
        p = stl_dir / f"{stem}.stl"
        if not p.is_file():
            print(f"{stem:<38}  [skip] no STL"); continue
        try:
            r = one(p, stem, args, bbox_map)
        except Exception as ex:
            print(f"{stem:<38}  [skip] {type(ex).__name__}: {ex}"); continue
        ref = ntop.get(stem, {})
        r["nTop_K"] = ref.get("K", "")
        r["nTop_H"] = ref.get("H", "")
        if ref.get("K"):
            r["K_ratio"] = round(r["py_K_area"] / ref["K"], 4)
        if ref.get("H"):
            r["H_ratio"] = round(r["py_H_area"] / ref["H"], 4)
            r["Hcot_ratio"] = round(r["py_Hcot_area"] / ref["H"], 4)
            r["H_ratio_flipped"] = round(-r["py_H_area"] / ref["H"], 4)
        rows.append(r)
        f = lambda v, w=10: (f"{v:+{w}.4f}" if isinstance(v, float) else f"{'':>{w}}")
        print(f"{stem:<34}{f(r['py_K_area'],10)}{f(r.get('nTop_K'),10)}"
              f"{f(r.get('K_ratio'),7)}  "
              f"{f(r['py_H_area'],9)}{f(r['py_Hcot_area'],9)}{f(r.get('nTop_H'),9)}"
              f"{f(r.get('H_ratio'),7)}{f(r.get('Hcot_ratio'),8)}")

    if not rows:
        sys.exit("\n[FATAL] nothing computed")

    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in COLS})

    def spread(col, lab):
        v = sorted(r[col] for r in rows if isinstance(r.get(col), float))
        if not v:
            print(f"  {lab}  no nTop reference to compare against"); return
        med = v[len(v)//2] if len(v) % 2 else 0.5*(v[len(v)//2-1]+v[len(v)//2])
        within = sum(1 for x in v if 0.9 <= x <= 1.1)
        print(f"  {lab}  n={len(v):2d}   median {med:+.3f}   "
              f"range [{v[0]:+.3f} .. {v[-1]:+.3f}]   within +/-10%: {within}/{len(v)}")

    print(f"\n{'=' * 70}\n  PYTHON / nTop  on the SAME exported STL\n{'=' * 70}")
    can = [r["H_cancel"] for r in rows if isinstance(r.get("H_cancel"), float)]
    if can:
        can.sort()
        print("  |mean H| / mean|H|    n=%2d   median %.3f   range [%.3f .. %.3f]"
              % (len(can), can[len(can)//2], can[0], can[-1]))
        print("     ~1.0 = one-signed, well conditioned.  ~0.1 = the mean is a small")
        print("     residual of cancelling values, so estimator choice is amplified.\n")
    spread("K_ratio", "Gaussian            ")
    spread("H_ratio", "mean, dihedral      ")
    spread("Hcot_ratio", "mean, cotan Laplace ")
    spread("H_ratio_flipped", "mean, sign flip     ")
    print("""
  READ IT LIKE THIS
    ratios near 1.0 with a TIGHT range -> the extraction is validated. That is the
      answer to Farooq, and it says nothing about the export tolerance either way.
    a tight range at some OTHER constant -> a fixed convention difference, find it.
    a WIDE range -> not validated. Say so; do not quote the median.
    'mean, sign flip' tighter than 'mean' -> the sign convention in
      feature_extraction.py is the wrong way round for the lattice. Do not change
      the script on one geometry - only on a consistent result across all of them.""")
    print(f"\n  -> {OUT_CSV}\n")


if __name__ == "__main__":
    main()
