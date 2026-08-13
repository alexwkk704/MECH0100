"""
predict.py - run the frozen forward model on any STL.

Loads a bundle written by export_forward_model.py (job 10) and turns an STL into
predicted homogenised properties. This is the ONLY sanctioned way to call the
forward model from outside the training pipeline; inverse design imports the
functions below rather than re-deriving any of it.

THE SIX STEPS (they are the whole point of this file)
  1. load the mesh in trimesh and measure its volume fraction
  2. sample n_points uniformly over the surface
  3. centre at the origin and scale into a 1x1x1 box
  4. attach relative density as a 4th per-point channel
  5. forward pass through the saved network, in eval() mode
  6. un-standardise with the training mean/std, then undo log10

Steps 2 and 3 are NOT reimplemented here. They are `pointcloud_prep.process`,
imported directly, because a hand-copied centre-and-scale that drifts by one
line silently produces confident wrong numbers - and nothing downstream would
notice. If pointcloud_prep changes, this changes with it, automatically.

DENSITY - the one input you must get right
  The channel was trained on the dataset's Rho_rel column.
    TPMS  Rho_rel == VF measured from the STL, exactly. --density measure is right.
    ADMS  Rho_rel is nTop's p_rel; STL VF differs by a median 0.33%, max 6.59%.
          Pass nTop's p_rel with --density <value>.

Usage
  python predict.py --stl part.stl --density 0.10
  python predict.py --stl part.stl --density measure --cell 10.0
  python predict.py --batch candidates.csv --out predictions.csv
      candidates.csv needs a column of STL paths ('stl'/'path'/'file') and
      optionally 'density' (blank or 'measure' -> measured from the mesh).
  python predict.py --self-test          # loads the bundle and reports its spec
"""

from __future__ import annotations
import os, sys, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_paths                              # noqa: E402
import pointcloud_prep                        # noqa: E402  (steps 2 + 3 live here)


# ---------------------------------------------------------------------------
def default_model_path(profile=None):
    prof = profile or run_paths.profile() or "COMBINED"
    return os.path.join(run_paths.BASE_DIR, "Final Model", prof, f"model_{prof}.pt")


def load_bundle(path=None, profile=None):
    import torch
    p = path or default_model_path(profile)
    if not os.path.exists(p):
        sys.exit(f"[E] no exported model at {p}\n"
                 f"    run 10_EXPORT_FORWARD_MODEL.bat first")
    try:
        b = torch.load(p, map_location="cpu", weights_only=False)
    except TypeError:                          # older torch has no weights_only
        b = torch.load(p, map_location="cpu")
    return b


def build_nets(bundle):
    """Rebuild the architecture and load every saved state dict."""
    import torch.nn as nn
    in_dim, deep, n_out = bundle["in_dim"], bundle["deep"], len(bundle["targets"])

    class PointNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Conv1d(in_dim, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
                nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
                nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU())
            if deep:
                self.head = nn.Sequential(
                    nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(256, n_out))
            else:
                self.head = nn.Sequential(
                    nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(256, 64), nn.ReLU(),
                    nn.Linear(64, n_out))

        def forward(self, x):
            x = x.transpose(1, 2)
            x = self.enc(x).max(dim=2).values
            return self.head(x)

    nets = []
    for sd in bundle.get("state_dicts") or [bundle["state_dict"]]:
        net = PointNet()
        net.load_state_dict(sd)
        net.eval()                             # dropout off, BatchNorm frozen
        nets.append(net)
    return nets


# ---------------------------------------------------------------------------
def mesh_report(stl_path, cell_mm=None):
    """STEP 1. Geometry health (GATE 2) and volume fraction (GATE 3).

    cell_mm : nominal unit-cell edge. Prefer it over the bounding box - measured
    against Zhezhe's IWP set the nominal 10 mm cell reproduced nTop's own
    vf_actual to 0.12%, where the bounding box was 4x worse (up to 0.46%),
    because the surface bulges past the cell wall.
    """
    import trimesh
    m = trimesh.load(stl_path, force="mesh")
    ext = np.asarray(m.bounding_box.extents, float)
    box = float(np.prod(ext)) if cell_mm is None else float(cell_mm) ** 3
    vol = float(m.volume)
    return dict(
        path=stl_path,
        watertight=bool(m.is_watertight),
        winding_ok=bool(m.is_winding_consistent),
        euler=int(m.euler_number),
        bodies=int(m.body_count),
        n_tris=int(len(m.faces)),
        bbox_mm=[round(float(v), 5) for v in ext],
        volume_mm3=vol,
        vf=vol / box if box > 0 else float("nan"),
        vf_basis=("nominal cell %.5g mm" % cell_mm) if cell_mm else "bounding box",
    )


def gate2(rep):
    """GATE 2 - mesh validity. Returns (ok, list_of_failures)."""
    bad = []
    if not rep["watertight"]:
        bad.append("not watertight")
    if not rep["winding_ok"]:
        bad.append("inconsistent winding")
    if rep["bodies"] != 1:
        bad.append(f"{rep['bodies']} disconnected bodies")
    if rep["euler"] % 2 != 0:
        bad.append(f"odd Euler number {rep['euler']}")
    return (not bad), bad


def gate3(rep, requested_vf, tol=0.05):
    """GATE 3 - generation sanity: measured VF within tol of what was asked."""
    if requested_vf in (None, "") or not np.isfinite(rep["vf"]):
        return None, None
    err = (rep["vf"] - float(requested_vf)) / float(requested_vf)
    return abs(err) <= tol, err


def gate4(bundle, target_name, target_value):
    """GATE 4 - range guard: refuse asks outside the training label range."""
    lo = bundle.get("label_min", {}).get(target_name)
    hi = bundle.get("label_max", {}).get(target_name)
    if lo is None or hi is None:
        return None, (None, None)
    return (float(lo) <= float(target_value) <= float(hi)), (lo, hi)


# ---------------------------------------------------------------------------
def cloud_from_stl(stl_path, bundle):
    """STEPS 2 + 3, delegated to pointcloud_prep so they cannot drift."""
    pts = pointcloud_prep.process(stl_path, int(bundle["n_points"]), int(bundle["pc_seed"]))
    return np.asarray(pts, np.float32)


def predict_clouds(clouds, densities, bundle):
    """STEPS 4, 5, 6 for a batch of already-prepared clouds.

    clouds    (B, N, 3) float32, already centred and unit-boxed
    densities (B,) relative density, or None when the model has no density channel
    returns   (B, T) in PHYSICAL units, column order = bundle['targets']
    """
    import torch
    X = np.asarray(clouds, np.float32)
    if bundle["use_density"]:
        if densities is None:
            raise ValueError("this model has a density channel - densities required")
        d = np.asarray(densities, np.float32).reshape(-1, 1, 1)
        if len(d) != len(X):
            raise ValueError(f"{len(d)} densities for {len(X)} clouds")
        X = np.concatenate([X, np.repeat(d, X.shape[1], axis=1)], axis=2).astype(np.float32)
    if X.shape[2] != bundle["in_dim"]:
        raise ValueError(f"built {X.shape[2]} channels, model expects {bundle['in_dim']}")

    nets = bundle.get("_nets")
    if nets is None:
        nets = build_nets(bundle)
        bundle["_nets"] = nets                 # cache across calls in a search loop

    outs = []
    with torch.no_grad():
        for net in nets:
            chunks = []
            for b in range(0, len(X), 32):
                chunks.append(net(torch.from_numpy(X[b:b + 32])).numpy())
            outs.append(np.concatenate(chunks, axis=0))
    P = np.mean(np.stack(outs), axis=0)

    P = P * bundle["ystd"] + bundle["ymean"]   # STEP 6
    if bundle["use_log"]:
        P = np.power(10.0, P)
    return P


def predict_stl(stl_path, density=None, bundle=None, cell_mm=None):
    """Full six-step chain for one STL. Returns (dict of predictions, mesh report)."""
    b = bundle if bundle is not None else load_bundle()
    rep = mesh_report(stl_path, cell_mm)
    dens = rep["vf"] if (density is None or str(density).lower() == "measure") else float(density)
    cloud = cloud_from_stl(stl_path, b)
    P = predict_clouds(cloud[None, ...], [dens] if b["use_density"] else None, b)[0]
    rep["density_used"] = dens
    return {t: float(P[j]) for j, t in enumerate(b["targets"])}, rep


# ---------------------------------------------------------------------------
def _fmt(rep):
    ok, bad = gate2(rep)
    return (f"    mesh: {'GATE2 pass' if ok else 'GATE2 FAIL - ' + '; '.join(bad)}"
            f"  | tris {rep['n_tris']}  euler {rep['euler']}  bodies {rep['bodies']}\n"
            f"    VF  : {rep['vf']:.6f}  ({rep['vf_basis']})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model")
    ap.add_argument("--profile")
    ap.add_argument("--stl")
    ap.add_argument("--density", default=None,
                    help="relative density, or 'measure' to take it from the mesh")
    ap.add_argument("--cell", type=float, default=None,
                    help="nominal unit-cell edge in mm (preferred over the bounding box)")
    ap.add_argument("--batch", help="CSV of candidates")
    ap.add_argument("--out", help="write batch results here")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    b = load_bundle(a.model, a.profile)
    print(f"[i] model   : {a.model or default_model_path(a.profile)}")
    print(f"[i] profile : {b.get('profile')}   built {b.get('built_utc')}")
    print(f"[i] trained : {b.get('n_train')} rows of {b.get('dataset_name')} "
          f"(sha1 {b.get('dataset_sha1')})")
    print(f"[i] seeds   : {b.get('seeds')}   models in bundle: {b.get('n_models', 1)}")
    print(f"[i] targets : {', '.join(b['targets'])}")
    print(f"[i] prep    : {b['n_points']} pts, seed {b['pc_seed']}, "
          f"density channel {b['use_density']} "
          f"(trained on {b.get('density_column')})")

    if a.self_test:
        print("\ntraining label range (GATE 4 bounds):")
        for t in b["targets"]:
            print(f"    {t:<12} {b['label_min'][t]:.6g} .. {b['label_max'][t]:.6g}")
        return

    if a.stl:
        pred, rep = predict_stl(a.stl, a.density, b, a.cell)
        print(f"\n{os.path.basename(a.stl)}")
        print(_fmt(rep))
        print(f"    density used: {rep['density_used']:.6f}")
        if a.density not in (None, "measure"):
            ok, err = gate3(rep, a.density)
            if ok is not None:
                print(f"    GATE3 {'pass' if ok else 'FAIL'}: measured VF is "
                      f"{err * 100:+.2f}% from the requested density")
        print("    predictions:")
        for t, v in pred.items():
            lo, hi = b["label_min"][t], b["label_max"][t]
            flag = "" if lo <= v <= hi else "   <- OUTSIDE the training range"
            print(f"      {t:<12} {v:.6g}{flag}")
        return

    if a.batch:
        import pandas as pd
        df = pd.read_csv(a.batch)
        col = next((c for c in ("stl", "path", "file", "STL") if c in df.columns), None)
        if col is None:
            sys.exit("[E] --batch CSV needs a column named stl / path / file")
        rows = []
        for i, r in df.iterrows():
            sp = str(r[col])
            d = r["density"] if "density" in df.columns else None
            if d is None or (isinstance(d, float) and np.isnan(d)):
                d = "measure"
            try:
                pred, rep = predict_stl(sp, d, b, a.cell)
                g2, bad = gate2(rep)
                rec = dict(stl=sp, ok=True, gate2=g2, gate2_detail="; ".join(bad),
                           vf=rep["vf"], density_used=rep["density_used"],
                           euler=rep["euler"], bodies=rep["bodies"])
                rec.update({f"pred_{k}": v for k, v in pred.items()})
            except Exception as ex:
                rec = dict(stl=sp, ok=False, gate2=False, gate2_detail=f"{type(ex).__name__}: {ex}")
            rows.append(rec)
            print(f"  [{i + 1}/{len(df)}] {os.path.basename(sp)} "
                  f"{'ok' if rows[-1]['ok'] else 'FAILED'}")
        out = a.out or os.path.splitext(a.batch)[0] + "_predictions.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\n[OK] {out}")
        return

    ap.print_help()


if __name__ == "__main__":
    main()
