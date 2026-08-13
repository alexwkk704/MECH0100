r"""
pointnet_split_8020.py — Farooq's stratified random 80/20 validation of the ADMS
PointNet. Added as an EXTRA validation step next to the grouped 5-fold CV.

Farooq (Teams): use a stratified random split — mix all samples, hold out 20% WITHIN
each geometry type (settings key `stratify_col`; auto-falls back to adms_type ->
topology -> family), so every topology is seen in training and the hidden
20% tests unseen parametric variations (density). Single split; the same protocol
Zhezhe uses.

This is deliberately the SAME model as your grouped-CV trainer: it imports
train_pointnet_v2's own settings loader (load_pn_cfg), cloud loader (build_xy),
tensor loader (load_tensors) and jitter (augment), and it mirrors that file's
PointNet class + fit_fold training loop VERBATIM. The ONLY difference from the
grouped 5-fold CV is the split (single stratified 80/20 instead of 5 grouped folds).
It does NOT modify train_pointnet_v2 or any other script.

  * IF YOU EVER CHANGE the PointNet or the training loop in train_pointnet_v2.py,
    mirror the same change in the PointNet class + fit() below, or the two numbers
    stop being comparable.

Reads data\ read-only. Writes into the current run folder (runs\<stamp>\, via
run_paths): pointnet_8020_results.csv and pointnet_8020_predictions.csv.

Run by RUN_ALL_ML.bat Step 8, or standalone:  python pointnet_split_8020.py
"""

import os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_paths
import tensor_ops
# reuse the SAME loaders/augment as the grouped-CV trainer (module-level, safe import)
from train_pointnet_v2 import load_pn_cfg, build_xy, load_tensors, augment

SPLIT_NAME = "stratified8020"


def stratified_holdout(topo, test_frac, seed):
    """Farooq's stratified random split: hold out test_frac WITHIN each geometry
    type, so every type appears in both train and test. Returns (trainval_idx,
    test_idx)."""
    rng = np.random.RandomState(seed)
    trv, te = [], []
    for t in pd.unique(topo):
        idx = np.where(topo == t)[0]; rng.shuffle(idx)
        n = len(idx); ntest = int(round(n * test_frac))
        ntest = min(max(ntest, 1), n - 1) if n >= 2 else 0
        te.extend(idx[:ntest]); trv.extend(idx[ntest:])
    return np.array(sorted(trv)), np.array(sorted(te))


def main():
    cfg = load_pn_cfg()
    np.random.seed(cfg["random_seed"])
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        sys.exit("[E] PyTorch not installed — run: pip install torch")
    torch.manual_seed(cfg["random_seed"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- data + preprocessing: IDENTICAL to train_pointnet_v2.run() ----
    X, meta = build_xy(cfg)
    if len(X) < 10:
        sys.exit(f"[E] only {len(X)} clouds matched — run pointcloud_prep.py first.")
    tgts = [t for t in cfg["targets"] if t in meta.columns]
    Y = meta[tgts].apply(pd.to_numeric, errors="coerce").values.astype(np.float32)
    MASK = ~np.isnan(Y)
    keep = MASK.any(axis=1)
    X, Y, MASK, meta = X[keep], Y[keep], MASK[keep], meta[keep].reset_index(drop=True)

    if int(cfg.get("use_density_channel", 0)):
        if "Rho_rel" not in meta.columns:
            sys.exit("[E] use_density_channel=1 but no 'Rho_rel' column")
        rho = pd.to_numeric(meta["Rho_rel"], errors="coerce").values.astype(np.float32)
        if np.isnan(rho).any():
            sys.exit("[E] some rows have no Rho_rel for the density channel")
        X = np.concatenate([X, np.repeat(rho[:, None, None], X.shape[1], axis=1)], axis=2).astype(np.float32)
        print(f"[i] density channel ON -> {X.shape}")
    in_dim = X.shape[2]

    use_log = int(cfg.get("log_targets", 0))
    if use_log:
        if (Y[MASK] <= 0).any():
            print("[!] non-positive targets -> log_targets disabled"); use_log = 0
        else:
            Y = np.where(MASK, np.log10(np.where(MASK, Y, 1.0)), np.nan).astype(np.float32)
            print("[i] targets log10-transformed")

    # ---- which column defines "one geometry type" for the stratified split ----
    # NOT hardcoded to adms_type: that column only exists in an ADMS-only dataset.
    # merge_datasets.py drops it (it is ADMS-only) and writes the unified `topology`
    # instead, so a COMBINED run has no adms_type at all. Settings key
    # `stratify_col` on the pointnet_<PROFILE> sheet wins; otherwise take the first
    # of adms_type / topology / family that is actually present.
    want = str(cfg.get("stratify_col", "") or "").strip()
    order = ([want] if want else []) + ["adms_type", "topology", "family"]
    strat_col = next((c for c in order if c in meta.columns), None)
    if strat_col is None:
        sys.exit("[E] no column to stratify on. Tried: " + ", ".join(order) +
                 ".  Present: " + ", ".join(map(str, meta.columns)) +
                 ".  Set `stratify_col` on the pointnet sheet to one of them.")
    if want and want != strat_col:
        print(f"[!] stratify_col='{want}' is not in the dataset - falling back to '{strat_col}'")
    STRAT_COL = strat_col
    topo = meta[STRAT_COL].astype(str).values
    print(f"[i] stratifying on '{STRAT_COL}'  ({len(pd.unique(topo))} groups)")

    # ---- PointNet + training loop: VERBATIM mirror of train_pointnet_v2 ----
    class PointNet(nn.Module):
        def __init__(self, n_out, in_dim=3, deep=True):
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
                    nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, n_out))
        def forward(self, x):
            x = x.transpose(1, 2)
            x = self.enc(x).max(dim=2).values
            return self.head(x)

    clip = float(cfg.get("grad_clip", 0) or 0)

    def masked_mse(pred, target, mask):
        diff = (pred - target) * mask
        return (diff ** 2).sum() / torch.clamp(mask.sum(), min=1.0)

    Ctens = load_tensors(meta, "Run", cfg) if cfg["rotation_aug"] else [None] * len(meta)

    def fit(tr, va, te):
        """VERBATIM mirror of train_pointnet_v2.fit_fold (single split; fold seed=0)."""
        ymean = np.array([Y[tr, j][MASK[tr, j]].mean() if MASK[tr, j].any() else 0.0
                          for j in range(len(tgts))], dtype=np.float32)
        ystd = np.array([Y[tr, j][MASK[tr, j]].std() if MASK[tr, j].any() else 1.0
                         for j in range(len(tgts))], dtype=np.float32) + 1e-8
        net = PointNet(len(tgts), in_dim=in_dim, deep=bool(int(cfg.get("deep_head", 0)))).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
        decay = float(cfg.get("lr_decay", 1.0))
        sched = (torch.optim.lr_scheduler.StepLR(opt, step_size=int(cfg.get("lr_step", 200)), gamma=decay)
                 if decay and decay != 1.0 else None)
        Ytr_s = np.where(MASK[tr], (Y[tr] - ymean) / ystd, 0.0).astype(np.float32)
        Yva_s = np.where(MASK[va], (Y[va] - ymean) / ystd, 0.0).astype(np.float32)
        Mtr, Mva = MASK[tr].astype(np.float32), MASK[va].astype(np.float32)
        Xtr, Xva = X[tr], X[va]

        if cfg["rotation_aug"] and cfg["n_rot"] > 0:
            rng = np.random.RandomState(cfg["random_seed"] + 1000)   # fold 0
            es = cfg["es_gpa"]; fdep, finv = set(cfg["frame_dependent"]), set(cfg["rotation_invariant"])
            aX, aY, aM = [], [], []
            for pos, gi in enumerate(tr):
                C = Ctens[gi]
                if C is None:
                    continue
                for _ in range(cfg["n_rot"]):
                    R = tensor_ops.random_rotation(rng)
                    xc = X[gi].copy()
                    p = X[gi][:, :3] @ R.T; p = p - p.mean(axis=0)
                    ext = float(np.ptp(p, axis=0).max())
                    if ext > 0:
                        p = p / ext
                    xc[:, :3] = p
                    Crot = tensor_ops.rotate_C(C, R)
                    yrow = np.zeros(len(tgts), np.float32); mrow = np.zeros(len(tgts), np.float32)
                    for j, t in enumerate(tgts):
                        if t in fdep:
                            val = tensor_ops.frame_dependent_value(t, Crot, es)
                            if val is None or (use_log and val <= 0):
                                continue
                            v = np.log10(val) if use_log else val
                            yrow[j] = (v - ymean[j]) / ystd[j]; mrow[j] = 1.0
                        elif t in finv:
                            yrow[j] = Ytr_s[pos, j]; mrow[j] = Mtr[pos, j]
                    aX.append(xc.astype(np.float32)); aY.append(yrow); aM.append(mrow)
            if aX:
                Xtr = np.concatenate([Xtr, np.stack(aX)], axis=0)
                Ytr_s = np.concatenate([Ytr_s, np.stack(aY)], axis=0)
                Mtr = np.concatenate([Mtr, np.stack(aM)], axis=0)
                print(f"    +{len(aX)} rotated copies (train {len(tr)} -> {len(Xtr)} rows)")

        best_val, best_state = float("inf"), None
        for ep in range(cfg["epochs"]):
            net.train(); perm = np.random.permutation(len(Xtr))
            for b in range(0, len(Xtr), cfg["batch_size"]):
                bi = perm[b:b + cfg["batch_size"]]
                if len(bi) < 2:
                    continue
                xb = torch.tensor(augment(Xtr[bi], cfg), device=dev)
                yb = torch.tensor(Ytr_s[bi], device=dev); mb = torch.tensor(Mtr[bi], device=dev)
                opt.zero_grad(); loss = masked_mse(net(xb), yb, mb); loss.backward()
                if clip > 0:
                    nn.utils.clip_grad_norm_(net.parameters(), clip)
                opt.step()
            if sched:
                sched.step()
            net.eval()
            with torch.no_grad():
                vloss = float(masked_mse(net(torch.tensor(Xva, device=dev)),
                                         torch.tensor(Yva_s, device=dev),
                                         torch.tensor(Mva, device=dev)).item())
            if vloss < best_val:
                best_val = vloss
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            return net(torch.tensor(X[te], device=dev)).cpu().numpy() * ystd + ymean

    # ---- stratified 80/20: hold out test_frac within each stratify group ----
    trv, te = stratified_holdout(topo, float(cfg.get("test_frac", 0.2)), cfg["random_seed"])
    # carve a val slice from the 80% for epoch selection (mirrors the CV's 3-way split)
    rng = np.random.RandomState(cfg["random_seed"])
    vf = float(cfg.get("val_frac", 0.15))
    vsz = max(2, int(round(vf * len(trv))))
    perm = rng.permutation(len(trv))
    va = trv[perm[:vsz]]; tr = trv[perm[vsz:]]
    print(f"[i] stratified 80/20 by {STRAT_COL}: train={len(tr)} val={len(va)} test={len(te)} "
          f"(groups {', '.join(pd.unique(topo))})")

    t0 = time.time()
    pred_te = fit(tr, va, te)
    secs = time.time() - t0
    print(f"[i] 80/20 training time: {secs:.0f}s")

    # ---- score the hidden 20% (un-log first, same as the CV scorer) ----
    yt_all = Y.copy()
    pr = pred_te.copy()
    if use_log:
        pr = np.power(10.0, pr)
        yt_all = np.where(MASK, np.power(10.0, yt_all), np.nan)
    print(f"\n=== PointNet — {SPLIT_NAME} (hidden 20%, N={len(te)}) ===")
    results, rec = [], []
    for j, t in enumerate(tgts):
        ok = MASK[te, j]
        if ok.sum() < 3:
            print(f"  {t:14s} only {int(ok.sum())} scored — skipped"); continue
        yt = yt_all[te][ok, j]; pp = pr[ok, j]
        ss = np.sum((yt - yt.mean()) ** 2)
        r2 = 1 - np.sum((yt - pp) ** 2) / ss if ss > 1e-9 * max(1.0, np.sum(yt ** 2)) else float("nan")
        m = np.abs(yt) > 1e-9
        mape = np.mean(np.abs((pp[m] - yt[m]) / yt[m])) * 100 if m.any() else float("nan")
        print(f"  {t:14s} R2={r2:.3f}  MAPE={mape:.1f}%  (N={int(ok.sum())})")
        results.append(dict(target=t, model="PointNet_v2", split=SPLIT_NAME,
                            N_test=int(ok.sum()), R2=round(float(r2), 4), MAPE=round(float(mape), 2),
                            train_seconds=round(secs, 1)))
    for a, i in enumerate(te):
        for j, t in enumerate(tgts):
            rec.append(dict(Run=meta.iloc[i]["Run"], **{STRAT_COL: topo[i]}, target=t,
                            y_pred=float(pr[a, j]),
                            y_true=(float(yt_all[i, j]) if MASK[i, j] else "")))

    out_csv = run_paths.run_path("pointnet_8020_results.csv")
    pd.DataFrame(results).to_csv(out_csv, index=False)
    pd.DataFrame(rec).to_csv(run_paths.run_path("pointnet_8020_predictions.csv"), index=False)
    print(f"[OK] wrote {out_csv}")
    print(f"[OK] wrote {run_paths.run_path('pointnet_8020_predictions.csv')}")


if __name__ == "__main__":
    main()
