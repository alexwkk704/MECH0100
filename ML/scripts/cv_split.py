"""
cv_split.py — ONE definition of the cross-validation folds, shared by every model.

Why this exists
---------------
RF/GPR (train.py), PointNet (train_pointnet_v2.py) and the side-by-side chart
(compare_models.py) must all score on the SAME folds, or a comparison is
meaningless. Before this file each script split independently: RF/GPR did a
RANDOM 5-fold and PointNet did a single random 80/20. Two problems:

  1. Random splits let near-identical geometries (same density_param, neighbouring
     t_input, i.e. one shape barely perturbed) sit in BOTH train and test. The
     model then "predicts" a row it has effectively already seen, so the reported
     R2/TAI is optimistic (the 2026-07-20 caveat).
  2. The two models used different protocols, so their numbers were not comparable.

This module fixes both. Folds are formed from WHOLE GROUPS (default: one group per
unique adms_type + density_param), so every near-duplicate of a geometry lands in
the same fold and can never straddle train/test. The assignment is DETERMINISTIC
and computed from group sizes only, so any script that reads the same dataset and
the same settings reproduces byte-identical folds without sharing a file.

No hardcoding: the grouping columns come from the settings workbook (group_cols).
If those columns are absent (e.g. a collaborator's dataset without adms_type), it
degrades gracefully to a deterministic per-row split and says so.
"""

import numpy as np
import pandas as pd


def _stem_col(df, preferred="Run"):
    if preferred in df.columns:
        return preferred
    if "file" in df.columns:
        return "file"
    return df.columns[0]


def _keys(df, group_cols, stem_col):
    """Return (per-row group key as str Series, grouped?).

    Grouped when every configured group column is present. Otherwise each row is
    its own group, keyed by its unique stem — a deterministic non-grouped split.
    """
    if group_cols and all(c in df.columns for c in group_cols):
        sub = df[group_cols]
        # A column that is entirely empty contributes NOTHING to the key, so the
        # grouping silently degrades to the remaining columns while the settings
        # sheet still claims leakage protection. That is worse than an error.
        # (Found 2026-08-03: dataset_TPMS.csv has density_param 100% NaN, so
        #  'topology, density_param' collapsed to topology alone.)
        dead = [c for c in group_cols if sub[c].isna().all()]
        if dead:
            raise SystemExit(
                "\n" + "=" * 70 +
                "\n[E] GROUPING IS NOT DOING WHAT THE SETTINGS SAY."
                "\n    These group_cols are EMPTY for every row: " + ", ".join(dead) +
                "\n    They add nothing to the group key, so near-duplicate geometries"
                "\n    would be split across folds and the R2 would be inflated by leakage."
                "\n    Fix group_cols on the train_/pointnet_ sheet to columns this"
                "\n    dataset actually carries. Present and non-empty: " +
                ", ".join(c for c in df.columns if not df[c].isna().all()) +
                "\n" + "=" * 70)
        partial = [c for c in group_cols if sub[c].isna().any()]
        if partial:
            print(f"[!] group_cols with some empty values -> keyed as 'NA': {', '.join(partial)}")
        # fillna BEFORE astype(str): on pandas 3 the new str dtype keeps missing
        # as float NaN, and "|".join then raises TypeError. Older pandas turned it
        # into the literal "nan" and carried on silently. Neither is acceptable.
        return sub.fillna("NA").astype(str).agg("|".join, axis=1), True
    return df[stem_col].astype(str), False


def build_fold_map(full_df, group_cols, n_folds, stem_col="Run", seed=42):
    """Compute the group -> fold assignment from the FULL dataset.

    Greedy largest-group-first balancing: repeatedly place the biggest remaining
    group into the currently-lightest fold. Deterministic (ties broken by group
    name), so it reproduces exactly in every script. Balances fold ROW counts,
    not just group counts, which matters when group sizes vary (1..13 here).

    Returns (fold_of: dict group_key->fold, grouped: bool, stem_col: str).
    """
    stem_col = _stem_col(full_df, stem_col)
    keys, grouped = _keys(full_df, group_cols, stem_col)
    sizes = keys.value_counts()                      # group_key -> row count
    if grouped:
        items = sorted(sizes.items(), key=lambda kv: (-kv[1], str(kv[0])))
    else:
        # every group has size 1; shuffle deterministically for balanced folds
        uniq = sorted(sizes.index.tolist())
        rng = np.random.RandomState(seed)
        rng.shuffle(uniq)
        items = [(g, 1) for g in uniq]
    load = [0] * n_folds
    fold_of = {}
    for g, sz in items:
        f = min(range(n_folds), key=lambda i: load[i])
        fold_of[g] = f
        load[f] += sz
    return fold_of, grouped, stem_col


def folds_for(df, group_cols, fold_of, stem_col):
    """Map any subset/superset frame to its fold ids using a prebuilt fold_of.

    Because fold_of is keyed by the GROUP string (not row index), a subset of the
    data (e.g. rows labelled for one target) inherits exactly the same fold per
    geometry as any other subset. Rows whose group was unseen in fold_of get -1.
    """
    keys, _ = _keys(df, group_cols, stem_col)
    return keys.map(lambda k: fold_of.get(k, -1)).to_numpy()


def assign(full_df, group_cols, n_folds, stem_col="Run", seed=42):
    """Convenience: return (fold_ids aligned to full_df, grouped, stem_col, fold_of)."""
    fold_of, grouped, stem_col = build_fold_map(full_df, group_cols, n_folds, stem_col, seed)
    ids = folds_for(full_df, group_cols, fold_of, stem_col)
    return ids, grouped, stem_col, fold_of
