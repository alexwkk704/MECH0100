"""
RECONCILE_MASTER.py — make sure NO completed run is missing from the master.

Scans every  Results/<batch>/Data/<run>/  folder on disk, compares against
Results/Results_summary.xlsx, and appends anything the master is missing.

Why this exists
---------------
The master summary is only written at the END of a batch (update_master).
If a batch is interrupted — Ctrl+C, SDK freeze, crash, power loss — every run
that already finished is safely on disk but never reaches the master, so it is
invisible to the charts, the scaling-law fits and the ML dataset.
(2026-07-18: 16 completed flow runs were stranded exactly this way.)

Run this any time. It is safe to run repeatedly: runs already in the master are
skipped, nothing is overwritten, and a timestamped backup is taken first.

How a run is judged COMPLETE
----------------------------
    <run>/<run>.csv exists and is > 100 bytes   (the homogenised C-tensor)
Runs without it (geometry failed / killed mid-solve) are reported as INCOMPLETE
and skipped — they have no analysable result.

Identity: one master row per (Run, Batch). The same geometry re-run in a later
batch is a new row with the next Version (v2, v3...), matching ntop_batch's
append-only convention.

All derived columns are produced by ntop_batch.build_summary itself, so ingested
rows are byte-for-byte consistent with rows written by a normal batch.

Usage
-----
    python RECONCILE_MASTER.py              # report, then ask before writing
    python RECONCILE_MASTER.py --dry-run    # report only, never writes
    python RECONCILE_MASTER.py --yes        # write without asking
"""

import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import ntop_batch as nb   # noqa: E402  (reuse its column defs + row builder)


def _has_result(rdir: Path) -> bool:
    """Has an analysable homogenisation result."""
    csv = rdir / f"{rdir.name}.csv"
    return csv.exists() and csv.stat().st_size > 100


def _has_metadata(rdir: Path) -> bool:
    """Has run_info.json with the input parameters.

    Without it the run's Density/Thickness/Inner_Size/Size_Multi/Seed cannot be
    recovered (the stem alone is not always parseable), and ingesting it would
    put a row with null parameters into the master — which would corrupt the
    scaling-law fits, the dedup keys and parameter_effects. Legacy pre-2026-07-13
    leftovers (e.g. 'ADMS_DF', 'ADMS_DF_7cube_seed1') are exactly this case and
    were deliberately excluded from the original migration as benchmark/junk.
    """
    import json
    p = rdir / "run_info.json"
    if not p.exists():
        return False
    try:
        return bool(json.loads(p.read_text(encoding="utf-8")).get("inputs"))
    except Exception:
        return False


def _is_complete(rdir: Path) -> bool:
    return _has_result(rdir) and _has_metadata(rdir)


def _load_settings():
    """Settings sheet from runs_input.xlsx; fall back to script defaults."""
    try:
        _rows, settings = nb.load_input_xlsx()
        return settings
    except SystemExit:
        pass
    except Exception:
        pass
    print("[i] could not read runs_input.xlsx settings — using defaults")
    return dict(nb.DEFAULT_SETTINGS)


def scan():
    """Return (batch_dir -> [ingestable run dirs]), no-result list, no-metadata list."""
    found, no_result, no_meta = defaultdict(list), [], []
    if not nb.RESULTS_DIR.exists():
        return found, no_result, no_meta
    for batch in sorted(p for p in nb.RESULTS_DIR.iterdir() if p.is_dir()):
        data = batch / "Data"
        if not data.is_dir():
            continue
        for rdir in sorted(p for p in data.iterdir() if p.is_dir()):
            if not _has_result(rdir):
                no_result.append(rdir)
            elif not _has_metadata(rdir):
                no_meta.append(rdir)
            else:
                found[batch].append(rdir)
    return found, no_result, no_meta


def main():
    dry = "--dry-run" in sys.argv
    assume_yes = "--yes" in sys.argv

    print(f"RECONCILE MASTER — scanning {nb.RESULTS_DIR}\n")
    found, no_result, no_meta = scan()
    n_disk = sum(len(v) for v in found.values())

    master_rows = nb.load_master_rows() if nb.MASTER_SUMMARY_XLSX.exists() else []
    have = {(r.get("Run"), r.get("Batch")) for r in master_rows}
    total_on_disk = n_disk + len(no_result) + len(no_meta)
    print(f"on disk : {total_on_disk} run folder(s) "
          f"({n_disk} ingestable) across {len(found)} batch folder(s)")
    print(f"master  : {len(master_rows)} rows")

    # Only report skips that are ALSO absent from the master — a legacy run with
    # no run_info.json that was migrated in long ago is not a problem.
    def _key(p):
        return (p.name, p.parent.parent.name)

    miss_no_result = [p for p in no_result if _key(p) not in have]
    miss_no_meta = [p for p in no_meta if _key(p) not in have]
    if miss_no_result:
        print(f"\nskipped : {len(miss_no_result)} run(s) with NO result "
              f"(no C-tensor CSV — geometry failed or killed mid-solve)")
        for p in miss_no_result[:10]:
            print(f"            {p.parent.parent.name}/{p.name}")
        if len(miss_no_result) > 10:
            print(f"            ... and {len(miss_no_result) - 10} more")
    if miss_no_meta:
        print(f"\nskipped : {len(miss_no_meta)} run(s) with NO run_info.json — "
              f"parameters unrecoverable, would insert null-parameter rows "
              f"(legacy/benchmark leftovers)")
        for p in miss_no_meta:
            print(f"            {p.parent.parent.name}/{p.name}")

    # which (run, batch) pairs are missing?
    missing = defaultdict(list)
    for batch, rdirs in found.items():
        for rdir in rdirs:
            if (rdir.name, batch.name) not in have:
                missing[batch].append(rdir.name)
    n_missing = sum(len(v) for v in missing.values())

    print(f"\nMISSING from master: {n_missing} run(s)")
    if not n_missing:
        print("[OK] master is complete — every run on disk is recorded.")
        return
    for batch, stems in sorted(missing.items()):
        print(f"  {batch.name}  ({len(stems)} run(s))")
        for s in stems:
            print(f"      {s}")

    if dry:
        print("\n[dry-run] nothing written.")
        return
    if not assume_yes:
        ans = input(f"\nAppend these {n_missing} run(s) to the master? [y/N] ")
        if ans.strip().lower() not in ("y", "yes"):
            print("aborted — nothing written.")
            return

    # backup master before touching it
    if nb.MASTER_SUMMARY_XLSX.exists():
        stamp = time.strftime("%Y%m%d_%H-%M")
        bak = nb.MASTER_SUMMARY_XLSX.with_name(
            f"Results_summary.backup_reconcile_{stamp}.xlsx")
        shutil.copy(nb.MASTER_SUMMARY_XLSX, bak)
        print(f"\n[backup] {bak.name}")

    settings = _load_settings()

    # build_summary and the chart builders read the module-level material
    # properties, which are normally published by ntop_batch.main(). main() does
    # not run here, so publish them explicitly or ingested rows would be scaled
    # with steel regardless of what runs_input.xlsx says.
    nb.ES_GPA_ACTIVE = float(settings["es_gpa"])
    nb.N_STIFF_ACTIVE = float(settings.get("n_stiff", nb.N_STIFF_ACTIVE))
    try:
        import chart_builder as _cb
        _cb.ES_GPA = float(settings["es_gpa"])
    except Exception:
        pass

    # Version numbering must continue from what the master already holds, and
    # from rows added earlier in this same reconcile pass.
    counts = nb.master_stem_counts(master_rows)

    new_rows = []
    for batch in sorted(missing):
        want = set(missing[batch])
        # Point ntop_batch at this batch folder, then let it build the rows.
        # ALL THREE globals must be set: build_summary saves the batch-level
        # workbook to SUMMARY_XLSX, and leaving it None makes openpyxl call
        # wb.save(None) -> "'NoneType' object has no attribute 'write'".
        nb.BATCH_DIR = batch
        nb.DATA_DIR = batch / "Data"
        nb.SUMMARY_XLSX = batch / "Results_summary.xlsx"
        stem_versions = {s: nb.next_version(counts.get(s, 0)) for s in want}
        try:
            rows = nb.build_summary(settings, stem_versions)
        except Exception as e:
            print(f"[!] {batch.name}: build_summary failed ({e}) — skipped")
            continue
        keep = [r for r in rows if r.get("Run") in want]
        for r in keep:
            counts[r["Run"]] = counts.get(r["Run"], 0) + 1
        new_rows.extend(keep)
        print(f"[built] {batch.name}: {len(keep)} row(s)")

    if not new_rows:
        print("[!] nothing could be built — master unchanged.")
        return

    nb.update_master(new_rows)
    print(f"\n[OK] appended {len(new_rows)} run(s) to the master.")
    print("     charts + analysis sheets rebuilt.")

    # verify
    after = nb.load_master_rows()
    still = [(r, b.name) for b, ss in missing.items() for r in ss
             if (r, b.name) not in {(x.get("Run"), x.get("Batch")) for x in after}]
    if still:
        print(f"[!] {len(still)} run(s) STILL missing after write — investigate:")
        for r, b in still[:10]:
            print(f"      {b}/{r}")
    else:
        print("[verified] every completed run on disk is now in the master.")


if __name__ == "__main__":
    main()
