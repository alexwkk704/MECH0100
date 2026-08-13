# ADMS — geometry generation and finite element homogenisation

The automation that produced the ADMS half of the database: 163 geometries
across three variants, generated headlessly in nTop and homogenised into 6×6
stiffness tensors.

> **The flat layout of this folder is required, not accidental.**
> `ntop_batch.py` resolves everything relative to its own directory —
> `SCRIPT_DIR = Path(__file__).resolve().parent` — and then looks for
> `input_template_<type>.json`, `runs_input.xlsx`, `Results/`, and any
> `*.ntop` beside itself. Every batch file does `cd /d "%~dp0"` before calling
> Python. Moving these into subfolders breaks the pipeline. Only `docs/` is
> safe to separate, because nothing reads it.

## The driver

| file | what it does |
|---|---|
| **`ntop_batch.py`** | **the production driver.** Reads the run sheet, picks the right nTop file for the variant, writes the input JSON, calls `ntopcl`, parses the outputs, appends to the master workbook. Resume-safe: a killed batch continues where it stopped. |

## Supporting scripts

| file | what it does |
|---|---|
| `verify_batch.py` | checks a finished batch — every requested run present, every output parsed, no silent gaps |
| `chart_builder.py` | the scaling-law fits and their charts |
| `RECONCILE_MASTER.py` | reconciles the master results workbook against what is actually on disk |
| `refresh_charts.py` | redraws charts from the current master without re-running anything |

## Configuration

| file | what it is |
|---|---|
| `input_template_DF.json` | the ten-input schema for the DF variant |
| `input_template_flow.json` | the same, Flow variant |
| `input_template_raw.json` | the same, Raw variant |
| `output_template.json` | the output schema |
| `runs_input.xlsx` | **the run sheet** — one row per geometry on the `runs` sheet, plus a `settings` sheet holding the solid material constants and the path to `ntopcl.exe` |

A template is regenerated automatically whenever its nTop file is newer than it,
so a stale schema cannot silently mis-map the inputs.

## Entry points — double-click these

| file | what it runs |
|---|---|
| `RUN_SIMULATION.bat` | the standard batch, straight from `runs_input.xlsx` |
| `SMOKE_TEST_3TYPES.bat` | one geometry per variant, to prove the chain works |
| `RUN_OVERNIGHT_HIGHVF.bat` | the high-density batch |
| `RECONCILE_MASTER.bat` | reports what is missing first, then writes |
| `REFRESH_CHARTS.bat` | redraws the charts only |

## `docs/`

| file | what it is |
|---|---|
| `ADMS_nTop_files.md` | all nine nTop files by name, class and size, what each class does, and where to download them |
| `Parameter_dependence_summary.md` | how each input affects the result |
| `original_notes.txt` | the original working notes for this folder |

## Fit conventions, which are not obvious from the code

`chart_builder.py` applies two rules that matter if you re-derive anything:

1. **Deduplication keeps repeats whose results differ.** Only bit-identical
   repeats are dropped, because a repeat returning a different number is an
   independent measurement and carries real scatter. *This is the opposite of
   the rule used to build the machine-learning dataset, where one row per unique
   geometry is mandatory to avoid leakage.*
2. **Scaling laws are fitted only on `Inner Size = 15` and `Size Multi = 1.6`,
   and separately per variant.** Other envelope settings follow different
   curves, and pooling the three variants into one fit changes the exponent.

## The nTop files

Not in this repository — 1.50 GiB across nine files, six of them individually
over GitHub's 100 MiB limit. They are on OneDrive:

<https://liveuclac-my.sharepoint.com/:f:/g/personal/ucemkaw_ucl_ac_uk/IgBdsfxh7yrES6lLWA3TIWFQAXt221DHvt-3cSCf4ZdJvxc?e=OXHbT1>

Ask Alex Wong for access. `docs/ADMS_nTop_files.md` lists them all.
