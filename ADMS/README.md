# ADMS — geometry generation and finite element homogenisation

The automation that produced the ADMS half of the database: 163 geometries
across three variants, generated headlessly in nTop and homogenised into 6×6
stiffness tensors.

## Files

| file | what it does |
|---|---|
| `ntop_batch.py` | **the production driver.** Reads a run sheet, resolves the right notebook per variant, writes the input JSON, calls `ntopcl`, parses the outputs, appends to the master workbook. Resume-safe — a killed batch continues where it stopped. |
| `verify_batch.py` | checks a completed batch: every requested run present, every output parsed, no silent gaps |
| `chart_builder.py` | the scaling-law fits and their charts |
| `RECONCILE_MASTER.py` | reconciles the master results workbook against what is on disk |
| `refresh_charts.py` | redraws charts from the current master without re-running anything |
| `input_template_{DF,flow,raw}.json` | the input schema per variant — ten inputs |
| `output_template.json` | the output schema |
| `runs_input.xlsx` | the run sheet: one row per geometry, plus a settings sheet holding the solid material constants |
| `*.bat` | entry points |

## Fit conventions, which are not obvious from the code

`chart_builder.py` applies two rules that matter if you re-derive anything:

1. **Deduplication keeps repeats whose results differ.** Only bit-identical
   repeats are dropped. A repeat that returns a different number is an
   independent measurement and carries real scatter. *This is the opposite of
   the rule used to build the machine-learning dataset, where one row per unique
   geometry is mandatory to avoid leakage.*
2. **Scaling laws are fitted only on `Inner Size = 15` and `Size Multi = 1.6`,
   and separately per variant.** Other envelope settings follow different
   curves, and pooling the three variants into one fit changes the exponent.

## The notebooks

Not in this repository — see `ADMS_NOTEBOOKS.md` for what they are, what each
class does, and where to get them.
