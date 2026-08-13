# TPMS — geometry generation and finite element homogenisation

The automation behind the TPMS half of the database: 183 geometries across 17
topologies. This toolchain is ZheZhe Du's work.

## Scripts

| file | what it does |
|---|---|
| `tpms_stl_batch_export.py` | **multi-file STL driver.** Walks a list of topologies, writes thickness and tolerance into each generator, exports an STL, then validates it — file exists, larger than 2000 bytes, non-zero binary triangle count. |
| `tpms_batch_run (3).py` | **single-file FEA driver.** Writes thickness, mesh edge length and path, runs the homogenisation and collects the tensor. |

## `docs/`

| file | what it is |
|---|---|
| `TPMS_nTop_files.md` | all 45 nTop files by name and size, the folder structure they expect, and where to download them |

## Mesh sizing differs from ADMS, deliberately

The TPMS driver sets **edge length proportional to wall thickness**, so mesh size
scales per topology and per density. ADMS uses a fixed edge length instead. This
is a considered difference between the two families, not an inconsistency.

## Material

TPMS geometries were solved with a polymer solid, 1.8 GPa and Poisson ratio 0.3,
against steel at 200 GPa for ADMS. Every learning target is normalised by the
family's own solid modulus, which is valid here only because the Poisson ratios
match — normalised stiffness depends on geometry *and* on Poisson ratio, so
dividing by the modulus alone does not make the labels material-free unless the
Poisson ratio is common. `ML/scripts/merge_datasets.py` aborts the merge if the
two differ.

## The nTop files

Not in this repository — 1.71 GiB across 45 files. They are on OneDrive:

<https://liveuclac-my.sharepoint.com/:f:/g/personal/ucemkaw_ucl_ac_uk/IgBdsfxh7yrES6lLWA3TIWFQAXt221DHvt-3cSCf4ZdJvxc?e=OXHbT1>

Ask Alex Wong for access. `docs/TPMS_nTop_files.md` lists them all.
