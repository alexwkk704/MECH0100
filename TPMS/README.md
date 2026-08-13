# TPMS — geometry generation and finite element homogenisation

The automation behind the TPMS half of the database: 183 geometries across 17
topologies. This toolchain is ZheZhe Du's work.

## Files

| file | what it does |
|---|---|
| `tpms_stl_batch_export.py` | multi-notebook STL driver. Walks a list of topologies, writes thickness and tolerance into each generator, exports an STL, then validates it — file exists, larger than 2000 bytes, non-zero binary triangle count. |
| `tpms_batch_run (3).py` | single-notebook FEA driver. Writes thickness, mesh edge length and path, runs the homogenisation and collects the tensor. |

## The notebooks

Not in this repository — see `TPMS_NOTEBOOKS.md` for the full list with sizes,
the folder structure they expect, and where to get them.

## Material

TPMS geometries were solved with a polymer solid, 1.8 GPa and Poisson ratio 0.3,
against steel at 200 GPa for ADMS. Every learning target is normalised by the
family's own solid modulus, which is valid here only because the Poisson ratios
match — the normalised stiffness depends on geometry *and* on Poisson ratio, so
dividing by the modulus alone does not make the labels material-free unless
Poisson ratio is common. `ML/scripts/merge_datasets.py` aborts the merge if the
two differ.
