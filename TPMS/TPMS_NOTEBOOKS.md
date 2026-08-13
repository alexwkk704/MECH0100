# TPMS nTop notebooks

The TPMS toolchain is **45 notebooks totalling 1.71 GiB**, so it is not stored in
this repository. Two of them individually exceed GitHub's 100 MiB per-file limit.

These are ZheZhe Du's notebooks.

## Where they are

**OneDrive — `MECH0100-nTop/TPMS/`**

<https://liveuclac-my.sharepoint.com/:f:/g/personal/ucemkaw_ucl_ac_uk/IgBdsfxh7yrES6lLWA3TIWFQAXt221DHvt-3cSCf4ZdJvxc?e=OXHbT1>

**Ask Alex Wong for access.**

| folder | contents |
|---|---|
| `TPMS STL/` | 27 generators, one per topology — geometry and STL export only |
| `TPMS FEA/` | 18 homogenisation notebooks covering the 17 topologies in `ML/data/dataset_TPMS_quad.csv` |

The FEA notebooks are `-s` saved result copies, so a mesh is baked into each.
They still work as input templates because the inputs are unchanged — that is
also why they are large.

## STL generators — 27

Driven by `tpms_stl_batch_export.py`, which writes only `t_max`, `t_min` (= −t),
`Tolerance` (0.2 mm) and `Path`. Everything else stays at the notebook default.
The driver validates each export: file exists, larger than 2000 bytes, non-zero
binary triangle count.

| file | size |
|---|---|
| `+-Y Generator.ntop` | 21.8 MB |
| `C(+-Y) Generator.ntop` | 20.7 MB |
| `C(D) Generator.ntop` | 39.8 MB |
| `C(G) Generator.ntop` | 24.1 MB |
| `C(I2-Y) Generator.ntop` | 24.6 MB |
| `C(S) Generator.ntop` | 34.4 MB |
| `C(Y) Generator.ntop` | 20.1 MB |
| `D Generator.ntop` | 17.4 MB |
| `D' Generator.ntop` | 25.6 MB |
| `F Generator.ntop` | 15.7 MB |
| `FRD Generator.ntop` | 19.4 MB |
| `G' Generator.ntop` | 23.1 MB |
| `G'2 Generator.ntop` | 23.4 MB |
| `Gyroid Generator.ntop` | 13.7 MB |
| `IWP Generator.ntop` | 3.91 MB |
| `K Generator.ntop` | 17.8 MB |
| `Lidinoid Generator.ntop` | 26.0 MB |
| `Neovius Generator.ntop` | 14.0 MB |
| `OCTO Generator.ntop` | 14.9 MB |
| `P Generator.ntop` | 12.1 MB |
| `P+C(P) Generator.ntop` | 15.9 MB |
| `Q star Generator.ntop` | 15.3 MB |
| `S Generator.ntop` | 23.6 MB |
| `Slotted-P Generator.ntop` | 17.9 MB |
| `Split-P Generator.ntop` | 21.6 MB |
| `W Generator.ntop` | 27.7 MB |
| `Y Generator.ntop` | 24.6 MB |

## FEA notebooks — 18

Driven by `tpms_batch_run (3).py`, which writes `t_max`, `t_min`, `Edge length`
and `Path`.

| file | size |
|---|---|
| `C(+-Y)_density0.21.ntop` | 84.6 MB |
| `C(+-Y)_density0.25.ntop` | 63.6 MB |
| `C(I2-Y) Generator_d0.20.ntop` | 73.2 MB |
| `C(Y) Generator_d0.20.ntop` | 65.5 MB |
| `D' Generator_d0.16.ntop` | 102.0 MB |  ⚠ over GitHub's limit
| `D Generator_density0.20.ntop` | 49.7 MB |
| `FRD Generator_density0.20.ntop` | 49.8 MB |
| `G' Generator_d0.20.ntop` | 40.3 MB |
| `Gyroid Generator_density0.21.ntop` | 22.2 MB |
| `IWP Generator_density0.20.ntop` | 18.5 MB |
| `K Generator_d0.20.ntop` | 188.0 MB |  ⚠ over GitHub's limit
| `Neovius Generator_density0.20.ntop` | 59.7 MB |
| `OCTO Generator_density0.20.ntop` | 97.9 MB |
| `P Generator_density0.20.ntop` | 17.1 MB |
| `P+C(P) Generator_d0.20.ntop` | 50.6 MB |
| `Q star Generator_density0.20.ntop` | 55.7 MB |
| `Slotted-P Generator_density0.20.ntop` | 46.4 MB |
| `Split-P Generator_density0.20.ntop` | 98.5 MB |

## Mesh sizing differs from ADMS, deliberately

The TPMS driver sets **edge length proportional to wall thickness**, so mesh size
scales per topology and per density. ADMS uses a fixed edge length instead. This
is a considered difference between the two families, not an inconsistency.

## Also needed to run these

A density→thickness lookup CSV per topology, with columns `density,t,vf_actual`.
These are not in the repository. The relation is close to linear — for IWP over
0.05–0.35, `density ≈ 0.26653·t + 0.0006`, with the achieved volume fraction
tracking the requested density to within 0.3 %.
