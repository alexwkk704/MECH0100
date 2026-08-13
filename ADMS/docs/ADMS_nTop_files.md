# ADMS nTop files

The `.ntop` nTop files that drive ADMS geometry generation and finite element
homogenisation are **not stored in this repository**. They total 1.50 GiB, and
six of the nine exceed GitHub's hard limit of 100 MiB for a single file.
Compression does not rescue them — a solved FEA nTop file only reaches about
65 % of its original size, because the baked-in mesh is close to incompressible.

## Where they are

**OneDrive — `MECH0100-nTop/ADMS/`**

<https://liveuclac-my.sharepoint.com/:f:/g/personal/ucemkaw_ucl_ac_uk/IgBdsfxh7yrES6lLWA3TIWFQAXt221DHvt-3cSCf4ZdJvxc?e=OXHbT1>

**Ask Alex Wong for access.**

## The three classes

| class | what it does |
|---|---|
| `*_generic_v1` | the **full production nTop file** — generates the lattice, meshes it, runs the six unit-strain load cases, and exports both the STL and the stiffness tensor. This is what `ntop_batch.py` drives, and what produced all 163 ADMS rows in `ML/data/dataset_ADMS.csv`. |
| `*_only_STL` | trimmed to geometry and STL export only. Used by the inverse-design fine search, where candidate geometries are generated but not yet simulated. |
| `*_only_FEA` | trimmed to the homogenisation half. Takes an existing STL and returns the tensor. |

Each class exists for all three ADMS variants — **DF**, **Flow** and **Raw**.

## The files

| file | class | variant | size | on OneDrive |
|---|---|---|---|---|
| `ADMS_DF_generic_v1.ntop` | generic | DF | 116 MiB | ✅ |
| `ADMS_Flow_generic_v1.ntop` | generic | Flow | 128 MiB | ✅ |
| `ADMS_Raw_generic_v1.ntop` | generic | Raw | 107 MiB | ✅ |
| `ADMS_DF_only_STL.ntop` | only_STL | DF | 94 MiB | ✅ |
| `ADMS_Flow_only_STL.ntop` | only_STL | Flow | 108 MiB | ✅ |
| `ADMS_Raw_only_STL.ntop` | only_STL | Raw | 87 MiB | ✅ |
| `ADMS_DF_only_FEA.ntop` | only_FEA | DF | 405 MiB | ✅ |
| `ADMS_Flow_only_FEA.ntop` | only_FEA | Flow | 403 MiB | ✅ |
| `ADMS_Raw_only_FEA.ntop` | only_FEA | Raw | 87 MiB | ✅ |

All nine are on the OneDrive. The three `generic_v1` nTop files are the ones that
produced every row of `ML/data/dataset_ADMS.csv`, so the database is reproducible
from these files together with `ntop_batch.py` and `runs_input.xlsx`.

## Where they go if you restore them

Place them **flat in the `ADMS/` folder**, beside `ntop_batch.py` — not in a
subfolder. `ntop_batch.py::resolve_nTop file_for_type()` globs `*.ntop` in its own
directory and filters on the substring `generic`, which is how the trimmed
nTop files live in the same folder without confusing the production resolver.

## Input schema

The generic nTop files take ten inputs, listed in `input_template_DF.json`:
Density, Thickness, Seed, Out Path, Inner Size, Size Multi, Stress Path,
Shear Stress Path, STL Path, Cell Size.

**Density is a target, not a measurement.** The nTop file returns the achieved
relative density; that measured value is what enters the dataset, and density is
never an input to the machine-learning model.

The trimmed `only_STL` and `only_FEA` nTop files have their own schemas — dump
them with `ntopcl -t <nTop file>` rather than assuming they match the generic.
