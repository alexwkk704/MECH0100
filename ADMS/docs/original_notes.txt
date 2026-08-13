ADMS LATTICE HOMOGENISATION + YIELD-ONSET BATCH RUNNER
======================================================
Prepared by Alex Wong (UCL MECH0100), July 2026.
Version 2: shear yield-onset added (Week 8).

WHAT IT DOES
------------
For each parameter row you enter, it runs the nTop notebook
(ADMS_DF_generic_v1.ntop) headlessly via ntopcl:
  1. generates the ADMS lattice (Spherene plugin),
  2. homogenises the unit cube -> 6x6 stiffness tensor,
  3. runs a linear static COMPRESSION (bottom fixed, top -0.667% strain
     along Z) and exports the von Mises stress point map,
  4. runs a linear static SHEAR (bottom fixed, top +0.667% strain along X,
     Uy=Uz=0) on the same FE mesh and exports its von Mises point map,
then post-processes everything into one summary spreadsheet.

The three analyses share the same FE volume mesh, so E, G, compression
onset and shear onset are directly comparable per configuration.

REQUIREMENTS (one-time)
-----------------------
  * nTop installed and signed in (open the nTop desktop app and log in once),
    with the Spherene plugin licensed.
  * Python 3.9+ on PATH (python.org installer, tick "Add to PATH").
    The .bat auto-installs the two needed packages (numpy, openpyxl).

HOW TO USE
----------
  1. Open runs_input.xlsx
       - sheet "runs":     one row per simulation
         (Density, Thickness, Inner Size, Size Multi, Seed;
          blank cells fall back to the notebook defaults)
       - sheet "settings": material yield strength Sigma_ys (default 250 MPa,
         steel), solid modulus Es (200 GPa), applied strain.
     Save and close the file.
  2. Double-click RUN_SIMULATION.bat
     Each run takes roughly 30 minutes, 3 hours max, depends on size.
     Adding the shear analysis costs ~3 minutes on top of the previous
     ~21 minutes per run. Progress streams in the window.

     Every invocation creates its OWN dated output folder:
         Results\YYYYMMDD_HH-MM_RUN\
     e.g. Results\20260713_17-24_RUN\  for a batch started 13/07/2026 17:24.
     Previous batches are kept intact - nothing is overwritten. To free
     space, delete old batch folders manually.

     BEFORE any simulation, the script cross-checks each row in runs_input.xlsx
     against Results\Results_summary.xlsx (the MASTER, cumulative across all
     previous batches). For each row already present it prompts:
         [Y] rerun as v2 (or v3...)   [N] skip   [A] yes to all remaining
     All prompts happen up front. Once answered, ntop starts and no more
     input is needed - safe to walk away.

  3. TWO summary spreadsheets, two roles:
       Results\Results_summary.xlsx
           MASTER. Cumulative across every batch, append-only. Reruns of
           the same geometry are kept as v2 / v3 / ... rather than replacing
           v1. Best for analysis / plots / comparison across sessions.
           Adds two columns vs the batch summary: Version + Batch.
       Results\<batch>\Results_summary.xlsx
           Snapshot of just that batch's runs (also carries Version+Batch).
       Both sheets have the same numeric columns.
           ELASTIC:      Rho_rel, E_iso, E/Es, nu_iso, G_iso, G/Gs,
                         TAI / Zener / A^U anisotropy, directional E range,
                         C11 / C12 / C44.
           COMPRESSION:  Sigma_applied, Sigma_p99, Sigma_max_raw,
                         Yield_onset, SCF_p99.
           SHEAR (new):  Tau_applied, Shear_p99, Shear_max_raw,
                         Shear_onset, SCF_shear_p99.
       - sheet "method":  what every column means and how it is computed.
  4. Per-run raw data is in Results\<batch>\Data\<run_name>\ :
       <run>.csv                6x6 C tensor (MPa)
       <run>_stress.csv         compression von Mises point map (Pa)
       <run>_shear_stress.csv   shear von Mises point map (Pa)  <-- new
       <run>.ntop               the executed notebook - open in the nTop
                                GUI to inspect the geometry, mesh and FE
                                results (both compression + shear cases)
       log.txt                  full ntopcl log
       run_info.json            inputs/outputs record

NOTES
-----
  * Each .bat invocation gets its own timestamped batch folder under
    Results\. Nothing is skipped or overwritten across batches - every
    invocation is a fresh, independent set of runs.
  * Within a single batch, the Results_summary.xlsx is built from that
    batch's Data\ folder only.

  * COMPRESSION Yield_onset = Sigma_applied x (Sigma_ys / Sigma_p99):
    the macroscopic compressive stress at which the 99th-percentile local
    von Mises stress reaches the material yield strength (linear-elastic
    scaling). Sigma_applied = E_iso x strain.

  * SHEAR    Shear_onset  = Tau_applied   x (Sigma_ys / Shear_p99):
    the macroscopic shear stress at which the 99th-percentile local
    von Mises stress under the shear load reaches yield.
    Tau_applied = G_iso x strain, where G_iso comes from homogenisation
    (Voigt-Reuss-Hill) and strain is the same 0.00667 as compression.

  * The raw max stress is mesh-sensitive at the cube's cut edges and is
    reported separately (Sigma_max_raw, Shear_max_raw) - use p99 for the
    onset calculation.
  * Some low-density / thin-wall combinations can fail (disconnected
    lattice -> singular FE matrix). The row is marked FAIL; see its log.txt.
  * If ntopcl.exe is not in the default location, set an NTOPCL environment
    variable to its full path.

Questions: ucemkaw@ucl.ac.uk
