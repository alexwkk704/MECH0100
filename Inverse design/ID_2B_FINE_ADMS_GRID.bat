@echo off
REM ===========================================================================
REM  ID_2B_FINE_ADMS_GRID.bat  --  double-click, no arguments.  ~30 minutes.
REM
REM  INVERSE DESIGN PHASE 2 (ADMS): GENERATE NEW GEOMETRY.
REM
REM  This is the step the whole method exists for. Phase 1 could only rank rows
REM  that already had an FEA result, so the model was reproducing numbers we
REM  already had. Here the model is the ONLY source of properties - none of
REM  these geometries has ever been simulated, and none of them is in the
REM  training set. GATE 5 requires the design carried into Phase 4 to come from
REM  this step.
REM
REM  WHAT IT DOES, PER CANDIDATE
REM      1  nTop generates the STL from Density / Thickness / Seed /
REM         Inner Size / Size Multi        (~2 min on the trimmed notebook)
REM      2  read p_rel out of the run - the MEASURED relative density.
REM         Density is never assumed: it is an outcome of Density x Thickness,
REM         and it is what the model's density channel was trained on.
REM      3  mesh checks, then predict.
REM
REM  THE GRID
REM      4 values of Density  (+/- 10 percent of the Phase 1 winner)
REM      4 values of Thickness (+/- 15 percent)
REM      Both axes are clipped to the range that topology was actually sampled
REM      over, so we are not extrapolating the GENERATOR - GATE 4 only guards
REM      the model, nothing else would catch that.
REM      Any pair that already exists in the database is skipped, which is
REM      GATE 5 enforced by construction.
REM
REM  GATES
REM      GATE 2  watertight, consistent winding, single body, even Euler number
REM      GATE 3  measured STL VF vs the notebook's own p_rel, to 7 percent
REM              (across the 163 ADMS rows those two already differ by up to
REM              6.59 percent, so a tighter threshold would fail real geometry)
REM      GATE 4  reject predictions outside the training label range
REM      GATE 5  by construction - see above
REM
REM  RESTARTABLE. A candidate whose STL is already there is reused, so an
REM  interrupted run picks up where it stopped. Pass --force to regenerate.
REM
REM  WHAT IT TOUCHES
REM      Writes ONLY under Share\Inverse design\search\. Share\ADMS is read-only:
REM      the notebook is copied once into search\ntop\ and driven from there.
REM      Nothing it writes ever enters the training dataset.
REM
REM      ID_2B_FINE_ADMS_GRID.bat           target E/Es = 0.10
REM      ID_2B_FINE_ADMS_GRID.bat 0.09      a different target
REM ===========================================================================
cd /d "%~dp0"

set "TGT=%~1"
if "%TGT%"=="" set "TGT=0.10"
set "ML_PROFILE=COMBINED"

echo.
echo ============================================================
echo   PHASE 2B  fine grid - GENERATES NEW GEOMETRY
echo   target E/Es = %TGT%
echo   16 nTop runs, roughly 30 minutes. Restartable.
echo ============================================================

if not exist "..\ML\Final Model\COMBINED\model_COMBINED.pt" echo [E] no exported model - run Share\ML\10_EXPORT_FORWARD_MODEL.bat first & pause & exit /b 1
if not exist "search\results\phase2_adms_verify.json" echo [E] run ID_2A_FINE_ADMS_VERIFY.bat first - the trimmed notebook has not been checked against the database & pause & exit /b 1

pip install numpy pandas openpyxl trimesh torch --quiet

python scripts\fine_adms.py --grid --target %TGT%
if errorlevel 1 goto :err

echo.
echo ============================================================
echo   DONE.  search\results\
echo       phase2_adms_E_over_Es_*.csv            every candidate
echo       phase2_adms_E_over_Es_*_manifest.json  what produced them
echo   STLs in  search\candidates\adms\
echo.
echo   Every geometry above is NEW. Its properties come from the
echo   model alone - Phase 4 runs ONE real FEA to test that.
echo ============================================================
pause
exit /b 0

:err
echo.
echo ***** Phase 2 failed - see above.
echo ***** Per-candidate nTop logs: search\runs\adms\<candidate>\log.txt
pause
exit /b 1
