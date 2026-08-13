@echo off
REM ===========================================================================
REM  ID_3C_FINE_TPMS_GRID.bat  --  double-click, no arguments.  ~25 minutes.
REM
REM  STEP 3 OF 3 ON THE TPMS SIDE. Run ID_3A and ID_3B first.
REM
REM  GENERATES NEW GEOMETRY. This is the TPMS half of what ID_2B does for ADMS.
REM  None of these geometries has ever been simulated and none is in the
REM  training set, so the model is the only source of their properties. GATE 5
REM  requires the design carried into Phase 4 to come from here.
REM
REM  PER CANDIDATE
REM      1  invert the calibration measured in ID_3A to get t for the wanted
REM         density. Inverting outside the scouted range is REFUSED - past it
REM         we have no evidence the relation holds, and GATE 4 only guards the
REM         model, not the generator.
REM      2  nTop generates the STL from t_max = t, t_min = -t, Tolerance 0.2 mm
REM      3  MEASURE the volume fraction off the STL. That measured value is
REM         what the model is fed - the requested density only chose t. For
REM         TPMS rows Rho_rel == VF exactly, so it is the same quantity the
REM         density channel was trained on.
REM      4  mesh checks, then predict.
REM
REM  THE GRID
REM      Density from the Phase 1 winner, plus and minus 0.035, step 0.005.
REM      Anything within half a step of a density already in the database is
REM      skipped - GATE 5 by construction. On the current files that is about
REM      12 new geometries out of 15 grid points.
REM
REM  GATES
REM      GATE 2  watertight, consistent winding, single body, even Euler.
REM              Checked against Zhezhe's real FRD_0.26 STL: passes, Euler -68.
REM      GATE 3  measured VF vs the density asked for, to 3 percent
REM      GATE 4  reject predictions outside the training label range
REM      GATE 5  by construction - see above
REM
REM  RESTARTABLE. A candidate whose STL is already there is reused, so an
REM  interrupted run picks up where it stopped. Pass --force to regenerate.
REM
REM  The wall-thickness check is deliberately NOT run on all 12 - it belongs in
REM  ID_3B, where it tests the chain against a known answer. Here there is no
REM  known answer to compare against.
REM
REM      ID_3C_FINE_TPMS_GRID.bat           target E/Es = 0.10
REM      ID_3C_FINE_TPMS_GRID.bat 0.09      a different target
REM ===========================================================================
cd /d "%~dp0"

set "TGT=%~1"
if "%TGT%"=="" set "TGT=0.10"
set "ML_PROFILE=COMBINED"

echo.
echo ============================================================
echo   PHASE 2  TPMS  STEP 3 of 3 - GENERATES NEW GEOMETRY
echo   target E/Es = %TGT%
echo   about 12 nTop runs. Restartable.
echo ============================================================

if not exist "..\ML\Final Model\COMBINED\model_COMBINED.pt" echo [E] no exported model - run Share\ML\10_EXPORT_FORWARD_MODEL.bat first & pause & exit /b 1
if not exist "search\results\phase2_tpms_verify.json" echo [E] run ID_3B_FINE_TPMS_VERIFY.bat first - the chain has not been checked & pause & exit /b 1

pip install numpy pandas openpyxl trimesh torch scipy --quiet

python scripts\fine_tpms.py --grid --target %TGT%
if errorlevel 1 goto :err

echo.
echo ============================================================
echo   DONE.  search\results\
echo       phase2_tpms_E_over_Es_*.csv            every candidate
echo       phase2_tpms_E_over_Es_*_manifest.json  what produced them
echo   STLs in  search\candidates\tpms\
echo.
echo   Every geometry above is NEW. Its properties come from the
echo   model alone - Phase 4 runs ONE real FEA to test that.
echo ============================================================
pause
exit /b 0

:err
echo.
echo ***** Phase 2 TPMS failed - see above.
echo ***** Per-candidate nTop logs: search\runs\tpms\<candidate>\log.txt
pause
exit /b 1
