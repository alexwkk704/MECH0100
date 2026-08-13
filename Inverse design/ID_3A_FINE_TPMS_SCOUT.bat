@echo off
REM ===========================================================================
REM  ID_3A_FINE_TPMS_SCOUT.bat  --  double-click, no arguments.  ~10 minutes.
REM
REM  STEP 1 OF 3 ON THE TPMS SIDE. Run this before ID_3B.
REM
REM  WHAT PROBLEM THIS SOLVES
REM      The TPMS notebook takes t_max / t_min, not a density. To build a
REM      geometry at a wanted density we need to know which t gives it.
REM      That mapping is recorded NOWHERE we can use:
REM        - t_input is EMPTY in every TPMS dataset (0 of 179 and 0 of 183)
REM        - the only density,t,vf_actual lookup under MECH0100 is
REM          Zhezhe\week7\IWP Generator.csv, which is IWP, not FRD
REM        - merged_training_data.csv is the OLD LINEAR-era file and is NOT
REM          READ BY ANY SCRIPT HERE. Not one cell of it.
REM
REM  SO IT IS MEASURED INSTEAD.
REM      4 nTop runs at t = 0.30, 0.60, 0.90, 1.30. Each STL's volume fraction
REM      is measured, and the curve is written to
REM          search\results\tpms_<topology>_calibration.csv
REM      ID_3B and ID_3C invert THAT. Every number in the chain is produced by
REM      the notebook we are actually driving, tonight, and is on disk.
REM
REM      The 4 runs are not overhead: each is a real geometry with a measured
REM      volume fraction and a model prediction, so they are candidates too.
REM
REM  IF THE RANGE MISSES
REM      The scout prints whether the centre row's volume fraction falls inside
REM      what it covered. If not, re-run with a wider range:
REM          ID_3A_FINE_TPMS_SCOUT.bat 0.2 2.0
REM      It also refuses later if the measured curve is not monotonic, because
REM      inverting an ambiguous curve would silently pick the wrong t.
REM
REM  WHAT IT TOUCHES
REM      Reads   Share\TPMS\TPMS STL\ntopfile+script\ntopfile\<Type>.ntop
REM              Share\ML\Final Model\COMBINED\model_COMBINED.pt
REM              Share\ML\data\dataset_COMBINED_quad.csv
REM      Writes  ONLY under Share\Inverse design\search\
REM      Share\TPMS is read-only: the notebook is copied into search\ntop\ and
REM      driven from there. Nothing written here enters the training dataset.
REM
REM      ID_3A_FINE_TPMS_SCOUT.bat              t from 0.30 to 1.30
REM      ID_3A_FINE_TPMS_SCOUT.bat 0.2 2.0      a wider range
REM ===========================================================================
cd /d "%~dp0"

set "TLO=%~1"
set "THI=%~2"
if "%TLO%"=="" set "TLO=0.30"
if "%THI%"=="" set "THI=1.30"
set "ML_PROFILE=COMBINED"

echo.
echo ============================================================
echo   PHASE 2  TPMS  STEP 1 of 3 - MEASURE the calibration
echo   t from %TLO% to %THI%, 4 runs. Roughly 10 minutes.
echo ============================================================

if not exist "..\ML\Final Model\COMBINED\model_COMBINED.pt" echo [E] no exported model - run Share\ML\10_EXPORT_FORWARD_MODEL.bat first & pause & exit /b 1
if not exist "..\TPMS\TPMS STL\ntopfile+script\ntopfile" echo [E] cannot find the TPMS STL notebooks & pause & exit /b 1

pip install numpy pandas openpyxl trimesh torch scipy --quiet

python scripts\fine_tpms.py --scout --t-lo %TLO% --t-hi %THI%
if errorlevel 1 goto :err

echo.
echo ============================================================
echo   CALIBRATION MEASURED.
echo       search\results\tpms_*_calibration.csv
echo.
echo   Check the printout above: the centre row's volume fraction
echo   must fall INSIDE the range covered. If it does not, re-run
echo   with a wider range, e.g.  ID_3A_FINE_TPMS_SCOUT.bat 0.2 2.0
echo.
echo   Next: ID_3B_FINE_TPMS_VERIFY.bat
echo ============================================================
pause
exit /b 0

:err
echo.
echo ***** Scout failed - see above.
echo ***** Per-run nTop logs: search\runs\tpms\ID2_SCOUT_*\log.txt
pause
exit /b 1
