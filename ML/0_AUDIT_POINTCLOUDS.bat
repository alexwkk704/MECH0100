@echo off
REM ===========================================================================
REM  0_AUDIT_POINTCLOUDS.bat  --  double-click, no arguments. WRITES NOTHING.
REM
REM  Reports which stored point clouds no longer match the CURRENT
REM  pointcloud sheet (n_points AND random_seed). Run it before deciding
REM  whether to regenerate.
REM
REM  WHY IT EXISTS
REM      pointcloud!random_seed was changed 42 -> 1 between 23 and 31 July, and
REM      the old staleness check only compared n_points. Every cloud that already
REM      existed silently kept its seed-42 points. GATE 0 found it on 12/08: two
REM      of ten replayed rows came back 39-48 %% different, the rest exact to 1e-7.
REM
REM  "seed UNRECORDED" is expected for every cloud built before 12/08 - the seed
REM  was not stored in the file until then.
REM
REM  --probe then PROVES it: for a few of those clouds it re-samples the STL at
REM  each candidate seed and reports which one reproduces the stored points
REM  exactly. "=> stored cloud was built with seed 42" is the confirmation.
REM  Bounded to a few files because loading a 100-500 MB ADMS STL is the slow part.
REM
REM  TO FIX: run 1_SETUP_ADMS.bat and 2_SETUP_TPMS.bat. They rebuild only the
REM  stale clouds. Then re-run 10_EXPORT_FORWARD_MODEL.bat, because the exported
REM  model was trained on the old ones.
REM ===========================================================================
cd /d "%~dp0"
set "SDIR="
if exist "scripts\pointcloud_prep.py" set "SDIR=scripts\"

echo.
echo ============================================================
echo   POINT CLOUD AUDIT - read only, nothing is written
echo ============================================================

echo.
echo ---- ADMS -------------------------------------------------
set "ML_PROFILE=ADMS"
python %SDIR%pointcloud_prep.py --probe 3
if errorlevel 1 goto :err

echo.
echo ---- TPMS -------------------------------------------------
set "ML_PROFILE=TPMS"
python %SDIR%pointcloud_prep.py --probe 2
if errorlevel 1 goto :err

echo.
echo ============================================================
echo   Nothing was changed. See the STALE counts above.
echo ============================================================
pause
exit /b 0

:err
echo.
echo ***** the audit failed - see above.
pause
exit /b 1
