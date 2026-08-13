@echo off
REM ===========================================================================
REM  10a_EXPORT_SMOKE_1SEED.bat  --  double-click, no arguments.
REM
REM  RUN THIS FIRST. It is 10_EXPORT_FORWARD_MODEL.bat with ONE seed instead of
REM  four: roughly a quarter of the time, and it exercises every single code
REM  path the full run uses - same data, same 346 rows, same 2048 points, same
REM  600 epochs, same rotation augmentation, same GATE 0.
REM
REM  WHY BOTHER
REM      The export scripts were validated end-to-end on a synthetic rig
REM      (16 meshes, 256 points, 6 epochs). The code paths are identical but the
REM      scale is not. If something breaks at real scale - memory, a stem that
REM      does not match, a trimesh version difference tripping GATE 0 - this
REM      finds it in about a quarter of the time instead of wasting the night.
REM
REM  WHAT TO CHECK, in this order
REM      1. "matched N/346 rows to clouds"  -> must say 346. Anything else, stop.
REM      2. "rotation_aug: matched N/346 tensors" -> must not be 0.
REM      3. GATE 0 must say PASS, worst error around 1e-7.
REM         A number near 1 %% is inside tolerance but means something moved -
REM         most likely a different trimesh build than the one that made the
REM         stored point clouds. Tell me before running the 4-seed version.
REM
REM  It writes to the SAME place as the full run:
REM      Final Model\COMBINED\
REM  so the 4-seed run simply overwrites it afterwards. Nothing to clean up.
REM ===========================================================================
cd /d "%~dp0"

set "PROF=%~1"
if "%PROF%"=="" set "PROF=COMBINED"
set "ML_PROFILE=%PROF%"
set "ML_ENSEMBLE=1"

set "SDIR="
if exist "scripts\export_forward_model.py" set "SDIR=scripts\"

echo.
echo ============================================================
echo   SMOKE RUN  --  ONE seed, profile %PROF%
echo   this is the cheap dry run before the 4-seed export
echo ============================================================

for /f "usebackq delims=" %%p in (`python %SDIR%run_paths.py dataset`) do set "DS=%%p"
for /f "usebackq delims=" %%p in (`python %SDIR%run_paths.py tensors`) do set "TZ=%%p"
echo   dataset  %DS%
echo   tensors  %TZ%
if not exist "%DS%" echo [E] dataset missing: %DS% & pause & exit /b 1
if not exist "%TZ%" echo [E] tensors missing: %TZ% - run collect_tensors.py first & pause & exit /b 1

pip install numpy pandas openpyxl trimesh torch --quiet

echo.
echo ---- training, 1 seed ------------------------------------
python %SDIR%export_forward_model.py
if errorlevel 1 goto :err

echo.
echo ---- GATE 0 round-trip -----------------------------------
python %SDIR%gate0_check.py --n 10 --tol 0.01
if errorlevel 1 goto :gate

echo.
echo ============================================================
echo   SMOKE RUN PASSED.
echo.
echo   Check the three things above, then run
echo       10_EXPORT_FORWARD_MODEL.bat
echo   for the real 4-seed ensemble.
echo ============================================================
pause
exit /b 0

:gate
echo.
echo ***** GATE 0 FAILED on the smoke run.
echo ***** Do NOT start the 4-seed run. Send me
echo *****   Final Model\%PROF%\GATE0_roundtrip.csv
echo ***** and I will find out which step of the prep moved.
pause
exit /b 1

:err
echo.
echo ***** the smoke run failed - see the messages above.
pause
exit /b 1
