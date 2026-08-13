@echo off
REM ===========================================================================
REM  10_EXPORT_FORWARD_MODEL.bat  --  double-click, no arguments.
REM
REM  JOB 10: the LAST step of the forward ML pipeline. Steps 1-9 build the
REM  database and measure how well the model generalises. This step takes the
REM  same validated recipe, trains on EVERY row, and freezes the result into a
REM  self-contained bundle.
REM
REM  Inverse design (Share\Inverse design) is a CONSUMER of what this writes.
REM  It is a separate namespace. Nothing here belongs to it and nothing here
REM  imports it.
REM
REM  WRITES  ->  Final Model\<PROFILE>\
REM      model_<PROFILE>.pt      weights + targets + normalisation + prep spec
REM      model_card.txt          what was trained, in plain English
REM      gate0_reference.csv     in-sample predictions for every training row
REM      GATE0_roundtrip.csv     the gate's own evidence
REM
REM  ENSEMBLE  ML_ENSEMBLE below sets how many seeds are trained and AVERAGED.
REM      Training is stochastic - weight init, batch order, jitter, rotation
REM      draws and dropout all move with the seed. The FOLDS do not; cv_split
REM      greedy-balances groups with no RNG at all. Averaging several seeds
REM      reduces that variance. Running a sweep and shipping the best-scoring
REM      seed would be selection on validation data, so nothing here offers it.
REM      1 = fastest, roughly 1-1.5 h.   4 = the agreed setting, overnight.
REM
REM  GATE 0 runs automatically afterwards and this .bat STOPS if it fails.
REM      It reloads 10 rows from their ORIGINAL STL and pushes them back through
REM      predict.py. Proven to catch a wrong centre (25 %% error) and a 5 %%
REM      scale slip (41 %% error) - tested by deliberately breaking the prep.
REM      Expect ~1e-7. Anything near 1 %% means something is wrong.
REM ===========================================================================
cd /d "%~dp0"

set "PROF=%~1"
if "%PROF%"=="" set "PROF=COMBINED"
set "ML_PROFILE=%PROF%"
set "ML_ENSEMBLE=4"

set "SDIR="
if exist "scripts\export_forward_model.py" set "SDIR=scripts\"

echo.
echo ============================================================
echo   JOB 10  EXPORT FORWARD MODEL   profile %PROF%
echo   seeds averaged: %ML_ENSEMBLE%
echo ============================================================

for /f "usebackq delims=" %%p in (`python %SDIR%run_paths.py dataset`) do set "DS=%%p"
for /f "usebackq delims=" %%p in (`python %SDIR%run_paths.py tensors`) do set "TZ=%%p"
echo   dataset  %DS%
echo   tensors  %TZ%
if not exist "%DS%" echo [E] dataset missing: %DS% & pause & exit /b 1
if not exist "%TZ%" echo [E] tensors missing: %TZ% - run collect_tensors.py first & pause & exit /b 1

pip install numpy pandas openpyxl trimesh torch --quiet

echo.
echo ---- training on all rows --------------------------------
python %SDIR%export_forward_model.py
if errorlevel 1 goto :err

echo.
echo ---- GATE 0 round-trip -----------------------------------
python %SDIR%gate0_check.py --n 10 --tol 0.01
if errorlevel 1 goto :gate

echo.
echo ============================================================
echo   DONE.  Final Model\%PROF%\
echo       model_%PROF%.pt        the frozen forward model
echo       model_card.txt         read this before quoting numbers
echo       GATE0_roundtrip.csv    the gate's evidence
echo.
echo   Inverse design can now load it:
echo       python scripts\predict.py --self-test
echo ============================================================
pause
exit /b 0

:gate
echo.
echo ***** GATE 0 FAILED.
echo ***** predict.py does not reproduce the training predictions, so the
echo ***** exported model must NOT be used for inverse design yet.
echo ***** Look at Final Model\%PROF%\GATE0_roundtrip.csv for the per-row errors.
pause
exit /b 1

:err
echo.
echo ***** the export failed - see the messages above.
pause
exit /b 1
