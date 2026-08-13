@echo off
REM ===========================================================================
REM  8_POINTNET_MISSING_HOLDOUTS.bat
REM
REM  BACKFILLS the four PointNet holdouts that were disabled when the COMBINED
REM  run was produced, straight INTO that run folder:
REM      unseen_variant_flow, extrap_low_density,
REM      extrap_high_density, extrap_high_v2
REM  RF and GPR already have all eight; only PointNet was missing these.
REM
REM  USAGE
REM      8_POINTNET_MISSING_HOLDOUTS.bat
REM          backfills the newest runs\COMBINED\2026* folder that has holdouts
REM      8_POINTNET_MISSING_HOLDOUTS.bat runs\COMBINED\20260811_0800
REM          backfills that folder
REM
REM  WHY A SCRATCH FOLDER IS USED IN THE MIDDLE
REM    train_pointnet_v2.py OVERWRITES pointnet_holdouts.csv with whatever is
REM    enabled for the run. Pointing it straight at the target would DELETE the
REM    four holdouts already there. So the new four are trained into
REM    runs\COMBINED\_staging_holdouts_<stamp>, then merged back into the target
REM    and the target's R2 charts are redrawn. The scratch folder is left behind
REM    as provenance; it is small and safe to delete.
REM
REM  COST  ~5-6 h. ML_HOLDOUTS_ONLY=1 skips the grouped-CV pass, which is the
REM        expensive half and is already done, and holdouts 1-4 are switched off
REM        so the 5.7 h already spent on them is not repeated.
REM
REM  SAFETY  the target's pointnet_holdouts.csv is copied to .bak_prebackfill
REM          before anything is merged. ML_settings.xlsx is backed up and the
REM          holdout keys are restored at the end, including on failure.
REM ===========================================================================
cd /d "%~dp0"
set "SDIR="
if exist "scripts\train_pointnet_v2.py" set "SDIR=scripts\"
set "ML_PROFILE=COMBINED"
set "ML_HOLDOUTS_ONLY=1"

REM --- which run are we backfilling ----------------------------------------
set "TARGET=%~1"
if "%TARGET%"=="" for /f "delims=" %%d in ('dir /b /ad /o-d "runs\COMBINED\2026*" 2^>nul') do if not defined TARGET if exist "runs\COMBINED\%%d\pointnet_holdouts.csv" set "TARGET=%CD%\runs\COMBINED\%%d"
if not defined TARGET echo [E] no run under runs\COMBINED has pointnet_holdouts.csv - pass the folder as an argument & pause & exit /b 1
if not exist "%TARGET%\pointnet_holdouts.csv" echo [E] %TARGET% has no pointnet_holdouts.csv & pause & exit /b 1

for /f "usebackq delims=" %%p in (`python %SDIR%run_paths.py dataset`) do set "DS=%%p"
for /f "usebackq delims=" %%p in (`python %SDIR%run_paths.py tensors`) do set "TZ=%%p"
echo.
echo   backfilling  %TARGET%
echo   dataset      %DS%
echo   tensors      %TZ%
echo.
if not exist "%DS%" echo [E] %DS% missing & pause & exit /b 1
if not exist "%TZ%" echo [E] %TZ% missing & pause & exit /b 1

copy /y "%TARGET%\pointnet_holdouts.csv" "%TARGET%\pointnet_holdouts.csv.bak_prebackfill" >nul
echo [backup] %TARGET%\pointnet_holdouts.csv.bak_prebackfill

if not exist "_backups" mkdir "_backups"
for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmm"`) do set "STAMP=%%i"
copy /y "ML_settings.xlsx" "_backups\ML_settings_premissingholdouts_%STAMP%.xlsx" >nul
echo [backup] _backups\ML_settings_premissingholdouts_%STAMP%.xlsx

pip install numpy scipy pandas scikit-learn matplotlib openpyxl torch --quiet

REM  switch OFF the four already done, so only 5-8 run this time
python %SDIR%set_pointnet_key.py --profile COMBINED holdout_1= holdout_2= holdout_3= holdout_4=
if errorlevel 1 goto :err

set "STAGE=%CD%\runs\COMBINED\_staging_holdouts_%STAMP%"
if not exist "%STAGE%" mkdir "%STAGE%"
set "ML_RUN_DIR=%STAGE%"
> "%STAGE%\MANIFEST.txt" echo scratch - PointNet holdouts 5-8 only, merged into %TARGET%

echo.
echo ============================================================
echo   PointNet - the four missing holdouts, no grouped CV
echo ============================================================
python %SDIR%train_pointnet_v2.py
if errorlevel 1 goto :err

call :restore
if errorlevel 1 goto :err

echo.
echo ==== merge back into the run folder ====
python %SDIR%merge_holdouts.py --into "%TARGET%" --from "%STAGE%"
if errorlevel 1 goto :err

echo.
echo ==== redraw that run's holdout R2 charts, all eight ====
python %SDIR%holdout_charts.py --run "%TARGET%"
if errorlevel 1 goto :err

echo.
echo ============================================================
echo   DONE.  %TARGET%\charts\
echo   Scratch folder kept: %STAGE%
echo   Next: RERUN_POINTNET_SEEDS.bat COMBINED
echo ============================================================
pause
exit /b 0

:restore
python %SDIR%set_pointnet_key.py --profile COMBINED "holdout_1=unseen_family_TPMS|family|TPMS" "holdout_2=unseen_family_ADMS|family|ADMS" "holdout_3=unseen_gyroid|topology|Gyroid Generator" "holdout_4=unseen_P|topology|P Generator"
exit /b %errorlevel%

:err
echo.
echo ***** FAILED. Restoring the holdout keys.
call :restore
pause
exit /b 1
