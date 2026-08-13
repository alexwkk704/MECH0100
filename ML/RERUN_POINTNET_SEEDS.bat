@echo off
REM ===========================================================================
REM  RERUN_POINTNET_SEEDS.bat  --  measure PointNet run-to-run variance.
REM
REM  USAGE
REM      RERUN_POINTNET_SEEDS.bat            same as ADMS
REM      RERUN_POINTNET_SEEDS.bat ADMS
REM      RERUN_POINTNET_SEEDS.bat COMBINED
REM      RERUN_POINTNET_SEEDS.bat TPMS
REM
REM  WHY THIS EXISTS
REM    The 07/08 ADMS run scored E/Es 0.694 on grouped CV where every other run
REM    scored 0.965-0.981. One fold did it: folds 0-3 were 0.872/0.992/0.987/
REM    0.991 and fold 4 was -0.325. Fold 4 is the hardest BY CONSTRUCTION - it
REM    is the only fold that must predict the densest geometry, VF 0.5622.
REM    A neural-network run is ONE DRAW: weight init, shuffling, augmentation
REM    and best-epoch selection all vary. One run was never "the" answer.
REM
REM  WHY COMBINED NEEDS IT TOO
REM    Merging TPMS does not add anything above VF 0.30, so the top of the
REM    range is exactly as isolated as before - but the share of rows at
REM    VF >= 0.38 falls from 29.4% (48/163) to 13.9% (48/346). The high-density
REM    region is HALF as represented in every training fold and validation
REM    slice, which is the condition the fold-4 diagnosis blamed.
REM    And ADMS already has four seeds: comparing a four-sample mean against a
REM    single COMBINED draw is not a comparison.
REM
REM  WHAT IT DOES
REM    Re-runs the grouped-CV PointNet at seeds 2, 3 and 4 for the chosen
REM    profile. With the existing seed-1 run that gives four samples.
REM    Holdouts are OFF - they are the slow part and not what is being measured.
REM    Every path comes from ML_settings.xlsx via run_paths.py; nothing here
REM    hardcodes a dataset filename.
REM
REM  WHAT IT DOES NOT DO
REM    It does not touch any existing run folder. Each seed gets its own.
REM    The settings file is backed up first and restored at the end.
REM
REM  TIME  roughly 2.5 h per seed for ADMS 163 rows; COMBINED has 346 rows so
REM        allow about 5 h per seed, i.e. 15 h for three. Leave it overnight.
REM ===========================================================================
cd /d "%~dp0"
set "SDIR="
if exist "scripts\train_pointnet_v2.py" set "SDIR=scripts\"

set "PROF=%~1"
if "%PROF%"=="" set "PROF=ADMS"
set "ML_PROFILE=%PROF%"
set "ML_HOLDOUTS_ONLY=0"

for /f "usebackq delims=" %%p in (`python %SDIR%run_paths.py dataset`) do set "DS=%%p"
for /f "usebackq delims=" %%p in (`python %SDIR%run_paths.py tensors`) do set "TZ=%%p"
echo.
echo   profile  %PROF%
echo   dataset  %DS%
echo   tensors  %TZ%
echo.
if not exist "%DS%" echo [E] %DS% missing & pause & exit /b 1
if not exist "%TZ%" echo [E] %TZ% missing - run collect_tensors.py first & pause & exit /b 1

if not exist "_backups" mkdir "_backups"
for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmm"`) do set "STAMP=%%i"
copy /y "ML_settings.xlsx" "_backups\ML_settings_preseedsweep_%PROF%_%STAMP%.xlsx" >nul
echo [backup] _backups\ML_settings_preseedsweep_%PROF%_%STAMP%.xlsx

pip install numpy scipy pandas scikit-learn matplotlib openpyxl torch --quiet

python %SDIR%set_pointnet_key.py --profile %PROF% run_holdouts=0
if errorlevel 1 goto :err

for %%S in (2 3 4) do call :one %%S
goto :done

:one
set "SEED=%1"
echo.
echo ============================================================
echo   %PROF%  SEED %SEED%  - grouped 5-fold CV only, no holdouts
echo ============================================================
python %SDIR%set_pointnet_key.py --profile %PROF% random_seed=%SEED%
if errorlevel 1 goto :err

set "ML_RUN_DIR=%CD%\runs\%PROF%\%STAMP%_seed%SEED%"
if not exist "%ML_RUN_DIR%\charts" mkdir "%ML_RUN_DIR%\charts"
copy /y "%DS%" "%ML_RUN_DIR%\" >nul
> "%ML_RUN_DIR%\MANIFEST.txt" echo PointNet seed sweep, profile %PROF%, seed %SEED%, grouped CV only, holdouts disabled.

python %SDIR%train_pointnet_v2.py
if errorlevel 1 goto :err
echo   [ok] seed %SEED% written to runs\%PROF%\%STAMP%_seed%SEED%
exit /b 0

:done
python %SDIR%set_pointnet_key.py --profile %PROF% random_seed=1 run_holdouts=1

echo.
echo ============================================================
echo   DONE.  Compare the samples:
echo.
echo     python scripts\seed_spread.py
echo.
echo   Folders under runs\%PROF%\ :
echo     %STAMP%_seed2
echo     %STAMP%_seed3
echo     %STAMP%_seed4
echo   plus your existing seed-1 run.
echo ============================================================
pause
exit /b 0

:err
echo.
echo ***** FAILED. Restoring the settings file.
python %SDIR%set_pointnet_key.py --profile %PROF% random_seed=1 run_holdouts=1
pause
exit /b 1
