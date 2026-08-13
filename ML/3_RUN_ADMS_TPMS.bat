@echo off
REM ===========================================================================
REM  3_RUN_ADMS_TPMS.bat -- merge the two datasets, then train the COMBINED model
REM
REM  NEEDS:  the datasets named in train_ADMS!dataset_csv (1_SETUP_ADMS.bat)
REM          and train_TPMS!dataset_csv (2_SETUP_TPMS.bat).
REM          Output name comes from train_COMBINED!dataset_csv.
REM
REM  Reads the train_COMBINED / pointnet_COMBINED sheets of ML_settings.xlsx.
REM  Results go to runs\COMBINED\<stamp>\ so they never mix with the other models.
REM
REM  READ THIS BEFORE QUOTING THE R2:
REM    several features identify which dataset a row came from on their own
REM    (fabric_DA is 1.000-1.003 for TPMS and 1.055-1.632 for ADMS, no overlap,
REM    because TPMS unit cells are cubic-symmetric and ADMS is aperiodic).
REM    So grouped-CV R2 is NOT evidence that merging helped. The headline number
REM    is the holdout rows in train_COMBINED: unseen family and unseen topology.
REM ===========================================================================
cd /d "%~dp0"
set "SDIR="
if exist "scripts\merge_datasets.py" set "SDIR=scripts\"
REM  Filenames are NOT hardcoded: run_paths.py reads dataset_csv from the
REM  train_<PROFILE> sheet of ML_settings.xlsx. Change the sheet and every
REM  guard, the merge inputs and the run snapshot follow automatically.
set "ML_PROFILE=ADMS"
for /f "usebackq delims=" %%p in (`python %SDIR%run_paths.py dataset`) do set "DS_A=%%p"
set "ML_PROFILE=TPMS"
for /f "usebackq delims=" %%p in (`python %SDIR%run_paths.py dataset`) do set "DS_T=%%p"
set "ML_PROFILE=COMBINED"
for /f "usebackq delims=" %%p in (`python %SDIR%run_paths.py dataset`) do set "DS_C=%%p"
echo   ADMS     %DS_A%
echo   TPMS     %DS_T%
echo   COMBINED %DS_C%

if not exist "%DS_A%" echo [E] %DS_A% missing - run 1_SETUP_ADMS.bat & pause & exit /b 1
if not exist "%DS_T%" echo [E] %DS_T% missing - run 2_SETUP_TPMS.bat & pause & exit /b 1

REM  The new run folder is created AFTER the menu, not before. Creating it up
REM  front made option [3] pick the empty folder it had just made as the "most
REM  recent" one. (Bug hit 2026-08-03 22:54.)

pip install trimesh numpy scipy pandas scikit-learn matplotlib openpyxl --quiet
pip install torch --quiet

echo.
echo   [1] FULL      everything, steps 1-9, into a NEW run folder
echo   [2] FROM 4    skip merge/RF/GPR/charts. Re-runs the grouped-CV PointNet
echo                 (steps 4-9) into a NEW run folder. This is the LONG one.
echo   [3] FROM 7    only the 80/20 split + comparisons (steps 7-9), written
echo                 back into the MOST RECENT run folder. Use this when the
echo                 grouped-CV PointNet already finished and only the tail failed.
echo.
set /p MODE="Choose [1/2/3]: "
if "%MODE%"=="3" goto :stage7
if "%MODE%"=="2" goto :stage4
set "NEXT=full"
goto :newrun

:newrun
for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmm"`) do set "STAMP=%%i"
set "ML_RUN_DIR=%CD%\runs\COMBINED\%STAMP%"
if not exist "%ML_RUN_DIR%\charts" mkdir "%ML_RUN_DIR%\charts"
echo Results to runs\COMBINED\%STAMP%\
goto :%NEXT%

:stage4
if not exist "%DS_C%" echo [E] %DS_C% missing - choose 1 & pause & exit /b 1
set "NEXT=pointnet"
goto :newrun

:stage7
REM  Reuse the newest existing run folder - step 8 compares the 80/20 result
REM  against pointnet_v2_results.csv, which lives IN that folder. Making a new
REM  stamp here would throw away the grouped-CV run we are trying to finish.
REM  Pick the newest folder that ACTUALLY HAS a finished grouped-CV PointNet,
REM  not merely the newest folder - an aborted or empty run must be skipped.
set "LASTRUN="
for /f "delims=" %%d in ('dir /b /ad /o-d "runs\COMBINED" 2^>nul') do (
  if not defined LASTRUN if exist "runs\COMBINED\%%d\pointnet_v2_results.csv" set "LASTRUN=%%d"
)
if not defined LASTRUN echo [E] no folder under runs\COMBINED contains pointnet_v2_results.csv - the grouped-CV PointNet has never finished. Choose 2. & pause & exit /b 1
set "ML_RUN_DIR=%CD%\runs\COMBINED\%LASTRUN%"
set "STAMP=%LASTRUN%"
if not exist "%ML_RUN_DIR%\charts" mkdir "%ML_RUN_DIR%\charts"
echo Resuming into runs\COMBINED\%LASTRUN%
goto :split8020

:full

echo.
echo ==== 1/10  merge ADMS + TPMS ====
python %SDIR%merge_datasets.py
if errorlevel 1 goto :err

echo.
echo ==== GATE  validate the combined dataset ====
python %SDIR%validate_dataset.py
if errorlevel 1 goto :badrows

echo.
echo ==== 2/10  RF / GPR / density baseline ====
python %SDIR%train.py
if errorlevel 1 goto :err
echo.
echo ==== 3/10  charts ====
python %SDIR%plot_results.py
if errorlevel 1 goto :err
:pointnet
REM  PointNet's rotation augmentation rotates the point cloud AND the 6x6 stiffness
REM  tensor together (Farooq's spec), so it needs the tensor cache named in pointnet_COMBINED!tensor_npz. The
REM  tensors are cached as C/Es - dimensionless - which is the only reason steel
REM  (ADMS, 200 GPa) and polymer (TPMS, 1.8 GPa) can share one file. Resume-safe.
echo.
echo ==== 4/10  cache the 6x6 stiffness tensors for rotation augmentation ====
python %SDIR%collect_tensors.py
if errorlevel 1 goto :err

echo.
echo ==== 5/10  PointNet, grouped CV ====
python %SDIR%train_pointnet_v2.py
if errorlevel 1 goto :err
echo.
echo ==== 6/10  all methods, identical folds ====
python %SDIR%compare_models.py
if errorlevel 1 goto :err
:split8020
echo.
echo ==== 7/10  PointNet, stratified 80/20 ====
REM  Stratifies on the settings key `stratify_col` (pointnet_COMBINED sheet).
REM  It must be a column that SURVIVES the merge: adms_type is ADMS-only and is
REM  dropped, so COMBINED stratifies on the unified `topology`.
python %SDIR%pointnet_split_8020.py
if errorlevel 1 goto :err
echo.
echo ==== 8/10  grouped CV vs 80/20 ====
python %SDIR%compare_cv_vs_8020.py
if errorlevel 1 goto :err

echo.
REM  Holdout R2 charts. The pipeline's own holdout charts are MAPE only, and
REM  MAPE has no zero point - it cannot show that a model is WORSE than simply
REM  predicting the mean, which on an unseen-family or extrapolation holdout is
REM  the whole question. R2 shows it. Reads what is already on disk, retrains
REM  nothing, takes seconds.
echo.
echo ==== 9/10  holdout R2 charts ====
python %SDIR%holdout_charts.py --run "%ML_RUN_DIR%"
if errorlevel 1 goto :err

echo.
echo ==== 10/10  snapshot settings + dataset ====
copy /y "ML_settings.xlsx" "%ML_RUN_DIR%\ML_settings_used.xlsx" >nul 2>&1
copy /y "%DS_C%" "%ML_RUN_DIR%\" >nul 2>&1
> "%ML_RUN_DIR%\MANIFEST.txt" echo COMBINED run %STAMP%
>> "%ML_RUN_DIR%\MANIFEST.txt" echo profile: COMBINED  (train_COMBINED / pointnet_COMBINED)
>> "%ML_RUN_DIR%\MANIFEST.txt" echo HEADLINE = the holdout rows, NOT the grouped-CV R2.

echo.
echo ===========================================================
echo  DONE.  runs\COMBINED\%STAMP%\
echo    compare against runs\ADMS\ on the SAME held-out ADMS rows
echo    - that is the test of whether merging actually helped.
echo ===========================================================
pause
exit /b 0
:err
echo. & echo ***** FAILED at the step above.
pause
exit /b 1
:badrows
echo. & echo ***** STOPPED: combined dataset failed validation. Nothing trained.
pause
exit /b 1
