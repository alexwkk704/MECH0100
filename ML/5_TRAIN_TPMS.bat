@echo off
REM ===========================================================================
REM  5_TRAIN_TPMS.bat -- TRAINING ONLY. No extraction.
REM
REM  NEEDS:  the dataset named in train_TPMS!dataset_csv  (2_SETUP_TPMS.bat)
REM  READS:  train_TPMS / pointnet_TPMS sheets of ML_settings.xlsx
REM  WRITES: runs\TPMS\<stamp>\
REM
REM  6 shared targets - the partner has no yield data
REM ===========================================================================
cd /d "%~dp0"
set "SDIR="
if exist "scripts\train.py" set "SDIR=scripts\"
set "ML_PROFILE=TPMS"

REM  Filenames are NOT hardcoded here: run_paths.py reads this profile's
REM  sheet in ML_settings.xlsx, so changing the sheet changes the guard too.
for /f "usebackq delims=" %%p in (`python %SDIR%run_paths.py dataset`) do set "DS=%%p"
if not exist "%DS%" echo [E] %DS% missing - run 2_SETUP_TPMS.bat & pause & exit /b 1

for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmm"`) do set "STAMP=%%i"
set "ML_RUN_DIR=%CD%\runs\TPMS\%STAMP%"
if not exist "%ML_RUN_DIR%\charts" mkdir "%ML_RUN_DIR%\charts"
echo Results to runs\TPMS\%STAMP%\

pip install numpy scipy pandas scikit-learn matplotlib openpyxl --quiet
pip install torch --quiet

echo. & echo ==== 1/8  RF / GPR / density baseline ====
python %SDIR%train.py
if errorlevel 1 goto :err
echo. & echo ==== 2/8  charts ====
python %SDIR%plot_results.py
if errorlevel 1 goto :err
REM collect_tensors is NOT ADMS-only: PointNet's rotation augmentation rotates the
REM point cloud AND the 6x6 tensor together, so the TPMS profile needs its own cache.
REM Resume-safe - re-running costs seconds if the cache is already complete.
echo. & echo ==== 3/8  refresh the stiffness-tensor cache ====
python %SDIR%collect_tensors.py
if errorlevel 1 goto :err
echo. & echo ==== 4/8  PointNet, grouped CV ====
python %SDIR%train_pointnet_v2.py
if errorlevel 1 goto :err
echo. & echo ==== 5/8  all methods, identical folds ====
python %SDIR%compare_models.py
if errorlevel 1 goto :err
echo. & echo ==== 6/8  PointNet, stratified 80/20 ====
python %SDIR%pointnet_split_8020.py
if errorlevel 1 goto :err
echo. & echo ==== 7/8  grouped CV vs 80/20 ====
python %SDIR%compare_cv_vs_8020.py
if errorlevel 1 goto :err

REM  Holdout R2 charts. The pipeline's own holdout charts are MAPE only, and
REM  MAPE has no zero point - it cannot show that a model is WORSE than simply
REM  predicting the mean, which on an unseen-family or extrapolation holdout is
REM  the whole question. R2 shows it. Reads what is already on disk, retrains
REM  nothing, takes seconds.
echo. & echo ==== 8/8  holdout R2 charts ====
python %SDIR%holdout_charts.py --run "%ML_RUN_DIR%"
if errorlevel 1 goto :err

copy /y "ML_settings.xlsx" "%ML_RUN_DIR%\ML_settings_used.xlsx" >nul 2>&1
copy /y "%DS%" "%ML_RUN_DIR%\" >nul 2>&1
> "%ML_RUN_DIR%\MANIFEST.txt" echo TPMS run %STAMP%  (profile TPMS)

echo. & echo ===========================================================
echo  DONE.  runs\TPMS\%STAMP%\
echo ===========================================================
pause
exit /b 0
:err
echo. & echo ***** FAILED at the step above.
pause
exit /b 1
