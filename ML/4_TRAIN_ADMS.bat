@echo off
REM ===========================================================================
REM  4_TRAIN_ADMS.bat -- TRAINING ONLY. No extraction.
REM
REM  NEEDS:  the dataset named in train_ADMS!dataset_csv  (1_SETUP_ADMS.bat)
REM  READS:  train_ADMS / pointnet_ADMS sheets of ML_settings.xlsx
REM  WRITES: runs\ADMS\<stamp>\
REM
REM  8 targets, incl. onset_n / shear_onset_n
REM ===========================================================================
cd /d "%~dp0"
set "SDIR="
if exist "scripts\train.py" set "SDIR=scripts\"
set "ML_PROFILE=ADMS"

REM  Filenames are NOT hardcoded here: run_paths.py reads this profile's
REM  sheet in ML_settings.xlsx, so changing the sheet changes the guard too.
for /f "usebackq delims=" %%p in (`python %SDIR%run_paths.py dataset`) do set "DS=%%p"
if not exist "%DS%" echo [E] %DS% missing - run 1_SETUP_ADMS.bat & pause & exit /b 1

for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmm"`) do set "STAMP=%%i"
set "ML_RUN_DIR=%CD%\runs\ADMS\%STAMP%"
if not exist "%ML_RUN_DIR%\charts" mkdir "%ML_RUN_DIR%\charts"
echo Results to runs\ADMS\%STAMP%\

pip install numpy scipy pandas scikit-learn matplotlib openpyxl --quiet
pip install torch --quiet

echo. & echo ==== 1/8  RF / GPR / density baseline ====
python %SDIR%train.py
if errorlevel 1 goto :err
echo. & echo ==== 2/8  charts ====
python %SDIR%plot_results.py
if errorlevel 1 goto :err
REM  PointNet's rotation augmentation rotates the point cloud AND the 6x6 tensor
REM  together, so it needs data\tensors_ADMS.npz. This used to be an unlabelled,
REM  UNGUARDED line: if it failed the .bat carried on and PointNet trained with a
REM  stale or missing cache, i.e. rotation augmentation silently did nothing.
REM  That is the exact failure that cost a 7-hour COMBINED run on 2026-08-03.
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
> "%ML_RUN_DIR%\MANIFEST.txt" echo ADMS run %STAMP%  (profile ADMS)

echo. & echo ===========================================================
echo  DONE.  runs\ADMS\%STAMP%\
echo ===========================================================
pause
exit /b 0
:err
echo. & echo ***** FAILED at the step above.
pause
exit /b 1
