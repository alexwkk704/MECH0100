@echo off
REM ===========================================================================
REM  1_SETUP_ADMS.bat -- EXTRACTION ONLY. No training. No PointNet.
REM
REM  STL -> features -> point clouds -> dataset_ADMS.csv
REM  Run this whenever the ADMS STLs or Results_summary.xlsx change.
REM  Takes ~30 min. Training is a separate file (4_TRAIN_ADMS.bat).
REM ===========================================================================
cd /d "%~dp0"
set "SDIR="
if exist "scripts\feature_extraction.py" set "SDIR=scripts\"
set "ML_PROFILE=ADMS"
REM  all paths + parameters come from ML_settings.xlsx sheet settings_ADMS

echo Installing/checking packages...
pip install trimesh numpy scipy pandas scikit-learn openpyxl --quiet

echo.
echo ==== 1/3  geometric features from the ADMS STLs  (resume-safe) ====
python %SDIR%feature_extraction.py
if errorlevel 1 goto :err

echo.
echo ==== 2/3  point clouds for PointNet  (resume-safe, slowest) ====
python %SDIR%pointcloud_prep.py
if errorlevel 1 goto :err

echo.
echo ==== 3/3  join features to the FEA labels ====
python %SDIR%labels_join.py
if errorlevel 1 goto :err

echo.
echo ==== GATE  validate before anyone trains on this ====
python %SDIR%validate_dataset.py
if errorlevel 1 goto :badrows

echo.
echo ===========================================================
echo  DONE.  data\dataset_ADMS.csv is ready.
echo    next:  4_TRAIN_ADMS.bat      (train the ADMS-only model)
echo       or: 2_SETUP_TPMS.bat      (build the TPMS side)
echo ===========================================================
pause
exit /b 0
:err
echo.
echo ***** FAILED at the step above. Steps 1-2 resume where they stopped.
pause
exit /b 1
:badrows
echo.
echo ***** STOPPED: dataset failed validation. Nothing was trained, on purpose.
pause
exit /b 1
