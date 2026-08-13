@echo off
REM ===========================================================================
REM  2_SETUP_TPMS.bat -- EXTRACTION ONLY. No training.
REM
REM  Every path, cell size, unit and material constant lives in ML_settings.xlsx:
REM      sheet settings_TPMS   stl_folder, cell_size_mm, thickness_convention
REM      sheet tpms_labels     tensor_root, tensor_unit, es, es_unit, nu,
REM                            element_order, output_csv
REM  Nothing below hardcodes a dataset filename - it is read back from the
REM  sheet via run_paths.py so the two can never drift apart.
REM  This file sets a profile name and nothing else.
REM
REM  TRUST BOUNDARY: only his STL files and his RAW 6x6 tensors are used. His
REM  feature/label CSVs are never read - his TAI uses a different formula
REM  (median 70%% off ours), so every label is recomputed with our tensor_ops.
REM ===========================================================================
cd /d "%~dp0"
set "SDIR="
if exist "scripts\feature_extraction.py" set "SDIR=scripts\"
set "ML_PROFILE=TPMS"

echo Installing/checking packages...
pip install trimesh numpy scipy pandas scikit-learn openpyxl --quiet
echo.
echo   [1] FULL   extract all STLs, point clouds, dataset + tensor cache
echo   [2] TEST   3 files only - check the numbers look sane first
echo.
set /p MODE="Choose [1/2]: "

if "%MODE%"=="2" (
  python %SDIR%feature_extraction.py --output-csv features_TPMS_TEST.csv --test 3
  goto :done
)

echo.
echo ==== 1/4  geometric features from the partner STLs ====
python %SDIR%feature_extraction.py
if errorlevel 1 goto :err

echo.
echo ==== 2/4  point clouds for PointNet  (resume-safe) ====
python %SDIR%pointcloud_prep.py
if errorlevel 1 goto :err

echo.
echo ==== 3/4  labels from his RAW 6x6 tensors, our formulas ====
python %SDIR%build_tpms_dataset.py
if errorlevel 1 goto :err

REM  PointNet's rotation augmentation rotates the point cloud AND the stiffness
REM  tensor together, so it needs the raw 6x6 for every geometry. Cached as C/Es
REM  (dimensionless) so steel and polymer can live in one file.
echo.
echo ==== 4/4  cache the 6x6 stiffness tensors for rotation augmentation ====
python %SDIR%collect_tensors.py
if errorlevel 1 goto :err

:done
echo.
echo ===========================================================
for /f "usebackq delims=" %%p in (`python %SDIR%run_paths.py dataset`) do set "DS=%%p"
for /f "usebackq delims=" %%p in (`python %SDIR%run_paths.py tensors`) do set "TZ=%%p"
echo  DONE.  %DS%
echo         %TZ%
echo    next:  3_RUN_ADMS_TPMS.bat   (merge + train the combined model)
echo       or: 5_TRAIN_TPMS.bat      (train TPMS on its own)
echo ===========================================================
pause
exit /b 0
:err
echo. & echo ***** FAILED at the step above. Steps 1-2 are resume-safe - delete
echo       data\features_TPMS.csv only for a clean restart.
pause
exit /b 1
