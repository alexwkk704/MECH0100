@echo off
REM ===========================================================================
REM  6_VALIDATE_MODEL.bat -- HOW WELL DOES THE MODEL GENERALISE?
REM
REM  Writes ONLY to  Share\ML\Validation\<PROFILE>_<stamp>\
REM  It never touches Share\ML\runs\ , so a validation run cannot disturb the
REM  headline results in runs\ADMS / runs\TPMS / runs\COMBINED.
REM
REM  WHY THIS EXISTS
REM    Grouped CV does NOT test topology generalisation. group_key is
REM    family|topology|density-band, so every topology appears in EVERY fold
REM    (measured 04/08: 18 of 18 TPMS topologies in all 5 folds). CV therefore
REM    answers "can it interpolate inside a topology it has seen", which is a
REM    much easier question than "can it reach a topology it has never seen".
REM
REM  WHAT IT RUNS -- every holdout_i on the train_<PROFILE> / pointnet_<PROFILE>
REM  sheets, now including EXTRAPOLATION holdouts (VF below / above the sampled
REM  range) as well as leave-one-topology-out:
REM    train.py              RF / GPR / density baseline on every holdout
REM    train_pointnet_v2.py  PointNet on the SAME holdouts (new 04/08 - PointNet
REM                          had never been holdout-tested at all)
REM
REM  READING THE OUTPUT: a NEGATIVE R2 means worse than predicting the mean,
REM  i.e. the model does not reach that region. That is a real result, not a bug.
REM ===========================================================================
cd /d "%~dp0"
set "SDIR="
if exist "scripts\train.py" set "SDIR=scripts\"

echo.
echo   Which model?   [1] ADMS   [2] TPMS   [3] COMBINED
set /p WHICH="Choose [1/2/3]: "
if "%WHICH%"=="2" (set "ML_PROFILE=TPMS") else if "%WHICH%"=="3" (set "ML_PROFILE=COMBINED") else (set "ML_PROFILE=ADMS")

if not exist "data\dataset_%ML_PROFILE%.csv" echo [E] data\dataset_%ML_PROFILE%.csv missing - run the setup bat first & pause & exit /b 1

echo.
echo   [1] HOLDOUTS ONLY   skip PointNet's grouped-CV pass. Much faster, and the
echo                       CV number is already in runs\%ML_PROFILE%\.
echo   [2] FULL            re-run CV as well, so CV and holdouts sit side by side.
set /p MODE="Choose [1/2]: "
if "%MODE%"=="2" (set "ML_HOLDOUTS_ONLY=0") else (set "ML_HOLDOUTS_ONLY=1")

for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmm"`) do set "STAMP=%%i"
set "ML_RUN_DIR=%CD%\Validation\%ML_PROFILE%_%STAMP%"
if not exist "%ML_RUN_DIR%\charts" mkdir "%ML_RUN_DIR%\charts"
echo.
echo Profile %ML_PROFILE%   holdouts_only=%ML_HOLDOUTS_ONLY%
echo Results to Validation\%ML_PROFILE%_%STAMP%\

pip install numpy scipy pandas scikit-learn matplotlib openpyxl --quiet
pip install torch --quiet

echo. & echo ==== 1/2  RF / GPR / density on every holdout ====
python %SDIR%train.py
if errorlevel 1 goto :err

echo. & echo ==== 2/2  PointNet on the SAME holdouts ====
python %SDIR%train_pointnet_v2.py
if errorlevel 1 goto :err

copy /y "ML_settings.xlsx" "%ML_RUN_DIR%\ML_settings_used.xlsx" >nul 2>&1
> "%ML_RUN_DIR%\MANIFEST.txt" echo VALIDATION run %STAMP%  profile %ML_PROFILE%  holdouts_only=%ML_HOLDOUTS_ONLY%
>> "%ML_RUN_DIR%\MANIFEST.txt" echo Generalisation test only. Headline CV numbers live in runs\%ML_PROFILE%\.
>> "%ML_RUN_DIR%\MANIFEST.txt" echo Negative R2 = worse than predicting the mean = model does not reach that region.

echo. & echo ===========================================================
echo  DONE.  Validation\%ML_PROFILE%_%STAMP%\
echo    model_results.csv      RF / GPR / density, per holdout
echo    pointnet_holdouts.csv  PointNet, same holdouts
echo ===========================================================
pause
exit /b 0
:err
echo. & echo ***** FAILED at the step above.
pause
exit /b 1
