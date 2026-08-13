@echo off
REM ===========================================================================
REM  7_HOLDOUT_R2_CHARTS.bat  --  backfill the holdout R2 charts.
REM
REM  The charts the pipeline writes for the holdouts show MAPE only. MAPE has no
REM  zero point, so it cannot show that a model is WORSE than predicting the
REM  mean - which on an unseen-family or extrapolation holdout is the whole
REM  question. R2 shows it, because it goes negative.
REM
REM  Nothing is retrained. This reads model_results.csv and, where present,
REM  pointnet_holdouts.csv, both already written by the pipeline.
REM
REM  WRITES, into each run's own charts\ folder:
REM      holdoutR2_<split>.png      one chart per holdout split
REM      holdoutR2_table.csv        the numbers behind them
REM  Nothing else in the run folder is touched. Existing MAPE charts stay.
REM
REM  USAGE
REM      7_HOLDOUT_R2_CHARTS.bat              COMBINED and ADMS  (default)
REM      7_HOLDOUT_R2_CHARTS.bat COMBINED
REM      7_HOLDOUT_R2_CHARTS.bat ADMS
REM      7_HOLDOUT_R2_CHARTS.bat TPMS
REM
REM  Runs in seconds. Safe to run while something else is training.
REM ===========================================================================
cd /d "%~dp0"
set "SDIR="
if exist "scripts\holdout_charts.py" set "SDIR=scripts\"

set "PROFS=%~1"
if "%PROFS%"=="" set "PROFS=COMBINED ADMS"

pip install numpy pandas matplotlib --quiet

for %%P in (%PROFS%) do call :one %%P
goto :done

:one
echo.
echo ============================================================
echo   holdout R2 charts - profile %1
echo ============================================================
python %SDIR%holdout_charts.py --profile %1
if errorlevel 1 goto :err
exit /b 0

:done
echo.
echo ============================================================
echo   DONE. Look in each run folder:
echo       runs\^<PROFILE^>\^<stamp^>\charts\holdoutR2_*.png
echo ============================================================
pause
exit /b 0

:err
echo.
echo ***** FAILED - see the message above.
pause
exit /b 1
