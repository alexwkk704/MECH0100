@echo off
REM ===========================================================================
REM  9_SEED_SWEEP_COMBINED.bat  --  double-click, no arguments.
REM
REM  Runs the COMBINED PointNet at seeds 2, 3 and 4 so every PointNet number
REM  gets an error bar. With the existing seed-1 run that is four samples.
REM
REM  WHY THIS IS NOT OPTIONAL
REM    A neural-net run is ONE DRAW. On ADMS the spread across four seeds was
REM    0.29 R2 on stiffness and 0.77 on TAI, and one run in four had a fold fail
REM    to train outright. Every current COMBINED number - 0.970 on the ADMS rows,
REM    -53.97 on unseen_family_ADMS, 0.934 on unseen_gyroid - is a single draw.
REM
REM  TIME  roughly 5 h per seed at 346 rows, so about 15 h. Leave it overnight.
REM        Holdouts are OFF during the sweep; CV variance is what is measured.
REM
REM  AFTER  it prints the comparison command, and the charts land in
REM         runs\COMBINED\_seed_summary\
REM ===========================================================================
cd /d "%~dp0"
call "%~dp0RERUN_POINTNET_SEEDS.bat" COMBINED
if errorlevel 1 exit /b 1

echo.
echo ============================================================
echo   seed sweep finished - building the spread charts
echo ============================================================
set "SDIR="
if exist "scripts\seed_spread.py" set "SDIR=scripts\"
python %SDIR%seed_spread.py --profile COMBINED
if errorlevel 1 goto :err

echo.
echo ============================================================
echo   DONE.  runs\COMBINED\_seed_summary\
echo       seed_spread_by_target.png    one bar per seed
echo       methods_seedmean.png         all methods, PointNet averaged
echo       seed_spread_summary.csv      mean / sd / min / max / range
echo ============================================================
pause
exit /b 0

:err
echo.
echo ***** the sweep finished but the charts failed - see above.
pause
exit /b 1
