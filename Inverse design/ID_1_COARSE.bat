@echo off
REM ===========================================================================
REM  ID_1_COARSE.bat  --  double-click, no arguments.
REM
REM  INVERSE DESIGN PHASE 1: score every geometry already in the database with
REM  the frozen forward model and rank by distance to the target.
REM  Free, seconds, generates no geometry and spends no Spherene tokens.
REM
REM      ID_1_COARSE.bat            target E/Es = 0.10
REM      ID_1_COARSE.bat 0.09       a different target
REM
REM  WHAT IT IS AND IS NOT
REM      Phase 1 LOCALISES the answer, it does not produce it. Every row it can
REM      return already exists and already has an FEA result, so a Phase 1
REM      winner proves nothing by itself - GATE 5 requires the design carried
REM      forward to be NEW geometry from Phase 2. What this buys is the
REM      neighbourhood to build the Phase 2 grid around, for free.
REM      Its predictions are IN-SAMPLE, so the agreement is an upper bound and
REM      never an accuracy figure.
REM
REM  FOLDER RULE
REM      Forward ML stays in Share\ML. This is a consumer: it imports predict.py
REM      from there rather than keeping a copy, so the prediction path cannot
REM      drift from the one GATE 0 verified. Nothing here writes into Share\ML.
REM
REM  GATES
REM      GATE 1  target must sit where BOTH families have data
REM      GATE 4  reject rows whose PREDICTED value leaves the training range
REM      plus a seed check: every stored cloud must carry the sampling seed the
REM      model expects. That is the check whose absence cost a day on 12/08.
REM
REM  REQUIRES  Share\ML\Final Model\COMBINED\model_COMBINED.pt with GATE 0 passed.
REM ===========================================================================
cd /d "%~dp0"

set "TGT=%~1"
if "%TGT%"=="" set "TGT=0.10"
set "ML_PROFILE=COMBINED"

echo.
echo ============================================================
echo   INVERSE DESIGN  PHASE 1  coarse search
echo   target E/Es = %TGT%
echo ============================================================

if not exist "..\ML\Final Model\COMBINED\model_COMBINED.pt" echo [E] no exported model - run Share\ML\10_EXPORT_FORWARD_MODEL.bat first & pause & exit /b 1

pip install numpy pandas openpyxl trimesh torch --quiet

python scripts\coarse_search.py --target %TGT% --objective E_over_Es --top 10
if errorlevel 1 goto :err

echo.
echo ============================================================
echo   DONE.  search\results\
echo       phase1_ranked_*.csv     every row, ranked
echo       phase1_summary_*.txt    the three winners + the caveats
echo.
echo   Next: Phase 2 builds a small grid of NEW geometries around
echo   those winners. It needs the nTop input schemas first.
echo ============================================================
pause
exit /b 0

:err
echo.
echo ***** Phase 1 failed - see above.
echo ***** If it stopped on a seed mismatch, run Share\ML\0_AUDIT_POINTCLOUDS.bat.
pause
exit /b 1
