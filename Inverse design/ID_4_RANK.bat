@echo off
REM ===========================================================================
REM  ID_4_RANK.bat  --  double-click, no arguments.  Seconds. No nTop.
REM
REM  INVERSE DESIGN PHASE 3: rank everything, name the two finalists.
REM
REM  (Bat number 4, phase number 3 - the TPMS side needed three bats, so the
REM   numbering ran ahead of the phases. The banner always says the phase.)
REM
REM  THE SELECTION RULE
REM      The model cannot tell its own candidates apart. Measured on this
REM      project's own reproduction runs, its in-sample error was +2.27 percent
REM      on ADMS and -3.91 percent on TPMS, and the honest grouped-CV MAPE is
REM      5.7 to 6.6 percent. The Phase 2 candidates sit 0.4 to 4 percent from
REM      target. They are a TIE. Sorting them by distance to target and reading
REM      off the top is sorting noise.
REM
REM      So anything within the tie band counts as equal on stiffness, and the
REM      tie is broken by the second objective: LOWEST TAI.
REM
REM      TAI is the Total Anisotropy Index - the distance FROM isotropy. It is
REM      zero for a perfectly isotropic material. MORE ISOTROPIC = LOWER TAI.
REM      Ranking "highest TAI" would pick the most anisotropic design.
REM
REM      The band defaults to 6.6 percent, the upper end of the grouped-CV
REM      MAPE. Set it BEFORE looking at the answer and quote it in the write-up
REM      - choosing it afterwards is cherry-picking.
REM
REM          ID_4_RANK.bat            target 0.10, band 6.6 percent
REM          ID_4_RANK.bat 0.10 5.7   a tighter band
REM
REM  GATE 5
REM      Only the ID2_ geometries are eligible. Phase 1 rows are written to a
REM      separate context file and can never be selected - every one of them
REM      already has an FEA result, so picking one would mean the model had
REM      reproduced a number we already had rather than designed anything.
REM ===========================================================================
cd /d "%~dp0"

set "TGT=%~1"
set "BAND=%~2"
if "%TGT%"=="" set "TGT=0.10"
if "%BAND%"=="" set "BAND=6.6"
set "ML_PROFILE=COMBINED"

echo.
echo ============================================================
echo   PHASE 3  rank all 28 new geometries, name the finalists
echo   target %TGT%   tie band +/-%BAND% percent
echo ============================================================

if not exist "search\results\phase2_adms_E_over_Es_*.csv" echo [E] no ADMS Phase 2 results - run ID_2B_FINE_ADMS_GRID.bat first & pause & exit /b 1
if not exist "search\results\phase2_tpms_E_over_Es_*.csv" echo [E] no TPMS Phase 2 results - run ID_3C_FINE_TPMS_GRID.bat first & pause & exit /b 1

pip install numpy pandas openpyxl --quiet

python scripts\rank_designs.py --target %TGT% --tie-band %BAND%
if errorlevel 1 goto :err

echo.
echo ============================================================
echo   DONE.  search\results\
echo       phase3_ranking_*.csv       every candidate, ranked
echo       phase3_finalists_*.json    the two designs for the FEA
echo       phase3_context_existing_*.csv   database rows, CONTEXT ONLY
echo.
echo   Next: ID_5_VALIDATE_FEA.bat
echo ============================================================
pause
exit /b 0

:err
echo.
echo ***** Phase 3 failed - see above.
pause
exit /b 1
