@echo off
REM ============================================================
REM RUN_CURV_VS_NTOP.bat
REM
REM Answers ONE question: does feature_extraction.py's curvature maths
REM read an exported STL the same way nTop reads the SAME STL?
REM
REM It compares against curvature_ntop.csv columns avg_Mn_K / avg_Mn_H,
REM which are nTop's reading of the exported STL (Curvature_Check with
REM Max Evenly Spaced Points on the imported Mesh + Offset Body(Box,-0.1)).
REM
REM READ-ONLY. Touches no STL, no notebook, no FEA result, and NOT
REM features_ADMS.csv. Writes one file: curv_vs_ntop.csv
REM
REM The settings below MUST match the nTop notebook or the numbers are
REM not comparable:
REM     --offset      0.1   = Offset Body Distance
REM     --spacing     0.3   = Point spacing (mm)
REM     --cell-size   3.0   = Cell Size input AND cell_size_mm in ML_settings
REM ============================================================
cd /d "%~dp0"
echo Installing/checking Python packages (first run only)...
pip install trimesh scipy numpy --quiet
pip install manifold3d --quiet
echo.
echo   [1] SELF-TEST only  - sphere, torus, cut sphere vs analytic answers
echo   [2] RUN  - the 11 pilot geometries
echo   [3] RUN  - every STL in ADMS_STL  (11 pilot domain settings)
echo.
echo   [4] RUN  - every STL, WHOLE SURFACE, no box filter
echo       Use this when the nTop notebook has no Filter Points by Volume.
echo.
echo   [5] RUN  - ONLY the geometries nTop has already done  (RECOMMENDED)
echo       Reads curvature_ntop_stl.csv and checks exactly those rows.
echo       Use after a --limit test run, before committing to all 131.
echo.
set /p MODE="Choose [1/2/3/4/5]: "
if "%MODE%"=="1" python curv_vs_ntop.py --selftest
if "%MODE%"=="2" python curv_vs_ntop.py --offset 0.1 --spacing 0.3 --cell-size 3.0 --ntop-csv curvature_ntop_stl.csv
if "%MODE%"=="3" python curv_vs_ntop.py --all --offset 0.1 --spacing 0.3 --cell-size 3.0 --ntop-csv curvature_ntop_stl.csv
if "%MODE%"=="4" python curv_vs_ntop.py --all --no-filter --spacing 0.3 --cell-size 3.0 --ntop-csv curvature_ntop_stl.csv
if "%MODE%"=="5" python curv_vs_ntop.py --match-ntop --offset 0.1 --spacing 0.3 --cell-size 3.0 --ntop-csv curvature_ntop_stl.csv
echo.
pause
