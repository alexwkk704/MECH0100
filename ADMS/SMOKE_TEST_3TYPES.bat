@echo off
REM ============================================================
REM SMOKE_TEST_3TYPES.bat
REM
REM One nTop run per ADMS variant (DF, flow, raw) using whatever
REM row(s) are in runs_input.xlsx, then a completeness check.
REM
REM Purpose: prove the updated autoscript still collects EVERYTHING
REM the ML needs - C tensor, both stress fields, the STL, and the
REM native curvature table - BEFORE spending days expanding the
REM database.
REM
REM Nothing here is edited per type: --type picks the notebook by
REM discovery and --refresh-schema rebuilds each input schema from
REM the current .ntop. --rerun-all answers the duplicate prompt so
REM you can walk away.
REM ============================================================
cd /d "%~dp0"
echo Checking Python packages (numpy, openpyxl, matplotlib)...
python -m pip install --quiet numpy openpyxl matplotlib pillow
if errorlevel 1 (
    echo.
    echo [ERROR] Python or pip not found. Install Python 3.9+ from python.org
    echo         and tick "Add python.exe to PATH" during install.
    pause
    exit /b 1
)

for %%T in (DF flow raw) do (
    echo.
    echo ============================================================
    echo   SMOKE TEST - ADMS type %%T
    echo ============================================================
    python ntop_batch.py --type %%T --refresh-schema --rerun-all
    if errorlevel 1 (
        echo.
        echo [ERROR] the %%T run failed - stopping here.
        pause
        exit /b 1
    )
)

echo.
echo ============================================================
echo   CHECKING the three batches collected everything
echo ============================================================
python verify_batch.py --last 3
if errorlevel 1 (
    echo.
    echo [!] Something is missing - read the FAIL lines above.
    echo     Do NOT start expanding the database yet.
    pause
    exit /b 1
)

echo.
echo All three types produced a complete run. Safe to expand the database.
pause
