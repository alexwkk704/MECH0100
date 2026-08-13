@echo off
REM ADMS lattice batch runner - double-click to run.
REM Reads runs_input.xlsx, runs nTop for each row, writes Results\Results_summary.xlsx
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
python ntop_batch.py
echo.
echo Done. Results are in the "Results" folder next to this file.
pause
