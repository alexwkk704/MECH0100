@echo off
REM RECONCILE_MASTER.bat - make sure every completed run on disk is in the master.
REM Safe to run any time. Shows what is missing, asks before writing, backs up first.
REM Use this after ANY interrupted batch (Ctrl+C, SDK freeze, crash, power loss).
cd /d "%~dp0"
python -m pip install --quiet numpy openpyxl matplotlib pillow
if errorlevel 1 (
    echo.
    echo [ERROR] Python or pip not found. Install Python 3.9+ from python.org
    echo         and tick "Add python.exe to PATH" during install.
    pause
    exit /b 1
)
echo.
echo === STEP 1: report what is missing (nothing is written yet) ===
python RECONCILE_MASTER.py --dry-run
echo.
echo === STEP 2: apply (you will be asked to confirm) ===
python RECONCILE_MASTER.py
echo.
pause
