@echo off
REM Re-render master charts from current data + chart_config sheet.
REM Use after editing labels on the 'chart_config' sheet in Results_summary.xlsx.
cd /d "%~dp0"
echo Checking Python packages (numpy, openpyxl, matplotlib)...
python -m pip install --quiet numpy openpyxl matplotlib pillow
if errorlevel 1 (
    echo.
    echo [ERROR] Python or pip not found. Install Python 3.9+ from python.org.
    pause
    exit /b 1
)
python refresh_charts.py
echo.
pause
