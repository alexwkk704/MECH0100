@echo off
REM ===========================================================================
REM  ID_3B_FINE_TPMS_VERIFY.bat  --  double-click, no arguments.  ~3 minutes.
REM
REM  STEP 2 OF 3 ON THE TPMS SIDE. Run ID_3A first. Run this before ID_3C.
REM
REM  A CLOSED-LOOP TEST OF THE WHOLE CHAIN
REM      Ask the calibration measured in ID_3A for the t that should give the
REM      volume fraction of a row that really exists - the Phase 1 TPMS winner -
REM      generate it, and measure what actually comes out. That exercises the
REM      notebook, the input-JSON shape, the inversion and the measurement in
REM      one go. Zhezhe made every training STL on HIS machine; nothing in this
REM      project has ever driven those notebooks from here.
REM
REM  THREE TARGETS, NOT ONE
REM      volume fraction   must land within 2 percent of the database
REM      wall thickness    must land within 5 percent of the database's
REM                        thickness_med_mm
REM      Euler number      reported, and loudly flagged if it differs
REM
REM      One target is not enough: the right volume fraction can be hit by the
REM      wrong geometry. The right volume fraction AND the right wall thickness
REM      AND the right Euler number cannot.
REM
REM      The thickness is measured with feature_extraction.wall_thickness - the
REM      SAME code that produced the dataset column - so the two are comparable
REM      by construction. The dataset used 4000 surface samples; 500 is used
REM      here because the median is stable well below that. Measured on
REM      Zhezhe's own FRD_0.26 STL: 250 / 500 / 1000 samples give
REM      0.5504 / 0.5514 / 0.5514 mm against the dataset's 0.5509 - inside
REM      0.1 percent - for seconds of work instead of a minute.
REM
REM  WHAT IT TOUCHES
REM      Writes ONLY under Share\Inverse design\search\. Share\TPMS is
REM      read-only. Nothing written here enters the training dataset.
REM
REM      ID_3B_FINE_TPMS_VERIFY.bat                     the Phase 1 winner
REM      ID_3B_FINE_TPMS_VERIFY.bat "IWP Generator_0.20"   a specific row
REM ===========================================================================
cd /d "%~dp0"

set "ROW=%~1"
set "ML_PROFILE=COMBINED"

echo.
echo ============================================================
echo   PHASE 2  TPMS  STEP 2 of 3 - closed-loop check
echo   Volume fraction, wall thickness and Euler number, against
echo   a row that really exists. ~3 minutes.
echo ============================================================

if not exist "..\ML\Final Model\COMBINED\model_COMBINED.pt" echo [E] no exported model - run Share\ML\10_EXPORT_FORWARD_MODEL.bat first & pause & exit /b 1

pip install numpy pandas openpyxl trimesh torch scipy --quiet

if "%ROW%"=="" goto :auto
python scripts\fine_tpms.py --verify --row "%ROW%"
goto :done

:auto
python scripts\fine_tpms.py --verify

:done
if errorlevel 1 goto :err

echo.
echo ============================================================
echo   VERIFY PASSED.
echo   Next: ID_3C_FINE_TPMS_GRID.bat   (12 new geometries)
echo ============================================================
pause
exit /b 0

:err
echo.
echo ***** VERIFY FAILED - do NOT run the grid yet.
echo *****
echo ***** Read which of the two gated targets missed:
echo *****   volume fraction off by more than 2 percent, or
echo *****   wall thickness off by more than 5 percent.
echo ***** Also read the Euler line - if it says DIFFERS, the geometry is
echo ***** not the same shape, whatever the numbers say.
echo *****
echo ***** If the calibration could not reach the wanted volume fraction,
echo ***** re-run ID_3A_FINE_TPMS_SCOUT.bat with a wider t range.
echo ***** The nTop log is in search\runs\tpms\ID2_VERIFY_*\log.txt
pause
exit /b 1
