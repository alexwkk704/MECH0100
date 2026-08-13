@echo off
REM ===========================================================================
REM  ID_2A_FINE_ADMS_VERIFY.bat  --  double-click, no arguments.  ~2 minutes.
REM
REM  RUN THIS BEFORE THE GRID.
REM
REM  The six trimmed notebooks (ADMS_*_only_STL / only_FEA) are new. NOT ONE row
REM  in the training database was produced by them - every ADMS row came from the
REM  generic_v1 notebooks. So before spending half an hour generating sixteen
REM  geometries with a notebook we have never checked, this regenerates ONE row
REM  that already exists and compares what comes out:
REM
REM      p_rel   from the notebook   vs   Rho_rel in the database
REM      VF      measured off the STL vs  VF in the database
REM
REM  Both must agree to 2 percent. If the trimming quietly changed the geometry,
REM  this is where it shows - in two minutes, not after sixteen generations and
REM  a wrong design carried into Phase 4.
REM
REM  It also prints the model's prediction against the row's real FEA number.
REM  That comparison is IN-SAMPLE - the model trained on this row - so it is a
REM  sanity check, never an accuracy figure. The honest error is the grouped-CV
REM  MAPE from the results chapter.
REM
REM  WHAT IT TOUCHES
REM      Reads   Share\ADMS\ADMS_<Variant>_only_STL.ntop   (copied, never run in place)
REM              Share\ML\Final Model\COMBINED\model_COMBINED.pt
REM              Share\ML\data\dataset_COMBINED_quad.csv
REM      Writes  ONLY under Share\Inverse design\search\
REM      Share\ADMS, ntop_batch.py, input_template_*.json, runs_input*.xlsx and
REM      Results_summary.xlsx are read-only to this. Nothing it writes ever
REM      enters the training dataset.
REM
REM      ID_2A_FINE_ADMS_VERIFY.bat                  the Phase 1 ADMS winner
REM      ID_2A_FINE_ADMS_VERIFY.bat <Run stem>       a specific row
REM ===========================================================================
cd /d "%~dp0"

set "ROW=%~1"
set "ML_PROFILE=COMBINED"

echo.
echo ============================================================
echo   PHASE 2A  reproduction check
echo   Regenerate ONE existing row with the trimmed notebook
echo   and compare it to the database. ~2 minutes.
echo ============================================================

if not exist "..\ML\Final Model\COMBINED\model_COMBINED.pt" echo [E] no exported model - run Share\ML\10_EXPORT_FORWARD_MODEL.bat first & pause & exit /b 1
if not exist "..\ADMS\ADMS_Raw_only_STL.ntop" echo [E] cannot find ..\ADMS\ADMS_Raw_only_STL.ntop & pause & exit /b 1

pip install numpy pandas openpyxl trimesh torch --quiet

if "%ROW%"=="" goto :auto
python scripts\fine_adms.py --verify --row "%ROW%"
goto :done

:auto
python scripts\fine_adms.py --verify

:done
if errorlevel 1 goto :err

echo.
echo ============================================================
echo   VERIFY PASSED. The trimmed notebook reproduces the
echo   database geometry.
echo.
echo   Next: ID_2B_FINE_ADMS_GRID.bat   (16 new geometries, ~30 min)
echo ============================================================
pause
exit /b 0

:err
echo.
echo ***** VERIFY FAILED - do NOT run the grid yet.
echo *****
echo ***** Read the numbers above. If p_rel or VF is more than 2 percent
echo ***** off the database, the trimmed notebook is not building the same
echo ***** geometry as the one the model was trained on, and every candidate
echo ***** the grid produces would inherit that difference.
echo *****
echo ***** The nTop log is in search\runs\adms\ID2_VERIFY_*\log.txt
pause
exit /b 1
