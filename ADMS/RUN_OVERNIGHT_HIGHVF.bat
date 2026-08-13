@echo off
REM ============================================================
REM RUN_OVERNIGHT_HIGHVF.bat   -- start it and go to sleep
REM
REM 44 runs: the VF 0.38-0.50 band, sampled at 2-3 different
REM (Density, Thickness) routes per volume-fraction node, so
REM volume fraction and morphology are DECOUPLED. That is the
REM point: at fixed VF the measured spread is 10-49% in E* and
REM 38-168% in TAI, and a model fed one point per VF can only
REM ever relearn the density power law.
REM
REM Order: DF (30 tok) -> raw (32) -> flow (130). Cheap first,
REM so if something goes wrong at 4am the expensive half is the
REM part still unspent.
REM
REM Does NOT stop on a failed run. ntop_batch marks the row FAIL
REM and carries on; this file carries on between types too.
REM ============================================================
cd /d "%~dp0"
echo Checking Python packages...
python -m pip install --quiet numpy openpyxl matplotlib pillow

echo.
echo ============================================================
echo   1/3  DF   15 runs  (~30 tokens)
echo ============================================================
python ntop_batch.py --type DF   --input runs_input_df_HIGH.xlsx   --refresh-schema --rerun-all

echo.
echo ============================================================
echo   2/3  raw  16 runs  (~32 tokens)
echo ============================================================
python ntop_batch.py --type raw  --input runs_input_raw_HIGH.xlsx  --refresh-schema --rerun-all

echo.
echo ============================================================
echo   3/3  flow 13 runs  (~130 tokens)
echo ============================================================
python ntop_batch.py --type flow --input runs_input_flow_HIGH.xlsx --refresh-schema --rerun-all

echo.
echo ============================================================
echo   CHECKING all three batches
echo ============================================================
python verify_batch.py --last 3

echo.
echo Overnight batch finished. Read the FAIL lines above, if any.
pause
