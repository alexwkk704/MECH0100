@echo off
REM ===========================================================================
REM  ID_5_VALIDATE_FEA.bat  --  double-click, no arguments.
REM
REM  INVERSE DESIGN PHASE 4: one real FEA per family. GATE 7.
REM
REM  THIS IS THE ONLY NUMBER IN THE WHOLE INVERSE DESIGN THAT IS NOT A
REM  PREDICTION. Everything up to here came from the model. This runs a real
REM  homogenisation on the two designs Phase 3 named and asks whether the model
REM  was right.
REM
REM      GATE 7   ^|measured - predicted^| / measured  ^<=  15 percent
REM
REM  Report the answer whichever way it comes out. A GATE 7 failure is a
REM  finding, not an embarrassment - it is the honest measurement of what a
REM  forward model trained on 346 geometries can do when asked to design one.
REM
REM  WHAT IT RUNS
REM      ADMS   Share\ADMS\ADMS_Raw_only_FEA.ntop
REM             Density / Thickness / Seed / Inner Size / Size Multi, the same
REM             parameters as the STL run, so it rebuilds the SAME geometry and
REM             homogenises it.  Es = 200 GPa, nu_s = 0.3.
REM      TPMS   Share\TPMS\TPMS FEA\FEA-P1\FRD Generator_density0.20.ntop
REM             t_max / t_min / Edge length.  Es = 1.8 GPa, tensor in kPa.
REM
REM      Every metric is computed by ntop_batch's OWN voigt_reuss_hill and
REM      tensorial_anisotropy_index, imported not copied, so the comparison is
REM      like-for-like with the training labels. Material constants come from
REM      ML_settings.xlsx, never from this file.
REM
REM  THE TPMS MESH SIZE - MEASURED, NOT ASSUMED
REM      Edge length is the FE mesh size: in the notebook it drives both
REM      Feature size and Edge length of the FE Volume Mesh. It is EDGE_C x t.
REM
REM      EDGE_C = 0.4, read off FRD Generator_density0.20.ntop, which -s saved
REM      after a real run:  t_max 0.7672, Edge length 0.3069, ratio 0.4000.
REM
REM      tpms_batch_run has EDGE_C = 2. That is a DIFFERENT model's value and
REM      it is wrong for FRD - at 2 the mesh would be 5 elements across a 10 mm
REM      cell and 0.28 through a 0.55 mm wall, i.e. the wall unresolved, and
REM      the TPMS GATE 7 would be meaningless. Only override this with
REM      evidence as good as a saved notebook:
REM          ID_5_VALIDATE_FEA.bat both 0.4
REM
REM  ONE-WAY RULE
REM      Everything lands in Share\Inverse design\validation\. These rows TEST
REM      the model, so they must NEVER be merged into the training dataset -
REM      a row the model was tested on cannot then be trained on. Nothing here
REM      writes to Share\ADMS, Share\TPMS or Results_summary.xlsx.
REM
REM          ID_5_VALIDATE_FEA.bat            both families
REM          ID_5_VALIDATE_FEA.bat adms       one family
REM          ID_5_VALIDATE_FEA.bat both 2.0   set the TPMS Edge length constant
REM ===========================================================================
cd /d "%~dp0"

set "FAM=%~1"
set "EDGEC=%~2"
set "REUSE=%~3"
if "%FAM%"=="" set "FAM=both"
if "%EDGEC%"=="" set "EDGEC=0.4"
set "ML_PROFILE=COMBINED"

echo.
echo ============================================================
echo   PHASE 4  one real FEA per family   GATE 7 = 15 percent
echo   families: %FAM%    TPMS Edge length = %EDGEC% x t
echo ============================================================

if not exist "search\results\phase3_finalists_E_over_Es_*.json" echo [E] no Phase 3 finalists - run ID_4_RANK.bat first & pause & exit /b 1

pip install numpy pandas openpyxl scipy --quiet

set "EXTRA="
if /i "%REUSE%"=="reuse" set "EXTRA=--from-existing"
python scripts\validate_fea.py --family %FAM% --edge-c %EDGEC% %EXTRA%
if errorlevel 1 goto :err

echo.
echo ============================================================
echo   DONE.  validation\phase4_gate7_*.json
echo   Per-run FEA output and logs in validation\adms\ and
echo   validation\tpms\.
echo.
echo   THE INVERSE DESIGN IS COMPLETE.
echo ============================================================
pause
exit /b 0

:err
echo.
echo ***** Phase 4 failed - see above.
echo ***** Per-run nTop logs: validation\<family>\<design>\log.txt
pause
exit /b 1
