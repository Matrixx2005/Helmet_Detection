@echo off
rem Danh gia day du tren Construction-PPE: YOLO (mAP) + he luat (cap nguoi).
rem Cach dung: run_evaluation.bat [duong_dan_construction-ppe.zip]
call "%~dp0_env.bat" || exit /b 1
set "ZIP=%~1"
if "%ZIP%"=="" set "ZIP=%USERPROFILE%\Downloads\construction-ppe.zip"
set "DS=data\construction-ppe"
set "LOG=%ROOT%\logs\evaluation.log"
set M="%PY%" -m helmet_ppe
echo === evaluation %date% %time% > "%LOG%"

set "MODELS=models\best.pt"
if exist models\best_v2.pt set "MODELS=models\best.pt models\best_v2.pt"

call :title "1. Giai nen dataset"
%M% prepare-dataset --zip "%ZIP%" --dest %DS% >> "%LOG%" 2>&1
call :rc
call :title "2. Danh gia YOLO tren val va test"
%M% eval-detector --dataset %DS% --models %MODELS% --splits val test >> "%LOG%" 2>&1
call :rc
call :title "3. Nhan chuan cap nguoi (val, test)"
%M% make-person-gt --dataset %DS% --split val --output data\person_gt\val_person_gt.csv >> "%LOG%" 2>&1
%M% make-person-gt --dataset %DS% --split test --output data\person_gt\test_person_gt.csv >> "%LOG%" 2>&1
call :rc

for %%W in (%MODELS%) do (
    call :title "4. Do nguong he luat tren val - %%~nW"
    %M% batch --model %%W --conf 0.05 --input %DS%\images\val --output outputs\ppe_val_lowconf_%%~nW --no-images >> "%LOG%" 2>&1
    %M% tune-rules --run outputs\ppe_val_lowconf_%%~nW --gt data\person_gt\val_person_gt.csv --output reports\rule_tuning\%%~nW >> "%LOG%" 2>&1
    call :rc
    call :title "5. Batch + danh gia he luat tren test - %%~nW"
    %M% batch --model %%W --input %DS%\images\test --output outputs\ppe_test_%%~nW >> "%LOG%" 2>&1
    %M% eval-rules --run outputs\ppe_test_%%~nW --gt data\person_gt\test_person_gt.csv --output reports\rule_eval\%%~nW >> "%LOG%" 2>&1
    call :rc
)
echo [XONG] Log: %LOG%
exit /b 0

:title
echo.>> "%LOG%"
echo ===== %~1 ===== >> "%LOG%"
echo %~1 ...
exit /b 0

:rc
echo     exit code = %errorlevel%>> "%LOG%"
echo     exit code = %errorlevel%
exit /b 0
