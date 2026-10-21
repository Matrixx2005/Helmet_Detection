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
    call :title "4. Do nguong tren val (YOLO conf 0.02 + do luoi offline) - %%~nW"
    %M% batch --model %%W --conf-person 0.02 --conf-helmet 0.02 --conf-no-helmet 0.02 --conf-pose-person 0.02 --input %DS%\images\val --output outputs\ppe_val_lowconf_%%~nW --no-images >> "%LOG%" 2>&1
    %M% tune-rules --use-pose on --run outputs\ppe_val_lowconf_%%~nW --gt data\person_gt\val_person_gt.csv --output reports\rule_tuning\%%~nW >> "%LOG%" 2>&1
    call :rc
    call :title "5. Batch + danh gia he luat tren test voi nguong da do - %%~nW"
    %M% batch --model %%W --params reports\rule_tuning\%%~nW\best_params.json --input %DS%\images\test --output outputs\ppe_test_%%~nW >> "%LOG%" 2>&1
    %M% eval-rules --run outputs\ppe_test_%%~nW --gt data\person_gt\test_person_gt.csv --output reports\rule_eval\%%~nW >> "%LOG%" 2>&1
    call :rc
)
call :title "5b. best.pt khong dung pose (do nguong rieng tren val, danh gia tren test)"
%M% tune-rules --use-pose off --run outputs\ppe_val_lowconf_best --gt data\person_gt\val_person_gt.csv --output reports\rule_tuning\best_nopose >> "%LOG%" 2>&1
%M% batch --model models\best.pt --params reports\rule_tuning\best_nopose\best_params.json --input %DS%\images\test --output outputs\ppe_test_best_nopose >> "%LOG%" 2>&1
%M% eval-rules --run outputs\ppe_test_best_nopose --gt data\person_gt\test_person_gt.csv --output reports\rule_eval\best_nopose >> "%LOG%" 2>&1
call :rc
call :title "6. Doi chung: luat ngay tho (khong thay mu = vi pham) tren test"
%M% eval-rules --baseline --run outputs\ppe_test_best --gt data\person_gt\test_person_gt.csv --output reports\rule_eval\baseline_naive >> "%LOG%" 2>&1
call :rc
call :title "7. Do nhay nguong IoU ghep nguoi (0.3)"
%M% eval-rules --iou 0.3 --run outputs\ppe_test_best --gt data\person_gt\test_person_gt.csv --output reports\rule_eval\best_iou03 >> "%LOG%" 2>&1
call :rc
call :title "8. Cau hinh ban dau de so sanh: nguong chung 0.25, chua do nguong"
%M% batch --model models\best.pt --no-pose --conf-person 0.25 --conf-helmet 0.25 --conf-no-helmet 0.25 --min-helmet-evidence 0.25 --min-no-helmet-evidence 0.25 --margin 0.1 --input %DS%\images\test --output outputs\ppe_test_best_untuned --no-images >> "%LOG%" 2>&1
%M% eval-rules --run outputs\ppe_test_best_untuned --gt data\person_gt\test_person_gt.csv --output reports\rule_eval\best_untuned >> "%LOG%" 2>&1
call :rc
call :title "9. Bang so sanh cac cau hinh tren test"
%M% compare-rules reports\rule_eval\best_untuned reports\rule_eval\baseline_naive reports\rule_eval\best_v2 reports\rule_eval\best_nopose reports\rule_eval\best --output reports\rule_eval\comparison >> "%LOG%" 2>&1
call :rc
echo [XONG] Log: %LOG%
exit /b 0

:title
echo.>> "%LOG%"
echo ===== %~1 ===== >> "%LOG%"
echo %~1 ...
exit /b 0

:rc
>> "%LOG%" echo     exit code = %errorlevel%
echo     exit code = %errorlevel%
exit /b 0
