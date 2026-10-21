@echo off
rem Chay toan bo quy trinh kiem tra cuoi cung, ghi log vao logs\.
call "%~dp0_env.bat" || exit /b 1
set "SUM=%ROOT%\logs\run_all.log"
set M="%PY%" -m helmet_ppe
echo === run_all %date% %time% > "%SUM%"

rem Sao luu trang thai ma nguon hien tai vao Git truoc khi chay.
git -C "%ROOT%" add -A > "%ROOT%\logs\git.log" 2>&1
git -C "%ROOT%" commit -q -m "Ghep mu theo dau (pose) + bo hop no_helmet duoi mu" >> "%ROOT%\logs\git.log" 2>&1
>> "%SUM%" echo git_commit: %errorlevel%

call "%ROOT%\scripts\setup.bat"
>> "%SUM%" echo setup: %errorlevel%

"%PY%" -c "import helmet_ppe.cli, helmet_ppe.batch, helmet_ppe.evaluation.detection_eval, helmet_ppe.evaluation.tuning; print('import OK')" > "%ROOT%\logs\imports.log" 2>&1
>> "%SUM%" echo imports: %errorlevel%

call "%ROOT%\scripts\run_tests.bat" >nul
>> "%SUM%" echo pytest: %errorlevel%

%M% predict "data\raw\anh_ct_nh (1).jpg" --output outputs\single > "%ROOT%\logs\predict.log" 2>&1
>> "%SUM%" echo predict: %errorlevel%

%M% batch --input data\raw --output outputs\experiment_01 > "%ROOT%\logs\batch_experiment_01.log" 2>&1
>> "%SUM%" echo batch_experiment_01: %errorlevel%

call "%ROOT%\scripts\run_evaluation.bat"
>> "%SUM%" echo evaluation: %errorlevel%

%M% batch --params reports\rule_tuning\best\best_params.json --input data\crowd_samples --output outputs\crowd_samples > "%ROOT%\logs\batch_crowd.log" 2>&1
>> "%SUM%" echo batch_crowd: %errorlevel%
%M% batch --params reports\rule_tuning\best_nopose\best_params.json --input data\crowd_samples --output outputs\crowd_samples_nopose > "%ROOT%\logs\batch_crowd_nopose.log" 2>&1
>> "%SUM%" echo batch_crowd_nopose: %errorlevel%

"%PY%" scripts\find_dataset_duplicates.py > "%ROOT%\logs\duplicates.log" 2>&1
>> "%SUM%" echo duplicates: %errorlevel%

"%PY%" scripts\collect_figures.py > "%ROOT%\logs\collect_figures.log" 2>&1
>> "%SUM%" echo collect_figures: %errorlevel%

"%PY%" scripts\check_streamlit.py > "%ROOT%\logs\streamlit_check.log" 2>&1
>> "%SUM%" echo streamlit_check: %errorlevel%

>> "%SUM%" echo === done %date% %time%
type "%SUM%"
