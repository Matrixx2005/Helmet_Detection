@echo off
rem Chay toan bo quy trinh kiem tra cuoi cung, ghi log vao logs\.
call "%~dp0_env.bat" || exit /b 1
set "SUM=%ROOT%\logs\run_all.log"
set M="%PY%" -m helmet_ppe
echo === run_all %date% %time% > "%SUM%"

call "%ROOT%\scripts\setup.bat"
echo setup: %errorlevel%>> "%SUM%"

"%PY%" -c "import helmet_ppe.cli, helmet_ppe.batch, helmet_ppe.evaluation.detection_eval, helmet_ppe.evaluation.tuning; print('import OK')" > "%ROOT%\logs\imports.log" 2>&1
echo imports: %errorlevel%>> "%SUM%"

call "%ROOT%\scripts\run_tests.bat" >nul
echo pytest: %errorlevel%>> "%SUM%"

%M% predict "data\raw\anh_ct_nh (1).jpg" --output outputs\single > "%ROOT%\logs\predict.log" 2>&1
echo predict: %errorlevel%>> "%SUM%"

%M% batch --input data\raw --output outputs\experiment_01 > "%ROOT%\logs\batch_experiment_01.log" 2>&1
echo batch_experiment_01: %errorlevel%>> "%SUM%"

call "%ROOT%\scripts\run_evaluation.bat"
echo evaluation: %errorlevel%>> "%SUM%"

"%PY%" scripts\check_streamlit.py > "%ROOT%\logs\streamlit_check.log" 2>&1
echo streamlit_check: %errorlevel%>> "%SUM%"

echo === done %date% %time%>> "%SUM%"
type "%SUM%"
