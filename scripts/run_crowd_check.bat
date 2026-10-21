@echo off
rem Kiem tra anh dong nguoi: chay voi nguong mac dinh va voi nguong rat thap (de xem detection tho).
call "%~dp0_env.bat" || exit /b 1
set M="%PY%" -m helmet_ppe
%M% batch --input data\crowd_samples --output outputs\crowd_default > "%ROOT%\logs\crowd_check.log" 2>&1
%M% batch --input data\crowd_samples --output outputs\crowd_lowconf --conf-person 0.02 --conf-helmet 0.02 --conf-no-helmet 0.02 >> "%ROOT%\logs\crowd_check.log" 2>&1
%M% batch --input data\crowd_samples --output outputs\crowd_lowconf_960 --imgsz 960 --conf-person 0.02 --conf-helmet 0.02 --conf-no-helmet 0.02 >> "%ROOT%\logs\crowd_check.log" 2>&1
>> "%ROOT%\logs\crowd_check.log" echo DONE
