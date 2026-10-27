@echo off
rem Ghi thong tin cac mo hinh (kien truc, so tham so, tham so huan luyen) vao reports\model_info.json
call "%~dp0_env.bat" || exit /b 1
"%PY%" scripts\model_info.py > "%ROOT%\logs\model_info.log" 2>&1
>> "%ROOT%\logs\model_info.log" echo exit: %errorlevel%
