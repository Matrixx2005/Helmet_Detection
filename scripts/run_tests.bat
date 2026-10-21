@echo off
rem Chay toan bo unit test + kiem thu tich hop voi models\best.pt.
call "%~dp0_env.bat" || exit /b 1
"%PY%" -m pytest -v > "%ROOT%\logs\pytest.log" 2>&1
set "RC=%errorlevel%"
type "%ROOT%\logs\pytest.log"
exit /b %RC%
