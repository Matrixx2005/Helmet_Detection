@echo off
rem Thiet lap moi truong chung cho cac script .bat (duoc goi bang "call").
chcp 65001 >nul
set "ROOT=%~dp0.."
pushd "%ROOT%"
set "ROOT=%CD%"
set "PY=%ROOT%\.venv\Scripts\python.exe"
set "PYTHONPATH=%ROOT%\src"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
if not exist "%PY%" (
    echo [LOI] Khong tim thay %PY%. Hay tao venv: python -m venv .venv
    exit /b 1
)
if not exist "%ROOT%\logs" mkdir "%ROOT%\logs"
exit /b 0
