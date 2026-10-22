@echo off
rem Mo giao dien demo Streamlit trong trinh duyet.
call "%~dp0_env.bat" || exit /b 1
rem Che do may ca nhan: cho phep chon thu muc anh co san tren may.
set "HELMET_LOCAL_MODE=1"
"%PY%" -m streamlit run app\streamlit_app.py
