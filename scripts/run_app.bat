@echo off
rem Mo giao dien demo Streamlit trong trinh duyet.
call "%~dp0_env.bat" || exit /b 1
"%PY%" -m streamlit run app\streamlit_app.py
