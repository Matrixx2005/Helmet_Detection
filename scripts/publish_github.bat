@echo off
rem Kiem tra giao dien, commit va day toan bo du an len GitHub.
call "%~dp0_env.bat" || exit /b 1
cd /d "%ROOT%"
set "LOG=%ROOT%\logs\publish_github.log"
set "URL=https://github.com/Matrixx2005/Helmet_Detection.git"
echo === publish %date% %time% > "%LOG%"

echo [1/3] Kiem tra giao dien Streamlit (ban chia se)...
"%PY%" scripts\check_streamlit.py >> "%LOG%" 2>&1
>> "%LOG%" echo check_streamlit: %errorlevel%
if errorlevel 1 goto :fail

echo [2/3] Commit...
git add -A >> "%LOG%" 2>&1
git commit -q -F "%ROOT%\logs\commit_msg.txt" >> "%LOG%" 2>&1
>> "%LOG%" echo commit: %errorlevel%
git remote -v >> "%LOG%" 2>&1
git remote add origin "%URL%" 2>nul || git remote set-url origin "%URL%"
>> "%LOG%" echo branch:
git branch --show-current >> "%LOG%" 2>&1
>> "%LOG%" echo tracked_models:
git ls-files models >> "%LOG%" 2>&1
git count-objects -vH >> "%LOG%" 2>&1

echo [3/3] Day len GitHub (neu hien cua so dang nhap GitHub, hay dang nhap)...
git push -u origin HEAD:main >> "%LOG%" 2>&1
>> "%LOG%" echo push: %errorlevel%
git log --oneline -3 >> "%LOG%" 2>&1
type "%LOG%"
pause
exit /b 0

:fail
type "%LOG%"
echo Kiem tra that bai, chua day len GitHub.
pause
exit /b 1
