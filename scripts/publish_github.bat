@echo off
rem Kiem tra giao dien, commit va day toan bo du an len GitHub.
call "%~dp0_env.bat" || exit /b 1
cd /d "%ROOT%"
set "LOG=%ROOT%\logs\publish_github.log"
set "URL=https://github.com/Matrixx2005/Helmet_Detection.git"
echo === publish %date% %time% > "%LOG%"

echo [1/4] Kiem tra giao dien Streamlit (ban chia se)...
"%PY%" scripts\check_streamlit.py >> "%LOG%" 2>&1
>> "%LOG%" echo check_streamlit: %errorlevel%
if errorlevel 1 goto :fail

echo [2/4] Commit...
git add -A >> "%LOG%" 2>&1
if exist "%ROOT%\logs\commit_msg.txt" git commit -q -F "%ROOT%\logs\commit_msg.txt" >> "%LOG%" 2>&1
>> "%LOG%" echo commit: %errorlevel%
git remote add origin "%URL%" 2>nul || git remote set-url origin "%URL%"

echo [3/4] Doc noi dung dang co tren GitHub...
git fetch origin >> "%LOG%" 2>&1
>> "%LOG%" echo fetch: %errorlevel%
>> "%LOG%" echo --- remote log
git log --format="%%h %%an %%ad %%s" -20 origin/main >> "%LOG%" 2>&1
>> "%LOG%" echo --- remote files
git ls-tree -r --name-only origin/main >> "%LOG%" 2>&1
>> "%LOG%" echo --- local is ahead of remote?
git merge-base --is-ancestor origin/main HEAD >> "%LOG%" 2>&1
>> "%LOG%" echo ancestor: %errorlevel%

echo [4/4] Day len GitHub...
git push -u origin HEAD:main >> "%LOG%" 2>&1
>> "%LOG%" echo push: %errorlevel%
git log --oneline -3 >> "%LOG%" 2>&1
type "%LOG%"
exit /b 0

:fail
type "%LOG%"
echo Kiem tra that bai, chua day len GitHub.
exit /b 1
