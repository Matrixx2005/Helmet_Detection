@echo off
rem Cai thu vien, chuyen code cu vao legacy\ va tao ban sao luu Git.
call "%~dp0_env.bat" || exit /b 1
set "LOG=%ROOT%\logs\setup.log"
echo === setup %date% %time% > "%LOG%"

"%PY%" -m pip install --upgrade pip >> "%LOG%" 2>&1
"%PY%" -m pip install -r requirements.txt >> "%LOG%" 2>&1
if errorlevel 1 (echo [LOI] pip install that bai, xem %LOG% & exit /b 1)

rem Code thu nghiem cu duoc giu lai trong legacy\ (khong xoa).
if not exist legacy mkdir legacy
for %%F in (association.py detector.py rule_engine.py test_association.py test_custom_model.py test_detector.py test_rule_engine.py test_yolo.py main.py batch_test_yolo.py video_processor.py) do (
    if exist "src\%%F" move /y "src\%%F" "legacy\%%F" >> "%LOG%" 2>&1
)

where git >nul 2>&1
if not errorlevel 1 if not exist .git (
    git init >> "%LOG%" 2>&1
    git add -A >> "%LOG%" 2>&1
    call :git_commit
)
"%PY%" -c "import ultralytics, streamlit, pytest; print('ultralytics', ultralytics.__version__, '| streamlit', streamlit.__version__, '| pytest', pytest.__version__)" >> "%LOG%" 2>&1
echo [XONG] Xem chi tiet: %LOG%
type "%LOG%" | findstr /i "ultralytics error loi"
exit /b 0

:git_commit
rem Dung danh tinh Git san co; neu chua cau hinh thi dung ten tai khoan Windows.
set "GITID="
git config user.email >nul 2>&1 || set GITID=-c user.name="%USERNAME%" -c user.email="%USERNAME%@localhost"
git %GITID% commit -m "Hoan thien he thong phat hien mu bao ho" >> "%LOG%" 2>&1
exit /b 0
