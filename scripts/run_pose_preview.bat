@echo off
call "%~dp0_env.bat" || exit /b 1
cd /d "%ROOT%\models"
"%PY%" "%ROOT%\scripts\pose_preview.py" "%ROOT%\data\crowd_samples" yolo26n-pose.pt > "%ROOT%\logs\pose_preview.log" 2>&1
>> "%ROOT%\logs\pose_preview.log" echo EXIT %errorlevel%
