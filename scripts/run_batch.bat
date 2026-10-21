@echo off
rem Xu ly hang loat. Cach dung: run_batch.bat [thu_muc_anh] [thu_muc_ket_qua]
call "%~dp0_env.bat" || exit /b 1
set "INPUT=%~1"
if "%INPUT%"=="" set "INPUT=data\test_images"
if "%~2"=="" (
    "%PY%" -m helmet_ppe batch --input "%INPUT%"
) else (
    "%PY%" -m helmet_ppe batch --input "%INPUT%" --output "%~2"
)
