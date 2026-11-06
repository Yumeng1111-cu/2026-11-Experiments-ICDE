@echo off
cd /d "%~dp0"
python -m pip install -r requirements.txt
if errorlevel 1 goto failure
python run.py
if errorlevel 1 goto failure
pause
exit /b 0
:failure
echo.
echo The run failed. Check the error message and README.md.
pause
exit /b 1
