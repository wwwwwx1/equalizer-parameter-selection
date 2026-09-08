@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
echo Create .venv and install requirements first.
pause
exit /b 1
)
.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
if errorlevel 1 (
echo FAILED - see details above.
pause
exit /b 1
)
pause
