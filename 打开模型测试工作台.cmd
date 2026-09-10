@echo off
chcp 65001 >nul
cd /d "%~dp0"
set /p TEST_PYTHON=<python_path.txt
"%TEST_PYTHON%" -X utf8 -m scripts.external_test_gui
if errorlevel 1 pause
