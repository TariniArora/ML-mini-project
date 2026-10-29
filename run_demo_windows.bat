@echo off
REM Starts the demo (webcam + panel + simulated robot). Extra arguments are passed on,
REM e.g.  run_demo_windows.bat --camera 1
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" call setup_windows.bat
".venv\Scripts\python.exe" main.py --demo %*
pause
