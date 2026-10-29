@echo off
REM One-time setup on Windows: creates .venv and installs the requirements.
cd /d "%~dp0"
set "PY=python"
where py >nul 2>nul && set "PY=py -3"
if exist ".venv\Scripts\python.exe" goto install
echo Creating virtual environment with %PY% ...
%PY% -m venv .venv
if errorlevel 1 goto fail
:install
echo Installing packages (PyTorch is a few hundred MB, this can take a while) ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto fail
echo.
echo Setup complete. Double-click run_demo_windows.bat to start.
pause
exit /b 0
:fail
echo.
echo Setup FAILED. Install Python 3.10 - 3.13 from python.org (tick "Add python.exe to PATH")
echo and run this file again.
pause
exit /b 1
