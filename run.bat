@echo off
REM /Users/ou/project/nexus_copilot/src-tauri/scripts/run.bat

setlocal

REM Get the directory of the script
set "SCRIPT_DIR=%~dp0"
set "BACKEND_DIR=%SCRIPT_DIR%..\backend"

REM The port is passed as the first argument
set "PORT=%~1"

echo Starting backend service on port %PORT%...

REM Navigate to the backend directory
cd /D "%BACKEND_DIR%"

REM Activate venv and run uvicorn
call "venv\Scripts\activate.bat"
uvicorn app.main:app --host 0.0.0.0 --port "%PORT%"