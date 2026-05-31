@echo off
setlocal
cd /d "%~dp0"
set "PHOENIX_WORKING_DIR=%cd%\logs\.phoenix"
set "PYTHONIOENCODING=utf-8"
venv\Scripts\python.exe -m phoenix.server.main serve
