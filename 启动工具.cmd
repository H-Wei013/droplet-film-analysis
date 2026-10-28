@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "TOOL_PYTHON="
if exist "local_python.cmd" call "local_python.cmd"
if exist ".venv\Scripts\python.exe" set "TOOL_PYTHON=%~dp0.venv\Scripts\python.exe"
if not defined TOOL_PYTHON set "TOOL_PYTHON=python"
"%TOOL_PYTHON%" launch.py --check >nul 2>&1
if errorlevel 1 (
  echo Python 环境尚未就绪。请先运行“配置环境.cmd”，或参考 README 配置环境。
  "%TOOL_PYTHON%" launch.py --check
  pause
  exit /b 1
)
set "TOOL_PYTHONW=%TOOL_PYTHON:python.exe=pythonw.exe%"
if exist "%TOOL_PYTHONW%" (
  start "" "%TOOL_PYTHONW%" "%~dp0launch.py"
) else (
  "%TOOL_PYTHON%" "%~dp0launch.py"
)
