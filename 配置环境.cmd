@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  py -3 -m venv .venv
  if errorlevel 1 (
    echo 请先安装 Python 3.13（含 tkinter），然后重新运行本脚本。
    pause
    exit /b 1
  )
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  echo 依赖安装失败。请检查网络和 Python 环境。
  pause
  exit /b 1
)
".venv\Scripts\python.exe" launch.py --check
pause
