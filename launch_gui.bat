@echo off
chcp 65001 >nul
setlocal

REM 切换到本批处理所在目录（项目根目录）
cd /d "%~dp0"

set "VENV_PYTHON=.venv\Scripts\python.exe"

if not exist "%VENV_PYTHON%" (
    echo [错误] 找不到虚拟环境解释器：%VENV_PYTHON%
    echo 请先在项目根目录运行：python -m venv .venv
    pause
    exit /b 1
)

echo 正在启动 Ballistic Sim 可视化界面...
"%VENV_PYTHON%" -m ballistic_sim.gui

if errorlevel 1 (
    echo [错误] 可视化界面启动失败，错误代码：%errorlevel%
    pause
)

endlocal
