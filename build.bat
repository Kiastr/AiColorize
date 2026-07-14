@echo off
setlocal enabledelayedexpansion
title AiColorize One-Click Builder

echo ===========================================
echo   AiColorize 桌面应用一键打包工具
echo ===========================================
echo.

:: 检查 Python 环境
set "PYTHON_EXE=python\python.exe"
if not exist "%PYTHON_EXE%" (
    echo [错误] 未在当前目录下找到 python 文件夹。
    echo 请确保您已经解压了 python_env_deoldify.zip 并且 build.bat 位于同一目录下。
    pause
    exit /b 1
)

echo [1/3] 正在安装必要依赖 (PyQt5, PyInstaller, requests)...
"%PYTHON_EXE%" -m pip install pyinstaller pyqt5 requests --no-warn-script-location
if errorlevel 1 (
    echo [错误] 依赖安装失败，请检查网络连接。
    pause
    exit /b 1
)

echo.
echo [2/3] 正在执行打包程序 (编译为单文件 EXE)...
echo 这可能需要几分钟，请耐心等待...
"%PYTHON_EXE%" -m PyInstaller --noconsole --onefile --name "AiColorize" --clean app.py
if errorlevel 1 (
    echo [错误] 打包失败。
    pause
    exit /b 1
)

echo.
echo [3/3] 正在清理临时文件...
rd /s /q build
del /q AiColorize.spec

echo.
echo ===========================================
echo   打包成功完成！
echo ===========================================
echo.
echo 生成的文件位于: dist\AiColorize.exe
echo 您现在可以将 dist 文件夹中的 exe 发送给任何人使用了。
echo.
pause
