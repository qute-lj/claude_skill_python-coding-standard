@echo off
echo ========================================
echo Python Conda 环境检查工具
echo ========================================
echo.

:: 检查 conda 是否安装
conda --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未找到 conda 命令
    echo 请确保已安装 Anaconda 或 Miniconda 并添加到 PATH
    pause
    exit /b 1
)

echo [信息] Conda 版本:
conda --version
echo.

echo [信息] 可用的 Conda 环境:
echo ----------------------------------------
conda env list
echo.

echo [信息] 当前环境信息:
echo ----------------------------------------
conda info
echo.

echo [信息] 当前激活的环境: %CONDA_DEFAULT_ENV%
if "%CONDA_DEFAULT_ENV%"=="" (
    echo [警告] 当前没有激活任何 conda 环境
    echo 使用的 Python:
    where python
)

echo.
echo ========================================
echo 建议操作:
echo 1. 激活环境: conda activate env_name
echo 2. 直接运行: conda run -n env_name python script.py
echo 3. 创建新环境: conda create -n new_env python=3.9 -y
echo ========================================
pause