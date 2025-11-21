@echo off
:: 使用方法: run_with_env.bat [环境名] [Python脚本] [参数...]

if "%1"=="" (
    echo 使用方法: run_with_env.bat ^<环境名^> ^<Python脚本^> [参数...]
    echo.
    echo 示例:
    echo   run_with_env.bat my_env main.py
    echo   run_with_env.bat my_env train.py --epochs 100
    echo   run_with_env.bat my_env -c "import numpy; print(numpy.__version__)"
    pause
    exit /b 1
)

set ENV_NAME=%1
set SCRIPT=%2
shift
shift

:: 检查环境是否存在
conda env list | findstr /C:"%ENV_NAME%" >nul
if %errorlevel% neq 0 (
    echo [错误] 环境 "%ENV_NAME%" 不存在
    echo 可用环境:
    conda env list
    pause
    exit /b 1
)

:: 检查脚本是否存在（如果不是 -c 命令）
if not "%SCRIPT%"=="-c" if not exist "%SCRIPT%" (
    echo [错误] 脚本文件 "%SCRIPT%" 不存在
    pause
    exit /b 1
)

echo [信息] 使用环境: %ENV_NAME%
echo [信息] 执行命令: python %SCRIPT% %*
echo.

:: 执行命令
conda run -n %ENV_NAME% python %SCRIPT% %*

echo.
echo [完成] 命令执行完毕
pause