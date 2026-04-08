@echo off
setlocal EnableDelayedExpansion

:: 1. Setup Encoding and Title
chcp 65001 >nul
title SputnikSim — Launch System

:: Set Working Directory to the folder where this BAT is located
set "ROOT_DIR=%~dp0"
cd /d "%ROOT_DIR%"

echo.
echo  ███████╗██████╗ ██╗   ██╗████████╗███╗   ██╗██╗██╗  ██╗███████╗██╗███╗   ███╗
echo  ██╔════╝██╔══██╗██║   ██║╚══██╔══╝████╗  ██║██║██║ ██╔╝██╔════╝██║████╗ ████║
echo  ███████╗██████╔╝██║   ██║   ██║   ██╔██╗ ██║██║█████╔╝ ███████╗██║██╔████╔██║
echo  ╚════██║██╔═══╝ ██║   ██║   ██║   ██║╚██╗██║██║██╔═██╗ ╚════██║██║██║╚██╔╝██║
echo  ███████║██║     ╚██████╔╝   ██║   ██║ ╚████║██║██║  ██╗███████║██║██║ ╚═╝ ██║
echo  ╚══════╝╚═╝      ╚═════╝    ╚═╝   ╚═╝  ╚═══╝╚═╝╚═╝  ╚═╝╚══════╝╚═╝╚═╝     ╚═╝
echo.
echo  Satellite Orbit Analysis and Simulation System
echo  ════════════════════════════════════════════════════════════════════════════════
echo.

:: 2. Check Python Installation
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python was not found!
    echo          Please install Python 3.8+ and add it to PATH.
    pause
    exit /b 1
)

:: 3. Virtual Environment Management
if not exist "venv" (
    echo  [SETUP] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo  [ERROR] Failed to create venv. Check folder permissions.
        pause
        exit /b 1
    )
)

:: Define paths to venv binaries directly (Avoids activation bugs)
set "VENV_PYTHON=%ROOT_DIR%venv\Scripts\python.exe"
set "VENV_PIP=%ROOT_DIR%venv\Scripts\pip.exe"

if not exist "%VENV_PYTHON%" (
    echo  [ERROR] Virtual environment is corrupted. Delete 'venv' folder and restart.
    pause
    exit /b 1
)

echo  [OK] Virtual Environment ready.
echo.

:: 4. Dependencies Installation
echo  [SETUP] Checking dependencies...
"%VENV_PYTHON%" -m pip install --upgrade pip --quiet

if exist "requirements.txt" (
    "%VENV_PIP%" install -r requirements.txt --quiet
    if errorlevel 1 (
        echo  [ERROR] Failed to install requirements. Check your internet connection.
        pause
        exit /b 1
    )
)

:: Ensure PyQt5 is present
"%VENV_PYTHON%" -c "import PyQt5" >nul 2>&1
if errorlevel 1 (
    echo  [SETUP] Installing PyQt5...
    "%VENV_PIP%" install PyQt5 --quiet
)

echo  [OK] All dependencies verified.
echo.

:: 5. Create Desktop Shortcut (Reliable PowerShell method)
set "SHORTCUT_NAME=SputnikSim.lnk"
set "ICON_PATH=%ROOT_DIR%app.ico"

for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "[Environment]::GetFolderPath('Desktop')"`) do set "DESKTOP_PATH=%%i"

if not exist "%DESKTOP_PATH%\%SHORTCUT_NAME%" (
    echo  [SETUP] Creating desktop shortcut...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$s=(New-Object -ComObject WScript.Shell).CreateShortcut('%DESKTOP_PATH%\%SHORTCUT_NAME%'); $s.TargetPath='%~f0'; $s.WorkingDirectory='%ROOT_DIR%'; if(Test-Path '%ICON_PATH%'){$s.IconLocation='%ICON_PATH%'}; $s.Description='SputnikSim Launcher'; $s.Save()"
    echo  [OK] Shortcut created on Desktop.
)

:: 6. Qt Environment Fix (Prevents UI crashes)
set "QT_PLUGIN_PATH=%ROOT_DIR%venv\Lib\site-packages\PyQt5\Qt5\plugins"
set "QT_QPA_PLATFORM_PLUGIN_PATH=%ROOT_DIR%venv\Lib\site-packages\PyQt5\Qt5\plugins\platforms"

:: 7. Check for AI model
if not exist "AI\model\tle_model_best.pth" (
    echo  [AI] WARNING: AI model not found. Some features might be disabled.
)

:: 8. Launch main.py
if exist "main.py" (
    echo  [SYS] Launching SputnikSim Application...
    echo  ════════════════════════════════════════════════════════════════════════════════
    echo.
    "%VENV_PYTHON%" "%ROOT_DIR%main.py"
    if errorlevel 1 (
        echo.
        echo  ════════════════════════════════════════════════════════════════════════════════
        echo  [ERROR] main.py завершился с ошибкой! Смотрите текст выше.
        echo  ════════════════════════════════════════════════════════════════════════════════
        pause
        exit /b 1
    )
) else (
    echo  [ERROR] File 'main.py' not found in %ROOT_DIR%
    pause
    exit /b 1
)

echo.
echo  ════════════════════════════════════════════════════════════════════════════════
echo  [SYS] Session ended.
pause