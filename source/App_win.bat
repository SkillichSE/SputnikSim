@echo off
chcp 65001 >nul
title SputnikSim — Launch
setlocal EnableDelayedExpansion

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

REM ======================================================
REM 1. Python check
REM ======================================================

python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found.
    echo          Install Python 3.8+ and add it to PATH.
    echo.
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version') do set PYVER=%%v
echo  [OK] Python detected: %PYVER%
echo.

REM ======================================================
REM 2. Create venv
REM ======================================================

if not exist venv (
    echo  [SETUP] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo  [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo  [OK] Virtual environment created
    echo.
)

call venv\Scripts\activate
if errorlevel 1 (
    echo  [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)

echo  [SYS] Virtual environment activated
echo.

REM ======================================================
REM 3. Update pip
REM ======================================================

echo  [SETUP] Upgrading pip...
python -m pip install --upgrade pip >nul 2>&1
echo  [OK] pip ready
echo.

REM ======================================================
REM 4. Install dependencies
REM ======================================================

if not exist requirements.txt (
    echo  [WARNING] requirements.txt not found.
) else (
    echo  [SETUP] Installing dependencies from requirements.txt...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo  [ERROR] Failed installing requirements.
        pause
        exit /b 1
    )
)

python -c "import PyQt5" >nul 2>&1
if errorlevel 1 (
    echo  [SETUP] Installing PyQt5...
    pip install PyQt5
    if errorlevel 1 (
        echo  [ERROR] Failed installing PyQt5.
        pause
        exit /b 1
    )
)

echo  [OK] All dependencies verified
echo.

REM ======================================================
REM 5. Create desktop shortcut (first run only)
REM ======================================================

echo  [SETUP] Checking desktop shortcut...

REM Full path to this BAT file
set "SCRIPT_PATH=%~f0"
set "SCRIPT_DIR=%~dp0"

REM Get real Desktop path (OneDrive & localization safe)
for /f "delims=" %%i in ('powershell -NoProfile -Command "[Environment]::GetFolderPath(\"Desktop\")"') do set "DESKTOP=%%i"

set "SHORTCUT=%DESKTOP%\SputnikSim.lnk"
set "ICON_PATH=%SCRIPT_DIR%app.ico"

REM If shortcut already exists → skip creation
if exist "%SHORTCUT%" (
    echo  [OK] Shortcut already exists.
) else (
    echo  [SETUP] Creating desktop shortcut...

    set "PS_TMP=%TEMP%\sputniksim_shortcut.ps1"
    (
    echo $src  = "%SCRIPT_PATH%"
    echo $dst  = "%SHORTCUT%"
    echo $wdir = "%SCRIPT_DIR%"
    echo $ico  = "%ICON_PATH%"
    echo $ws   = New-Object -ComObject WScript.Shell
    echo $sc   = $ws.CreateShortcut($dst^)
    echo $sc.TargetPath       = $src
    echo $sc.WorkingDirectory = $wdir
    echo if (Test-Path $ico^) { $sc.IconLocation = $ico }
    echo $sc.Description      = "SputnikSim Launcher"
    echo $sc.Save(^)
    ) > "%PS_TMP%"

    powershell -NoProfile -ExecutionPolicy Bypass -File "%PS_TMP%" >nul 2>&1

    if errorlevel 1 (
        echo  [WARNING] Failed to create shortcut.
    ) else (
        echo  [OK] Shortcut created: SputnikSim
    )

    del "%PS_TMP%" >nul 2>&1
)

echo.
REM ======================================================
REM 6. Check for AI model
REM ======================================================

if not exist AI\model\tle_model_best.pth (
    echo  [AI] Model not found.
    echo       To train:
    echo       cd AI ^&^& python train_model.py
    echo.
)

REM ======================================================
REM 7. Check main.py
REM ======================================================

if not exist main.py (
    echo  [ERROR] main.py not found.
    pause
    exit /b 1
)

set QT_PLUGIN_PATH=%CD%\venv\Lib\site-packages\PyQt5\Qt5\plugins
set QT_QPA_PLATFORM_PLUGIN_PATH=%CD%\venv\Lib\site-packages\PyQt5\Qt5\plugins\platforms

echo  [SYS] Launching SputnikSim...
echo  ════════════════════════════════════════════════════════════════════════════════
echo.

python main.py

echo.
echo  ════════════════════════════════════════════════════════════════════════════════
echo  [SYS] Session ended.
pause