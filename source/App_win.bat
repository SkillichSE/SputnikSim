@echo off
chcp 65001 >nul
title SputnikSim — Launch

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

if not exist venv (
    echo  [SETUP] First launch — creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo.
        echo  [ERROR] Failed to create virtual environment.
        echo          Make sure Python 3.8+ is installed and available in PATH.
        echo.
        pause
        exit /b 1
    )
    echo  [OK]    Virtual environment created
    echo.
)

call venv\Scripts\activate

if not exist venv\Lib\site-packages\PyQt5 (
    echo  [SETUP] Installing dependencies — this may take a while...
    echo.
    python -m pip install --upgrade pip -q
    pip install -r requirements.txt -q
    pip install PyQt5 -q
    echo  [OK]    Dependencies installed
    echo.
) else (
    echo  [OK]    Dependencies already installed
    echo.
)

if not exist AI\model\tle_model_best.pth (
    echo  [AI]    Model not found.
    echo          To train: cd AI ^&^& python train_model.py
    echo.
)

echo  [SYS]   Launching SputnikSim...
echo  ════════════════════════════════════════════════════════════════════════════════
echo.

python main.py

echo.
echo  ════════════════════════════════════════════════════════════════════════════════
echo  [SYS]   Session ended.
pause

