#!/bin/bash

echo  ███████╗██████╗ ██╗   ██╗████████╗███╗   ██╗██╗██╗  ██╗███████╗██╗███╗   ███╗
echo  ██╔════╝██╔══██╗██║   ██║╚══██╔══╝████╗  ██║██║██║ ██╔╝██╔════╝██║████╗ ████║
echo  ███████╗██████╔╝██║   ██║   ██║   ██╔██╗ ██║██║█████╔╝ ███████╗██║██╔████╔██║
echo  ╚════██║██╔═══╝ ██║   ██║   ██║   ██║╚██╗██║██║██╔═██╗ ╚════██║██║██║╚██╔╝██║
echo  ███████║██║     ╚██████╔╝   ██║   ██║ ╚████║██║██║  ██╗███████║██║██║ ╚═╝ ██║
echo  ╚══════╝╚═╝      ╚═════╝    ╚═╝   ╚═╝  ╚═══╝╚═╝╚═╝  ╚═╝╚══════╝╚═╝╚═╝     ╚═╝

if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 not found. Install it:"
    echo "  sudo apt install python3 python3-pip python3-venv"
    exit 1
fi

echo "[OK] Python3 found: $(python3 --version)"

if [ ! -d "venv" ]; then
    echo "[...] Creating virtual environment..."
    python3 -m venv venv
    echo "[OK] venv created"
else
    echo "[OK] venv already exists"
fi

source venv/bin/activate

echo "[...] Installing dependencies..."
pip install --upgrade pip -q

pip install torch --index-url https://download.pytorch.org/whl/cpu -q
pip install numpy scikit-learn sgp4 requests qasync -q

echo "[...] Installing PyQt5..."
pip install PyQt5 -q || {
    echo "[WARN] pip install failed, trying system package..."
    sudo apt install python3-pyqt5 -y
}

echo ""
echo "[OK] Dependencies installed"
echo ""
echo "================================"
echo "  Starting SputnikSim..."
echo "================================"
python3 main.py
