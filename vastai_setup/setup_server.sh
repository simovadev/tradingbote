#!/bin/bash
# Setup serveur Vast.ai (Ubuntu 24.04) pour build ML TradingBot V4
# Usage : bash setup_server.sh

set -e

echo "============================================="
echo "  SETUP VAST.AI - TRADINGBOT V4"
echo "============================================="
echo ""

# 1. Update apt + tools de base
echo "[1/6] Update apt + tools..."
apt-get update -qq
apt-get install -y -qq \
    git wget curl htop unzip vim nano \
    build-essential \
    software-properties-common \
    ca-certificates

# 2. Python (3.12 par defaut sur Ubuntu 24.04)
echo "[2/6] Install Python + pip + venv..."
apt-get install -y -qq \
    python3 python3-pip python3-venv python3-dev

PYBIN=$(which python3)
PYVER=$($PYBIN --version | awk '{print $2}')
echo "Python : $PYVER ($PYBIN)"

# 3. Workdir + clone repo
echo "[3/6] Setup /workspace/TradingBot..."
mkdir -p /workspace/TradingBot
cd /workspace/TradingBot

if [ ! -d ".git" ]; then
    echo "  Clone repo GitHub..."
    git clone https://github.com/simovadev/tradingbote.git .
else
    echo "  Repo deja la, git pull..."
    git pull
fi

# 4. Venv + requirements.txt
echo "[4/6] Setup venv + install requirements.txt..."
if [ ! -d "venv" ]; then
    $PYBIN -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip -q

# Install via requirements.txt (skip MetaTrader5 sur Linux : Windows only)
echo "  Install requirements.txt (skip MetaTrader5 = Windows only)..."
grep -v "^MetaTrader5" requirements.txt > /tmp/requirements_linux.txt
pip install -q -r /tmp/requirements_linux.txt

echo "  Versions installees :"
python -c "import pandas, numpy, pyarrow, lightgbm, sklearn; \
  print(f'    pandas   {pandas.__version__}'); \
  print(f'    numpy    {numpy.__version__}'); \
  print(f'    pyarrow  {pyarrow.__version__}'); \
  print(f'    lightgbm {lightgbm.__version__}'); \
  print(f'    sklearn  {sklearn.__version__}')"

# 5. Create data dirs
echo "[5/6] Create data dirs..."
mkdir -p data/cache
mkdir -p data/ml_partial_M1
mkdir -p data_admiral
mkdir -p data_csv

# 6. Verifie l'install
echo "[6/6] Smoke test imports..."
python -c "
import sys
sys.path.insert(0, '/workspace/TradingBot')
from bot_v2.concepts.order_block import detect_order_blocks
from bot_v2.pipeline import evaluate_ob
from bot_v2.ml_filter import _features_from_result
print('  OK : pipeline + ml_filter')
"

echo ""
echo "============================================="
echo "  SETUP TERMINE"
echo "============================================="
echo ""
echo "CPU      : $(nproc) threads"
echo "RAM      : $(free -h | awk '/^Mem:/ {print $2}') total"
echo "Disk     : $(df -h /workspace | tail -1 | awk '{print $4 " free"}')"
echo ""
echo "Prochaine etape :"
echo "1. Upload CSV depuis ton PC (PowerShell)"
echo "2. cd /workspace/TradingBot && source venv/bin/activate"
echo "3. bash vastai_setup/build_v4_all.sh 2>&1 | tee build_v4.log"
