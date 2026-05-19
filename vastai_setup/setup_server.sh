#!/bin/bash
# Setup serveur Vast.ai pour build ML TradingBot
# Usage : bash setup_server.sh
# Pre-req : SSH connection au serveur Vast.ai

set -e

echo "=== SETUP VAST.AI POUR BUILD TRADINGBOT ==="
echo ""

# 1. Update system
echo "[1/5] Update apt..."
apt-get update -qq
apt-get install -y python3.11 python3.11-venv python3-pip git wget curl htop -qq

# 2. Setup workdir
echo "[2/5] Setup workdir /workspace/TradingBot..."
mkdir -p /workspace/TradingBot
cd /workspace/TradingBot

# 3. Clone repo (si pas deja la)
if [ ! -d ".git" ]; then
    echo "[3/5] Clone repo GitHub..."
    git clone https://github.com/simovadev/tradingbote.git .
else
    echo "[3/5] Repo deja la, pull..."
    git pull
fi

# 4. Setup venv + dependencies
echo "[4/5] Setup Python venv + install deps..."
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip -q
pip install -q \
    pandas \
    numpy \
    pyarrow \
    lightgbm \
    scikit-learn \
    psutil

# 5. Create data dirs
echo "[5/5] Create data dirs..."
mkdir -p data/cache
mkdir -p data/ml_partial_M1
mkdir -p data_admiral

echo ""
echo "=== SETUP TERMINE ==="
echo ""
echo "Prochaine etape :"
echo "1. Upload data depuis ton PC :"
echo "   scp -P <port> -r data/cache/ root@<ip>:/workspace/TradingBot/data/"
echo "   scp -P <port> -r data_admiral/ root@<ip>:/workspace/TradingBot/"
echo ""
echo "2. Lancer build all :"
echo "   cd /workspace/TradingBot && bash vastai_setup/build_all.sh"
