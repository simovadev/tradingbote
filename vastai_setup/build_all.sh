#!/bin/bash
# Build ML datasets V3.5 pour les 13 actifs sur Vast.ai 96 cores
# Usage : bash build_all.sh

set -e

cd /workspace/TradingBot
source venv/bin/activate

# Adapter les paths Windows -> Linux dans le code (1 fois)
echo "[1/4] Adaptation paths Windows -> Linux..."
find bot_v2/ -name "*.py" -exec sed -i 's|c:/Users/Shadow/TradingBot|/workspace/TradingBot|g' {} \;
find bot_v2/ -name "*.py" -exec sed -i 's|c:\\\\Users\\\\Shadow\\\\TradingBot|/workspace/TradingBot|g' {} \;

# Augmenter n_workers pour exploiter les 96 cores
echo "[2/4] Augmente n_workers a 32 (sweet spot 96 cores AMD)..."
sed -i 's|n_workers = min(6, len(tasks_to_do))|n_workers = min(32, len(tasks_to_do))|g' bot_v2/ml_dataset.py

# Liste des 13 actifs a builder (XAUUSD deja fait)
ACTIFS=("SP500" "NAS100" "GER40" "UK100" "FRA40" "JP225" "EURUSD" "GBPUSD" "USDJPY" "AUDUSD" "USDCAD" "USDCHF" "DJ30")

echo "[3/4] Build des ${#ACTIFS[@]} actifs en sequence..."
for ACTIF in "${ACTIFS[@]}"; do
    echo ""
    echo "=========================================="
    echo "  BUILD $ACTIF (V3.5 sur Admiral)"
    echo "=========================================="
    SCRIPT="bot_v2/ml_dataset_${ACTIF}_admiral_8ans_V3_5.py"
    if [ -f "$SCRIPT" ]; then
        python "$SCRIPT" 2>&1 | tee "build_${ACTIF}.log"
    else
        echo "SKIP : $SCRIPT n'existe pas"
    fi
done

echo ""
echo "[4/4] Train des modeles..."
for ACTIF in "${ACTIFS[@]}"; do
    SCRIPT="bot_v2/ml_train_${ACTIF}_admiral_8ans_V3_5.py"
    if [ -f "$SCRIPT" ]; then
        echo "Train $ACTIF..."
        python "$SCRIPT" 2>&1 | tee "train_${ACTIF}.log"
    fi
done

echo ""
echo "=== BUILD ALL TERMINE ==="
echo "Modeles dans : bot_v2/ml_model_*_admiral_v3_5.pkl"
echo "Datasets dans : data/ml_dataset_*_admiral_8ans_V3_5.parquet"
echo ""
echo "Download depuis ton PC :"
echo "  scp -P <port> root@<ip>:/workspace/TradingBot/bot_v2/ml_model_*_admiral_v3_5.pkl ./bot_v2/"
echo "  scp -P <port> root@<ip>:/workspace/TradingBot/bot_v2/ml_features_*_admiral_v3_5.json ./bot_v2/"
