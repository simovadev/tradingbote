#!/bin/bash
# Train V4 modeles LightGBM pour 14 actifs.
# Sur Vast.ai : ~10-30 min total apres build.

set -e
cd /workspace/TradingBot
export PYTHONPATH=/workspace/TradingBot

ASSETS=("XAUUSD" "NAS100" "GER40" "BTCUSD" "EURUSD" "GBPUSD" "AUDUSD" "USDJPY" "SP500" "DJ30" "UK100" "FRA40" "USDCAD" "USDCHF")

START=$(date +%s)
for ASSET in "${ASSETS[@]}"; do
    T0=$(date +%s)
    echo ""
    echo "========================================"
    echo "  TRAIN $ASSET V4 ($(date +%H:%M:%S))"
    echo "========================================"
    python3 -m bot_v2.ml_train_v4 "$ASSET" 2>&1
    T1=$(date +%s)
    DUR=$((T1 - T0))
    echo "Duree $ASSET : ${DUR}s"
done
TOTAL=$(date +%s)
echo ""
echo "========================================"
echo "  ALL TRAINS DONE in $((TOTAL - START))s"
echo "========================================"
ls -lh /workspace/TradingBot/bot_v2/ml_model_*_admiral_v4.pkl
