#!/bin/bash
# Build V4 (displacement dynamique) pour les 14 actifs.
# Sur Vast.ai EPYC 9554/9654 96-128 cores -> ~6h total estimé.

set -e
cd /workspace/TradingBot
export PYTHONPATH=/workspace/TradingBot

ASSETS=("XAUUSD" "NAS100" "GER40" "BTCUSD" "EURUSD" "GBPUSD" "AUDUSD" "USDJPY" "SP500" "DJ30" "UK100" "FRA40" "USDCAD" "USDCHF")

START=$(date +%s)
for ASSET in "${ASSETS[@]}"; do
    T0=$(date +%s)
    echo ""
    echo "========================================"
    echo "  BUILD $ASSET V4 ($(date +%H:%M:%S))"
    echo "========================================"
    python3 -m bot_v2.build_v4_dataset "$ASSET" 2>&1 | tail -20
    T1=$(date +%s)
    DUR=$((T1 - T0))
    echo "Duree $ASSET : ${DUR}s"
done
TOTAL=$(date +%s)
echo ""
echo "========================================"
echo "  ALL BUILDS DONE in $((TOTAL - START))s"
echo "========================================"
ls -lh /workspace/TradingBot/data/ml_dataset_*_admiral_8ans_V4.parquet
