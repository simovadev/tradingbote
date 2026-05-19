#!/bin/bash
# Build + Train V3.5 pour les 13 actifs (XAUUSD deja fait)
# Usage : bash build_train_13.sh

cd /workspace/TradingBot
export PYTHONPATH=/workspace/TradingBot

# Ordre : les + petits d'abord pour validation rapide
ASSETS=("EURUSD" "GBPUSD" "USDJPY" "AUDUSD" "USDCAD" "USDCHF" "FRA40" "UK100" "GER40" "SP500" "NAS100" "DJ30" "BTCUSD")

START=$(date +%s)
echo "=== BUILD + TRAIN ${#ASSETS[@]} ACTIFS V3.5 ==="
echo "Debut : $(date)"
echo ""

for ASSET in "${ASSETS[@]}"; do
    T0=$(date +%s)
    echo ""
    echo "========================================"
    echo "  [$ASSET] BUILD ($(date +%H:%M:%S))"
    echo "========================================"
    BUILD_SCRIPT="bot_v2/ml_dataset_${ASSET}_admiral_8ans_V3_5.py"
    TRAIN_SCRIPT="bot_v2/ml_train_${ASSET}_admiral_8ans_V3_5.py"

    if [ ! -f "$BUILD_SCRIPT" ]; then
        echo "SKIP : $BUILD_SCRIPT introuvable"
        continue
    fi

    # Build dataset (64 workers + full output pour voir chunks)
    N_WORKERS=64 python3 -u "$BUILD_SCRIPT" 2>&1 | grep -E 'Cores|Decoupage|chunks|workers|XAUUSD|EURUSD|GBPUSD|USDJPY|USDCAD|USDCHF|AUDUSD|GER40|UK100|FRA40|SP500|NAS100|DJ30|BTCUSD|JP225|XAGUSD|Dataset' | tail -30
    DUR_BUILD=$(($(date +%s) - T0))
    echo ""
    echo "  Build $ASSET : ${DUR_BUILD}s"

    # Train si dataset present
    DATASET="data/ml_dataset_${ASSET}_admiral_8ans_V3_5.parquet"
    if [ -f "$DATASET" ]; then
        echo "  [$ASSET] TRAIN ($(date +%H:%M:%S))"
        python3 -u "$TRAIN_SCRIPT" 2>&1 | tail -20
    else
        echo "  SKIP train : $DATASET non trouve"
    fi

    DUR_TOTAL=$(($(date +%s) - T0))
    echo "  TOTAL $ASSET : ${DUR_TOTAL}s"
done

TOTAL=$(($(date +%s) - START))
echo ""
echo "========================================"
echo "  ALL DONE in ${TOTAL}s ($(date))"
echo "========================================"
ls -lh /workspace/TradingBot/data/ml_dataset_*_admiral_8ans_V3_5.parquet | awk '{print $5, $9}'
echo ""
ls -lh /workspace/TradingBot/bot_v2/ml_model_*_admiral_v3_5.pkl | awk '{print $5, $9}'
