#!/bin/bash
# Watch progression build V4 par actif.
# Usage : bash vastai_setup/watch_v4.sh

ASSETS=(XAUUSD NAS100 GER40 BTCUSD EURUSD GBPUSD AUDUSD USDJPY SP500 DJ30 UK100 FRA40 USDCAD USDCHF)
PARTIAL_DIR=/workspace/TradingBot/data/ml_partial_M1

while true; do
    clear
    echo "=== Progression V4 par actif ==="
    for a in "${ASSETS[@]}"; do
        n=$(ls "$PARTIAL_DIR"/${a}_M1_*.parquet 2>/dev/null | wc -l)
        printf "  %-8s : %3d chunks\n" "$a" "$n"
    done
    echo ""
    date
    sleep 30
done
