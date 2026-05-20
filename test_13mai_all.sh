#!/bin/bash
# Test bot live 13 mai sur 14 actifs
cd /workspace/TradingBot
ASSETS=(XAUUSD NAS100 GER40 BTCUSD EURUSD GBPUSD AUDUSD USDJPY SP500 DJ30 UK100 FRA40 USDCAD USDCHF)

for ASSET in "${ASSETS[@]}"; do
    echo ""
    echo "==================================================="
    echo "JOUR 2026-05-13 : $ASSET"
    echo "==================================================="
    python3 simulate_live_24h.py "$ASSET" --day 2026-05-13 2>&1 | grep -E "OBs jour J|max=|>= 0.75|>= 0.80|Trades pris|lots<=0_proba"
done
