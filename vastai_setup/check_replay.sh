#!/bin/bash
# Check status des 14 replays
cd /workspace/TradingBot
ASSETS=(XAUUSD NAS100 GER40 BTCUSD EURUSD GBPUSD AUDUSD USDJPY SP500 DJ30 UK100 FRA40 USDCAD USDCHF)
for asset in "${ASSETS[@]}"; do
    log="logs/replay_${asset}.log"
    if [ ! -f "$log" ]; then continue; fi
    echo "=== $asset ==="
    tail -15 "$log" | grep -E 'RECAP|Ordres places|Filled|Closed|WIN|LOSS|WR|PnL|Resource|Aborted|crash|terminate'
    echo ""
done
