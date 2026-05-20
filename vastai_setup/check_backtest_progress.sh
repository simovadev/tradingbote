#!/bin/bash
# Affiche la progression de chaque backtest en cours
cd /workspace/TradingBot
for f in logs/backtest_*.log; do
    if [ ! -f "$f" ]; then continue; fi
    asset=$(basename "$f" .log | sed 's/backtest_//')
    echo "=== $asset ==="
    tail -10 "$f" | grep -E 'OBs detectes|MSS detectes|Progression|RECAP|WIN  |LOSS |WR brut|PnL|exception|Trades pris'
    echo ""
done
