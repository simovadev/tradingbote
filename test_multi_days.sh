#!/bin/bash
# Test simulation sur 5 jours differents
cd /workspace/TradingBot
source venv/bin/activate
for d in 2026-01-13 2026-01-22 2026-02-03 2026-02-10 2026-02-24; do
    echo ""
    echo "============================================"
    echo "JOUR: $d"
    echo "============================================"
    python3 simulate_live_24h.py XAUUSD --day "$d" 2>&1 | grep -E "OBs jour J|min=|max=|>= 0.75|>= 0.70|>= 0.65|Trades pris" | head -10
done
