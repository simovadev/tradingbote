#!/bin/bash
# Backtest V5 batch (rapide) sur donnees Vantage pour 14 actifs en PARALLELE.
# Bcp plus rapide que replay (detecte les OBs en batch, pas scan-par-scan).
# Sur EPYC 96c : ~10-15 min total.

set -e
cd /workspace/TradingBot
export PYTHONPATH=/workspace/TradingBot

ASSETS=("XAUUSD" "NAS100" "GER40" "BTCUSD" "EURUSD" "GBPUSD" "AUDUSD" "USDJPY" "SP500" "DJ30" "UK100" "FRA40" "USDCAD" "USDCHF")

if [ ! -d "data_vantage" ]; then
    echo "!! data_vantage/ manquant"
    exit 1
fi

mkdir -p logs
START=$(date +%s)

# Lance les 14 backtests en parallele
PIDS=()
for ASSET in "${ASSETS[@]}"; do
    LOG="logs/backtest_${ASSET}.log"
    echo "==> Lance backtest $ASSET (log: $LOG)"
    python3 backtest_v5_vantage.py "$ASSET" > "$LOG" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "14 backtests lances en parallele (PIDs: ${PIDS[*]})"
echo "Attente fin..."

# Affiche progression toutes les 30s
(
    while true; do
        sleep 30
        N_DONE=0
        for ASSET in "${ASSETS[@]}"; do
            if grep -q "RECAP" "logs/backtest_${ASSET}.log" 2>/dev/null; then
                N_DONE=$((N_DONE + 1))
            fi
        done
        echo "  Progression : $N_DONE/14 actifs termines ($(date +%H:%M:%S))"
        if [ "$N_DONE" -eq 14 ]; then break; fi
    done
) &
WATCHER_PID=$!

# Attendre tous
for PID in "${PIDS[@]}"; do
    wait $PID
done
kill $WATCHER_PID 2>/dev/null || true

TOTAL=$(($(date +%s) - START))
echo ""
echo "================================================================"
echo "  ALL BACKTESTS DONE in ${TOTAL}s"
echo "================================================================"
echo ""

# Recap
echo "RECAP PAR ACTIF :"
printf "%-10s %8s %6s %6s %7s %10s %8s\n" "ASSET" "TRADES" "WIN" "LOSS" "WR" "PnL\$" "TR/J"
echo "------------------------------------------------------------------------"
TOTAL_TRADES=0
TOTAL_WIN=0
TOTAL_LOSS=0
TOTAL_PNL=0
for ASSET in "${ASSETS[@]}"; do
    LOG="logs/backtest_${ASSET}.log"
    if [ ! -f "$LOG" ]; then continue; fi
    TRADES=$(grep -oE "Trades pris.*: [0-9]+" "$LOG" | tail -1 | grep -oE "[0-9]+$" || echo 0)
    WIN=$(grep -oE "WIN  : [0-9]+" "$LOG" | tail -1 | grep -oE "[0-9]+" || echo 0)
    LOSS=$(grep -oE "LOSS : [0-9]+" "$LOG" | tail -1 | grep -oE "[0-9]+" || echo 0)
    WR=$(grep -oE "WR brut : [0-9.]+" "$LOG" | tail -1 | grep -oE "[0-9.]+" || echo 0)
    PNL=$(grep -oE "PnL total : [-+0-9.]+" "$LOG" | tail -1 | grep -oE "[-+0-9.]+" || echo 0)
    TRJ=$(grep -oE "Trades/jour : [0-9.]+" "$LOG" | tail -1 | grep -oE "[0-9.]+" || echo 0)
    printf "%-10s %8d %6d %6d %6s%% %+10s %8s\n" "$ASSET" "$TRADES" "$WIN" "$LOSS" "$WR" "$PNL" "$TRJ"
    TOTAL_TRADES=$((TOTAL_TRADES + TRADES))
    TOTAL_WIN=$((TOTAL_WIN + WIN))
    TOTAL_LOSS=$((TOTAL_LOSS + LOSS))
    TOTAL_PNL=$(echo "$TOTAL_PNL + $PNL" | bc -l)
done
echo "------------------------------------------------------------------------"
if [ $((TOTAL_WIN + TOTAL_LOSS)) -gt 0 ]; then
    TOTAL_WR=$(echo "scale=1; $TOTAL_WIN * 100 / ($TOTAL_WIN + $TOTAL_LOSS)" | bc -l)
else
    TOTAL_WR=0
fi
printf "%-10s %8d %6d %6d %6s%% %+10s\n" "TOTAL" "$TOTAL_TRADES" "$TOTAL_WIN" "$TOTAL_LOSS" "$TOTAL_WR" "$TOTAL_PNL"
