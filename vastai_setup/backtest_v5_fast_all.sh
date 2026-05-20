#!/bin/bash
# Backtest V5 FAST sur 14 actifs en parallele.
# Utilise fenetre live (2000 M1) au lieu de tout l'historique -> 100x plus rapide.

set -e
cd /workspace/TradingBot
export PYTHONPATH=/workspace/TradingBot

ASSETS=("XAUUSD" "NAS100" "GER40" "BTCUSD" "EURUSD" "GBPUSD" "AUDUSD" "USDJPY" "SP500" "DJ30" "UK100" "FRA40" "USDCAD" "USDCHF")

mkdir -p logs
START=$(date +%s)

PIDS=()
for ASSET in "${ASSETS[@]}"; do
    LOG="logs/backtest_${ASSET}.log"
    python3 backtest_v5_vantage_fast.py "$ASSET" > "$LOG" 2>&1 &
    PIDS+=($!)
    echo "==> $ASSET lance"
done

echo ""
echo "14 backtests FAST en parallele (PIDs: ${PIDS[*]})"

# Watcher
(
    while true; do
        sleep 30
        N_DONE=0
        for ASSET in "${ASSETS[@]}"; do
            if grep -q "RECAP" "logs/backtest_${ASSET}.log" 2>/dev/null; then
                N_DONE=$((N_DONE + 1))
            fi
        done
        echo "  [$(date +%H:%M:%S)] $N_DONE/14 termines"
        if [ "$N_DONE" -eq 14 ]; then break; fi
    done
) &
WATCHER_PID=$!

for PID in "${PIDS[@]}"; do
    wait $PID
done
kill $WATCHER_PID 2>/dev/null || true

TOTAL=$(($(date +%s) - START))
echo ""
echo "================================================================"
echo "  ALL BACKTESTS FAST DONE in ${TOTAL}s"
echo "================================================================"
echo ""

# Recap
printf "%-10s %8s %5s %5s %7s %10s %7s %6s\n" "ASSET" "TRADES" "WIN" "LOSS" "WR" "PnL\$" "PnL%" "TR/J"
echo "----------------------------------------------------------------------"
TOTAL_TR=0; TOTAL_W=0; TOTAL_L=0; TOTAL_PNL=0
for ASSET in "${ASSETS[@]}"; do
    LOG="logs/backtest_${ASSET}.log"
    if [ ! -f "$LOG" ]; then continue; fi
    TR=$(grep -oE "Trades pris.*: [0-9]+" "$LOG" | tail -1 | grep -oE "[0-9]+$" || echo 0)
    W=$(grep -oE "WIN  : [0-9]+" "$LOG" | tail -1 | grep -oE "[0-9]+" || echo 0)
    L=$(grep -oE "LOSS : [0-9]+" "$LOG" | tail -1 | grep -oE "[0-9]+" || echo 0)
    WR=$(grep -oE "WR brut : [0-9.]+" "$LOG" | tail -1 | grep -oE "[0-9.]+" || echo 0)
    PNL=$(grep -oE "PnL total : [-+0-9.]+" "$LOG" | tail -1 | grep -oE "[-+0-9.]+" || echo 0)
    PCT=$(grep -oE "PnL % : [-+0-9.]+" "$LOG" | tail -1 | grep -oE "[-+0-9.]+" || echo 0)
    TRJ=$(grep -oE "Trades/jour : [0-9.]+" "$LOG" | tail -1 | grep -oE "[0-9.]+" || echo 0)
    printf "%-10s %8d %5d %5d %6s%% %+10s %+6s%% %6s\n" "$ASSET" "$TR" "$W" "$L" "$WR" "$PNL" "$PCT" "$TRJ"
    TOTAL_TR=$((TOTAL_TR + TR))
    TOTAL_W=$((TOTAL_W + W))
    TOTAL_L=$((TOTAL_L + L))
    TOTAL_PNL=$(echo "$TOTAL_PNL + $PNL" | bc -l)
done
echo "----------------------------------------------------------------------"
TOTAL_WR=0
if [ $((TOTAL_W + TOTAL_L)) -gt 0 ]; then
    TOTAL_WR=$(echo "scale=1; $TOTAL_W * 100 / ($TOTAL_W + $TOTAL_L)" | bc -l)
fi
printf "%-10s %8d %5d %5d %6s%% %+10s\n" "TOTAL" "$TOTAL_TR" "$TOTAL_W" "$TOTAL_L" "$TOTAL_WR" "$TOTAL_PNL"
