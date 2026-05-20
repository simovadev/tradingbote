#!/bin/bash
# Simulate live week parallele : 14 actifs en parallele en background
# Avec 192 threads, on alloue 13 threads par actif (14 * 13 = 182)

set -e
cd /workspace/TradingBot
export PYTHONPATH=/workspace/TradingBot

ASSETS=("XAUUSD" "NAS100" "GER40" "BTCUSD" "EURUSD" "GBPUSD" "AUDUSD" "USDJPY" "SP500" "DJ30" "UK100" "FRA40" "USDCAD" "USDCHF")

mkdir -p logs
START=$(date +%s)

PIDS=()
for ASSET in "${ASSETS[@]}"; do
    LOG="logs/simulate_${ASSET}.log"
    echo "==> Lance $ASSET en background (log: $LOG)"
    OMP_NUM_THREADS=2 LIGHTGBM_NUM_THREADS=2 python3 simulate_live_week.py "$ASSET" > "$LOG" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "14 simulations lancees (PIDs: ${PIDS[*]})"

# Wait all
for PID in "${PIDS[@]}"; do
    wait $PID
done

TOTAL=$(($(date +%s) - START))
echo ""
echo "================================================================"
echo "  ALL SIMULATE DONE in ${TOTAL}s"
echo "================================================================"
echo ""

# Recap
printf "%-10s %7s %5s %5s %6s %10s\n" "ASSET" "TRADES" "WIN" "LOSS" "WR" "PnL\$"
echo "----------------------------------------------------------------------"
TOT_T=0; TOT_W=0; TOT_L=0; TOT_PNL=0
for ASSET in "${ASSETS[@]}"; do
    LOG="logs/simulate_${ASSET}.log"
    [ ! -f "$LOG" ] && continue
    T=$(grep -oE "Trades pris.*: [0-9]+" "$LOG" | tail -1 | grep -oE "[0-9]+$" || echo 0)
    W=$(grep -oE "WIN/LOSS : [0-9]+" "$LOG" | tail -1 | grep -oE "[0-9]+$" || echo 0)
    L=$(grep -oE "WIN/LOSS : [0-9]+/[0-9]+" "$LOG" | tail -1 | awk -F'/' '{print $2}' | grep -oE "[0-9]+" || echo 0)
    WR=$(grep -oE "WR : [0-9.]+" "$LOG" | tail -1 | grep -oE "[0-9.]+" || echo 0)
    PNL=$(grep -oE "PnL : [-+0-9.]+" "$LOG" | tail -1 | grep -oE "[-+0-9.]+" || echo 0)
    printf "%-10s %7d %5d %5d %5.1f%% %+10s\n" "$ASSET" "$T" "$W" "$L" "$WR" "$PNL"
    TOT_T=$((TOT_T + T))
    TOT_W=$((TOT_W + W))
    TOT_L=$((TOT_L + L))
    TOT_PNL=$(echo "$TOT_PNL + $PNL" | bc -l)
done
echo "----------------------------------------------------------------------"
TOT_WR=0
[ $((TOT_W + TOT_L)) -gt 0 ] && TOT_WR=$(echo "scale=1; $TOT_W * 100 / ($TOT_W + $TOT_L)" | bc -l)
printf "%-10s %7d %5d %5d %5s%% %+10s\n" "TOTAL" "$TOT_T" "$TOT_W" "$TOT_L" "$TOT_WR" "$TOT_PNL"
echo ""
echo "Trades/jour total : $(echo "scale=1; $TOT_T / 7" | bc -l)"
