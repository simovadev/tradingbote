#!/bin/bash
# Test bot live 13 mai sur 14 actifs EN PARALLELE (~60s)
cd /workspace/TradingBot
ASSETS=(XAUUSD NAS100 GER40 BTCUSD EURUSD GBPUSD AUDUSD USDJPY SP500 DJ30 UK100 FRA40 USDCAD USDCHF)

mkdir -p logs
START=$(date +%s)

PIDS=()
for ASSET in "${ASSETS[@]}"; do
    LOG="logs/test13mai_${ASSET}.log"
    OMP_NUM_THREADS=2 LIGHTGBM_NUM_THREADS=2 python3 simulate_live_24h.py "$ASSET" --day 2026-05-13 > "$LOG" 2>&1 &
    PIDS+=($!)
done

echo "14 tests lances en parallele..."
for PID in "${PIDS[@]}"; do
    wait $PID
done
TOTAL=$(($(date +%s) - START))
echo ""
echo "DONE in ${TOTAL}s"
echo ""

# Recap
printf "%-10s %5s %5s %5s %6s %7s %6s\n" "ASSET" "OBs" "ML" "TRADE" "MaxP" "WIN/LO" "WR"
echo "------------------------------------------------------"
for ASSET in "${ASSETS[@]}"; do
    LOG="logs/test13mai_${ASSET}.log"
    [ ! -f "$LOG" ] && continue
    OBS=$(grep "OBs evalues" "$LOG" | tail -1 | grep -oE "[0-9]+$")
    ML=$(grep "OBs passe pipeline" "$LOG" | tail -1 | grep -oE "[0-9]+$")
    TRADES=$(grep "Trades pris" "$LOG" | tail -1 | grep -oE "[0-9]+$")
    MAXP=$(grep "max=" "$LOG" | tail -1 | grep -oE "max=[0-9.]+" | grep -oE "[0-9.]+")
    WL=$(grep -E "WIN  :|WIN.: " "$LOG" | tail -1)
    printf "%-10s %5s %5s %5s %6s %7s\n" "$ASSET" "${OBS:-0}" "${ML:-0}" "${TRADES:-0}" "${MAXP:-?}" "${WL:-0/0}"
done
