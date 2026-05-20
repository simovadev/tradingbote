#!/bin/bash
cd /workspace/TradingBot
echo ""
printf "%-10s %7s %5s %5s %6s %10s %7s\n" "ASSET" "TRADES" "WIN" "LOSS" "WR" "PnL" "MaxP"
echo "-----------------------------------------------------------------"
for ASSET in XAUUSD NAS100 GER40 BTCUSD EURUSD GBPUSD AUDUSD USDJPY SP500 DJ30 UK100 FRA40 USDCAD USDCHF; do
    LOG="logs/simulate_${ASSET}.log"
    [ ! -f "$LOG" ] && continue
    TRADES=$(grep "Trades pris" "$LOG" | tail -1 | grep -oE "[0-9]+$")
    # WIN/LOSS : 1/2 -> 1 WIN, 2 LOSS
    WL=$(grep "WIN/LOSS" "$LOG" | tail -1 | grep -oE "[0-9]+/[0-9]+")
    WIN=$(echo "$WL" | awk -F'/' '{print $1}')
    LOSS=$(echo "$WL" | awk -F'/' '{print $2}')
    WR=$(grep "WR :" "$LOG" | tail -1 | grep -oE "[0-9.]+")
    PNL=$(grep "PnL :" "$LOG" | tail -1 | grep -oE "[-+0-9.]+")
    MAXP=$(grep "Max proba" "$LOG" | tail -1 | grep -oE "Max proba : [0-9.]+" | grep -oE "[0-9.]+$")
    printf "%-10s %7s %5s %5s %5s%% %+10s %7s\n" "$ASSET" "$TRADES" "$WIN" "$LOSS" "$WR" "$PNL" "$MAXP"
done
