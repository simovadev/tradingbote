#!/bin/bash
# Replay V5 sur donnees Vantage pour 14 actifs en PARALLELE.
# Sur Vast.ai EPYC 96 cores : ~10-20 min total (vs ~3-4h sequentiel).
#
# Usage : bash vastai_setup/replay_v5_all.sh [--scan-every N] [--start YYYY-MM-DD]
#
# Defaut : scan toutes les 1 min, plage = toute la data Vantage dispo.

set -e
cd /workspace/TradingBot
export PYTHONPATH=/workspace/TradingBot

# Args optionnels passes au script Python
EXTRA_ARGS="$*"

ASSETS=("XAUUSD" "NAS100" "GER40" "BTCUSD" "EURUSD" "GBPUSD" "AUDUSD" "USDJPY" "SP500" "DJ30" "UK100" "FRA40" "USDCAD" "USDCHF")

# Verifie data_vantage
if [ ! -d "data_vantage" ]; then
    echo "!! data_vantage/ manquant. Upload via : scp -P PORT -r data_vantage root@IP:/workspace/TradingBot/"
    exit 1
fi
N_PARQUETS=$(ls data_vantage/*.parquet 2>/dev/null | wc -l)
echo "data_vantage/ : $N_PARQUETS parquets"
if [ "$N_PARQUETS" -lt 50 ]; then
    echo "!! Pas assez de parquets ($N_PARQUETS < 50)"
    exit 1
fi

# Verifie modeles V5
N_MODELS=$(ls bot_v2/ml_model_*_admiral_v5.pkl 2>/dev/null | wc -l)
echo "Modeles V5    : $N_MODELS"
if [ "$N_MODELS" -lt 14 ]; then
    echo "!! Modeles V5 manquants ($N_MODELS < 14)"
    exit 1
fi

mkdir -p logs
START=$(date +%s)

# Lance les 14 replays en parallele (background)
PIDS=()
for ASSET in "${ASSETS[@]}"; do
    LOG="logs/replay_${ASSET}.log"
    echo "==> Lance replay $ASSET (log: $LOG)"
    python3 replay_v5_vantage.py "$ASSET" $EXTRA_ARGS > "$LOG" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "14 replays lances en parallele (PIDs: ${PIDS[*]})"
echo "Attente fin..."

# Attendre tous
for PID in "${PIDS[@]}"; do
    wait $PID
done

TOTAL=$(($(date +%s) - START))
echo ""
echo "================================================================"
echo "  ALL REPLAYS DONE in ${TOTAL}s"
echo "================================================================"
echo ""

# Recap par actif
echo "RECAP PAR ACTIF :"
echo ""
printf "%-10s %8s %6s %6s %7s %10s\n" "ASSET" "PLACED" "WIN" "LOSS" "WR" "PnL\$"
echo "------------------------------------------------------------"
TOTAL_PLACED=0
TOTAL_WIN=0
TOTAL_LOSS=0
TOTAL_PNL=0
for ASSET in "${ASSETS[@]}"; do
    LOG="logs/replay_${ASSET}.log"
    if [ ! -f "$LOG" ]; then
        continue
    fi
    PLACED=$(grep -oE "Ordres places  : [0-9]+" "$LOG" | tail -1 | grep -oE "[0-9]+" || echo 0)
    WIN=$(grep -oE "WIN  : [0-9]+" "$LOG" | tail -1 | grep -oE "[0-9]+" || echo 0)
    LOSS=$(grep -oE "LOSS : [0-9]+" "$LOG" | tail -1 | grep -oE "[0-9]+" || echo 0)
    WR=$(grep -oE "WR   : [0-9.]+" "$LOG" | tail -1 | grep -oE "[0-9.]+" || echo 0)
    PNL=$(grep -oE "PnL  : [-+0-9.]+" "$LOG" | tail -1 | grep -oE "[-+0-9.]+" || echo 0)
    printf "%-10s %8d %6d %6d %6.1f%% %+10.2f\n" "$ASSET" "$PLACED" "$WIN" "$LOSS" "$WR" "$PNL"
    TOTAL_PLACED=$((TOTAL_PLACED + PLACED))
    TOTAL_WIN=$((TOTAL_WIN + WIN))
    TOTAL_LOSS=$((TOTAL_LOSS + LOSS))
    TOTAL_PNL=$(echo "$TOTAL_PNL + $PNL" | bc -l)
done
echo "------------------------------------------------------------"
if [ $((TOTAL_WIN + TOTAL_LOSS)) -gt 0 ]; then
    TOTAL_WR=$(echo "scale=1; $TOTAL_WIN * 100 / ($TOTAL_WIN + $TOTAL_LOSS)" | bc -l)
else
    TOTAL_WR=0
fi
printf "%-10s %8d %6d %6d %6s%% %+10.2f\n" "TOTAL" "$TOTAL_PLACED" "$TOTAL_WIN" "$TOTAL_LOSS" "$TOTAL_WR" "$TOTAL_PNL"

echo ""
echo "Trades csv par actif : replay_v5_vantage_*.csv"
