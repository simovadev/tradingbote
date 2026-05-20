#!/bin/bash
# Build V5 14 actifs EN PARALLELE (vs build_v5_all.sh = sequentiel)
# Sur EPYC 96c/192t avec ~700GB RAM : 14 process en parallele, 13 workers chacun = 182 threads
# Estimation : ~3-5 min total (vs 16 min sequentiel)

set -e
cd /workspace/TradingBot
export PYTHONPATH=/workspace/TradingBot

ASSETS=("XAUUSD" "NAS100" "GER40" "BTCUSD" "EURUSD" "GBPUSD" "AUDUSD" "USDJPY" "SP500" "DJ30" "UK100" "FRA40" "USDCAD" "USDCHF")

mkdir -p logs
START=$(date +%s)

# Lance les 14 actifs en background (chaque process internal = 13 workers)
PIDS=()
for ASSET in "${ASSETS[@]}"; do
    LOG="logs/build_${ASSET}.log"
    echo "==> Lance $ASSET en background (log: $LOG)"
    N_WORKERS=13 python3 -m bot_v2.build_v5_dataset "$ASSET" > "$LOG" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "14 builds lances en parallele (PIDs: ${PIDS[*]})"
echo "Attente fin..."

# Watcher : affiche progression toutes les 30s
(
    while true; do
        sleep 30
        N_DONE=0
        for ASSET in "${ASSETS[@]}"; do
            if grep -q "RECAP.*candidats totaux" "logs/build_${ASSET}.log" 2>/dev/null; then
                N_DONE=$((N_DONE + 1))
            fi
        done
        echo "  [$(date +%H:%M:%S)] $N_DONE/14 actifs termines"
        if [ "$N_DONE" -eq 14 ]; then break; fi
    done
) &
WATCHER_PID=$!

# Attendre tous les builds
for PID in "${PIDS[@]}"; do
    wait $PID
done
kill $WATCHER_PID 2>/dev/null || true

TOTAL=$(($(date +%s) - START))
echo ""
echo "================================================================"
echo "  ALL BUILDS V5 PARALLELE DONE in ${TOTAL}s"
echo "================================================================"
echo ""

# Recap par actif
for ASSET in "${ASSETS[@]}"; do
    LOG="logs/build_${ASSET}.log"
    RECAP=$(grep "RECAP.*candidats totaux" "$LOG" 2>/dev/null | head -1)
    if [ -n "$RECAP" ]; then
        echo "  $RECAP"
    else
        echo "  KO: $ASSET"
    fi
done

echo ""
ls -lh /workspace/TradingBot/data/ml_dataset_*_admiral_8ans_V5.parquet
