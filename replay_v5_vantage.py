"""Replay V5 sur donnees Vantage : simule le bot live bougie par bougie.

Reutilise EXACTEMENT le meme code que live_runner_v2 (scan_asset, place_order, etc.)
mais avec un MT5ExecutorReplay qui sert les bougies Vantage depuis parquets.

Resultat = identique a ce que le bot aurait fait s'il avait tourne en direct ces 7 mois.

Usage:
    python replay_v5_vantage.py XAUUSD                   # un actif
    python replay_v5_vantage.py XAUUSD --start 2026-04   # depuis avril
    python replay_v5_vantage.py --all                    # 14 actifs (long)
"""
import os
import sys
import time
import argparse
import logging
from pathlib import Path

# FIX (2026-05-20) : limiter les threads par process pour permettre 14 replays en parallele
# Sans ça, chaque LightGBM/OpenMP cree 96 threads -> 14*96=1344 threads -> crash "Resource temporarily unavailable"
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("LIGHTGBM_NUM_THREADS", "2")

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from bot_v2.mt5_executor_replay import MT5ExecutorReplay
from bot_v2.live_state import LiveState
from bot_v2.live_runner_v2 import scan_asset
from bot_v2 import live_runner_v2

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
# Reduire le bruit du live_runner
logging.getLogger("bot_v2.live_runner_v2").setLevel(logging.WARNING)


ALL_ASSETS = [
    "XAUUSD", "NAS100", "GER40", "BTCUSD",
    "EURUSD", "GBPUSD", "AUDUSD", "USDJPY",
    "SP500", "DJ30", "UK100", "FRA40",
    "USDCAD", "USDCHF",
]


def replay_asset(asset: str, start_ts: pd.Timestamp = None, end_ts: pd.Timestamp = None,
                 scan_every_min: int = 1):
    """Replay live V5 pour 1 actif sur la plage [start, end].

    scan_every_min : intervalle entre 2 scans (1 = chaque minute, 5 = toutes les 5 min).
    Plus haut = plus rapide mais peut manquer des setups si recent_cutoff=3min.
    """
    print(f"\n{'='*70}", flush=True)
    print(f"REPLAY LIVE V5 : {asset}", flush=True)
    print(f"{'='*70}", flush=True)

    # Init mock executor
    mt5 = MT5ExecutorReplay(data_dir="data_vantage", root=ROOT)
    mt5.initialize()

    # Charge le parquet M1 pour determiner la plage
    df_m1 = mt5._load_df(asset, "M1")
    if df_m1 is None:
        print(f"  KO: data_vantage/{asset}_M1.parquet manquant", flush=True)
        return None
    print(f"  M1 dispo : {df_m1.index[0]} -> {df_m1.index[-1]} ({len(df_m1):,} bougies)", flush=True)

    # Plage replay
    full_start = df_m1.index[0] + pd.Timedelta(hours=24)  # +24h pour avoir HTF
    full_end = df_m1.index[-1]
    if start_ts is None:
        start_ts = full_start
    if end_ts is None:
        end_ts = full_end
    start_ts = max(start_ts, full_start)
    end_ts = min(end_ts, full_end)
    print(f"  Replay   : {start_ts} -> {end_ts}", flush=True)
    print(f"  Scan toutes les {scan_every_min} min", flush=True)

    # Override BOT_START_TS pour eviter le rejet pre-boot
    # On simule un demarrage 1 min avant start_ts
    live_runner_v2.BOT_START_TS = start_ts - pd.Timedelta(minutes=1)

    state = LiveState()
    balance = 154.96  # 104.96 + bonus 50

    # Boucle de scan
    scan_count = 0
    setups_placed = 0
    t_real_start = time.time()
    t_last_log = t_real_start

    # On avance par scan_every_min minutes
    now = start_ts
    while now <= end_ts:
        mt5.set_now(now)
        mt5.update_orders()  # check fills/SL/TP avant scan

        try:
            setups = scan_asset(mt5, asset, state, balance=balance, debug_diag=False)
        except Exception as e:
            log.error(f"scan_asset fail at {now}: {e}")
            now += pd.Timedelta(minutes=scan_every_min)
            continue

        for s in setups:
            # On verifie : ts >= BOT_START_TS (deja gere par live_runner)
            # On place l'ordre
            r = s["r"]
            setup = r.trade_setup
            order_type = "BUY_LIMIT" if s["ob"].direction == "bullish" else "SELL_LIMIT"

            # Calcule lots avec compute_position_size
            from bot_v2.trade_setup import compute_position_size
            lots, risk_usd = compute_position_size(
                setup.entry_price, setup.stop_loss, asset,
                balance=balance, risk_pct=0.01,
            )
            if lots <= 0:
                continue

            mt5.place_limit_order(
                symbol=asset, order_type=order_type,
                entry=setup.entry_price, sl=setup.stop_loss, tp=setup.take_profit,
                lots=lots, magic=20260520, comment=f"V5_p{s['proba']:.3f}",
            )
            setups_placed += 1

        scan_count += 1
        now += pd.Timedelta(minutes=scan_every_min)

        if time.time() - t_last_log > 30:
            done_pct = (now - start_ts).total_seconds() / (end_ts - start_ts).total_seconds() * 100
            stats = mt5.stats_recap()
            print(f"  {now} ({done_pct:.0f}%) | scans={scan_count} placed={setups_placed} | "
                  f"filled={stats['filled']} closed={stats['closed']} W={stats['wins']} L={stats['losses']} "
                  f"WR={stats['wr']:.1f}% PnL={stats['pnl_usd']:+.2f}", flush=True)
            t_last_log = time.time()

    # Final update
    mt5.update_orders()

    duration = time.time() - t_real_start
    stats = mt5.stats_recap()
    period_days = (end_ts - start_ts).days
    print(f"\n  RECAP REPLAY {asset} ({duration:.0f}s reel) :", flush=True)
    print(f"    Periode replay : {start_ts.date()} -> {end_ts.date()} ({period_days}j)", flush=True)
    print(f"    Scans          : {scan_count:,}", flush=True)
    print(f"    Ordres places  : {setups_placed}", flush=True)
    print(f"    Filled         : {stats['filled']}", flush=True)
    print(f"    Closed         : {stats['closed']}", flush=True)
    print(f"    Cancelled      : {stats['cancelled']}", flush=True)
    print(f"    WIN  : {stats['wins']}", flush=True)
    print(f"    LOSS : {stats['losses']}", flush=True)
    print(f"    WR   : {stats['wr']:.1f}%", flush=True)
    print(f"    PnL  : {stats['pnl_usd']:+.2f} USD", flush=True)
    print(f"    Trades/jour : {setups_placed/period_days:.2f}", flush=True)

    # Save trades
    rows = []
    for o in mt5._orders:
        rows.append({
            "ticket": o.ticket, "symbol": o.symbol, "direction": o.direction,
            "type": o.type, "entry": o.entry, "sl": o.sl, "tp": o.tp, "lots": o.lots,
            "time_setup": o.time_setup, "state": o.state,
            "fill_time": o.fill_time, "fill_price": o.fill_price,
            "close_time": o.close_time, "close_price": o.close_price,
            "outcome": o.outcome, "pnl_usd": o.pnl_usd, "comment": o.comment,
        })
    if rows:
        df_orders = pd.DataFrame(rows)
        out_csv = Path(ROOT) / f"replay_v5_vantage_{asset}.csv"
        df_orders.to_csv(out_csv, index=False)
        print(f"    Orders sauves  : {out_csv.name}", flush=True)

    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("asset", nargs="?", help="Asset name ou --all")
    p.add_argument("--all", action="store_true")
    p.add_argument("--start", help="Date debut (YYYY-MM-DD ou YYYY-MM)", default=None)
    p.add_argument("--end", help="Date fin (YYYY-MM-DD)", default=None)
    p.add_argument("--scan-every", type=int, default=1, help="Scan every N min (1=chaque min, 5=toutes les 5min)")
    args = p.parse_args()

    start_ts = pd.Timestamp(args.start, tz="UTC") if args.start else None
    end_ts = pd.Timestamp(args.end, tz="UTC") if args.end else None

    assets = ALL_ASSETS if (args.all or args.asset == "--all") else [args.asset]

    all_stats = {}
    t_global = time.time()
    for a in assets:
        if a not in ALL_ASSETS:
            print(f"!! Actif inconnu : {a}", flush=True)
            continue
        try:
            all_stats[a] = replay_asset(a, start_ts=start_ts, end_ts=end_ts,
                                        scan_every_min=args.scan_every)
        except Exception as e:
            print(f"!! FAIL {a}: {e}", flush=True)
            import traceback; traceback.print_exc()

    if len(all_stats) > 1:
        print(f"\n\n{'='*90}", flush=True)
        print(f"RECAP GLOBAL REPLAY V5 VANTAGE ({time.time()-t_global:.0f}s)", flush=True)
        print(f"{'='*90}", flush=True)
        print(f"{'ASSET':<10} {'PLACED':>8} {'CLOSED':>8} {'WIN':>5} {'LOSS':>5} {'WR':>7} {'PnL$':>10}", flush=True)
        print("-"*90, flush=True)
        total = {"placed": 0, "wins": 0, "losses": 0, "pnl": 0}
        for a, s in all_stats.items():
            if s is None:
                continue
            print(f"{a:<10} {s['total']:>8} {s['closed']:>8} {s['wins']:>5} {s['losses']:>5} {s['wr']:>6.1f}% {s['pnl_usd']:>+10.2f}", flush=True)
            total["placed"] += s['total']
            total["wins"] += s['wins']
            total["losses"] += s['losses']
            total["pnl"] += s['pnl_usd']
        print("-"*90, flush=True)
        total_wr = total["wins"] / (total["wins"] + total["losses"]) * 100 if (total["wins"] + total["losses"]) > 0 else 0
        print(f"{'TOTAL':<10} {total['placed']:>8} {'':>8} {total['wins']:>5} {total['losses']:>5} {total_wr:>6.1f}% {total['pnl']:>+10.2f}", flush=True)


if __name__ == "__main__":
    main()
