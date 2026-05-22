"""Lance le backtest V12 ULTRA-REALISTE en parallele : 14 process, 1 par actif.

Chaque actif est independant cote pendings -> on peut tourner les 14 en
parallele sans interferer. Sur Vast 128c/512t, on parallelise sur 14 cores
=> backtest mois complet en ~5-15min (au lieu de 30-60min sequentiel).

Usage Vast :
    cd /workspace/TradingBot
    nohup python3 -u run_v12_backtest_parallel.py --start 2026-05-01 --end 2026-05-31 > /workspace/bt_v12.log 2>&1 & disown
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault("SWS_OVERRIDE", "1")
os.environ.setdefault("RR_OVERRIDE", "1.5")
os.environ.setdefault("BUILD_DATA_DIR", "data_vantage")

# detection chemin
ROOT = "c:/Users/Shadow/TradingBot" if sys.platform == "win32" else "/workspace/TradingBot"
sys.path.insert(0, ROOT)

import pandas as pd

LIVE_ASSETS = [
    "XAUUSD", "NAS100", "GER40", "BTCUSD",
    "EURUSD", "GBPUSD", "AUDUSD", "USDJPY",
    "SP500", "DJ30", "UK100", "FRA40",
    "USDCAD", "USDCHF",
]


def backtest_one_asset(args_tuple):
    """Worker : run le backtest pour un actif."""
    asset, start_str, end_str, step = args_tuple
    # Re-init env vars dans le worker (multiprocessing fork peut perdre)
    os.environ.setdefault("SWS_OVERRIDE", "1")
    os.environ.setdefault("RR_OVERRIDE", "1.5")
    os.environ.setdefault("BUILD_DATA_DIR", "data_vantage")
    import sys as _sys
    _sys.path.insert(0, ROOT)
    import pandas as _pd

    from backtest_v12_realistic import run_backtest

    start = _pd.Timestamp(start_str, tz="UTC")
    end = _pd.Timestamp(end_str, tz="UTC")
    if end.hour == 0 and end.minute == 0:
        end = end + _pd.Timedelta(hours=23, minutes=59)

    print(f"[WORKER {asset}] start", flush=True)
    t0 = time.time()
    trades, stats = run_backtest(start, end, [asset], scan_step_min=step)
    elapsed = time.time() - t0
    print(f"[WORKER {asset}] done in {elapsed/60:.1f}min, {stats['n_trades']} trades", flush=True)

    # Sauve les trades de ce worker
    out = f"{ROOT}/data/bt_v12_{asset}.csv"
    if trades:
        df = _pd.DataFrame([{
            "instrument": t.instrument, "direction": t.direction,
            "ob_ts": t.ob_validation_ts, "place_ts": t.pending_placed_ts,
            "fill_ts": t.fill_ts, "exit_ts": t.exit_ts,
            "outcome": t.outcome, "ml": t.ml_proba,
            "rr": t.rr, "pnl_r": t.pnl_r,
        } for t in trades])
        df.to_csv(out, index=False)
    return {
        "asset": asset, "n_trades": stats["n_trades"],
        "n_pendings": stats["n_pendings"], "elapsed_min": elapsed/60,
        "csv": out,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--step", type=int, default=1, help="Pas en minutes (default 1)")
    p.add_argument("--assets", nargs="+", default=None)
    p.add_argument("--workers", type=int, default=14,
                   help="Nb workers parallel (default 14 = 1 par actif)")
    args = p.parse_args()

    assets = args.assets or LIVE_ASSETS
    print(f"=== BACKTEST V12 PARALLELE ===", flush=True)
    print(f"Periode : {args.start} -> {args.end}", flush=True)
    print(f"Actifs : {assets}", flush=True)
    print(f"Workers : {args.workers}", flush=True)
    print(f"Step    : {args.step} min", flush=True)
    print(f"Cores dispo : {os.cpu_count()}", flush=True)
    print()

    tasks = [(a, args.start, args.end, args.step) for a in assets]
    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(backtest_one_asset, t): t[0] for t in tasks}
        done = 0
        for fut in as_completed(futures):
            asset = futures[fut]
            try:
                r = fut.result()
                results.append(r)
                done += 1
                print(f"[{done}/{len(tasks)}] {asset} OK : {r['n_trades']} trades en {r['elapsed_min']:.1f}min", flush=True)
            except Exception as e:
                print(f"[FAIL] {asset} : {e}", flush=True)
                import traceback; traceback.print_exc()

    elapsed = time.time() - t0
    print(f"\n=== TERMINE en {elapsed/60:.1f}min ===", flush=True)

    # Consolide tous les CSV en un seul recap
    all_trades = []
    for r in results:
        try:
            df = pd.read_csv(r["csv"])
            all_trades.append(df)
        except Exception:
            continue
    if all_trades:
        big = pd.concat(all_trades, ignore_index=True)
        out_all = f"{ROOT}/data/bt_v12_ALL.csv"
        big.to_csv(out_all, index=False)
        print(f"Recap consolide : {out_all} ({len(big)} trades)", flush=True)

        # Stats
        closed = big[big["outcome"].isin(["WIN", "LOSS"])]
        if len(closed):
            wr = (closed["outcome"] == "WIN").mean() * 100
            pnl_r = closed["pnl_r"].sum()
            print(f"\n=== STATS GLOBALES ===")
            print(f"  Trades fermes : {len(closed)}")
            print(f"  WR            : {wr:.1f}%")
            print(f"  PnL (R)       : {pnl_r:.1f}")
            print(f"  PnL/trade     : {pnl_r/len(closed):.3f} R")
            print(f"\n  Par actif :")
            for inst, sub in closed.groupby("instrument"):
                wr_a = (sub["outcome"] == "WIN").mean() * 100
                print(f"    {inst:8s} : {len(sub):3d} trades, WR={wr_a:.1f}%, PnL={sub['pnl_r'].sum():+.1f}R")


if __name__ == "__main__":
    main()
