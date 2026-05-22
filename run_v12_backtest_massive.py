"""Backtest V12 ULTRA-REALISTE - VERSION PARALLELISATION MASSIVE.

Au lieu de 14 workers (1 par actif), on decoupe chaque actif en N sous-periodes
de ~3-5 jours. Avec 14 actifs x 7 sous-periodes = 98 taches paralleles.
Sur Vast 128 cores physiques (256 threads), on tourne avec 96-128 workers.

Gain attendu : x7 sur la duree totale du backtest.

Subtilite : chaque sous-periode commence avec un buffer "warmup" de 30 jours
amont pour avoir les memes contextes que le live (M1=88k, M15=11k, etc.).

Pendings : un pending ouvert a la fin d'une sous-periode est "perdu" (compte
OPEN). En pratique, comme l'expiration est 60min, l'impact est marginal.

Usage Vast :
    cd /workspace/TradingBot
    nohup python3 -u run_v12_backtest_massive.py --start 2026-05-01 --end 2026-05-31 \
        --period_days 4 --workers 96 > /workspace/bt_v12.log 2>&1 & disown
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

ROOT = "c:/Users/Shadow/TradingBot" if sys.platform == "win32" else "/workspace/TradingBot"
sys.path.insert(0, ROOT)

import pandas as pd

LIVE_ASSETS = [
    "XAUUSD", "NAS100", "GER40", "BTCUSD",
    "EURUSD", "GBPUSD", "AUDUSD", "USDJPY",
    "SP500", "DJ30", "UK100", "FRA40",
    "USDCAD", "USDCHF",
]


def backtest_task(args_tuple):
    """Worker : run backtest pour 1 actif x 1 sous-periode."""
    asset, start_str, end_str, step, task_id = args_tuple
    os.environ.setdefault("SWS_OVERRIDE", "1")
    os.environ.setdefault("RR_OVERRIDE", "1.5")
    os.environ.setdefault("BUILD_DATA_DIR", "data_vantage")
    import sys as _sys
    _sys.path.insert(0, ROOT)
    import pandas as _pd
    import time as _t

    from backtest_v12_realistic import run_backtest

    start = _pd.Timestamp(start_str, tz="UTC")
    end = _pd.Timestamp(end_str, tz="UTC")
    if end.hour == 0 and end.minute == 0:
        end = end + _pd.Timedelta(hours=23, minutes=59)

    t0 = _t.time()
    try:
        trades, stats = run_backtest(start, end, [asset], scan_step_min=step)
    except Exception as e:
        import traceback
        return {"task_id": task_id, "asset": asset, "start": start_str, "end": end_str,
                "ok": False, "error": str(e), "trace": traceback.format_exc()[-500:]}
    elapsed = _t.time() - t0

    # Sauve CSV (1 par tache)
    out = f"{ROOT}/data/bt_v12_{asset}_{task_id:03d}.csv"
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
        "task_id": task_id, "asset": asset, "start": start_str, "end": end_str,
        "ok": True, "n_trades": stats["n_trades"], "n_pendings": stats["n_pendings"],
        "elapsed_s": elapsed, "csv": out if trades else None,
    }


def split_period(start: pd.Timestamp, end: pd.Timestamp, period_days: int):
    """Decoupe [start, end] en sous-periodes de period_days jours."""
    periods = []
    cur = start
    while cur < end:
        nxt = min(cur + pd.Timedelta(days=period_days), end)
        periods.append((cur, nxt))
        cur = nxt
    return periods


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--step", type=int, default=5)
    p.add_argument("--assets", nargs="+", default=None)
    p.add_argument("--period_days", type=int, default=4,
                   help="Decoupage sous-periodes (default 4 jours)")
    p.add_argument("--workers", type=int, default=96)
    args = p.parse_args()

    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    if end.hour == 0 and end.minute == 0:
        end = end + pd.Timedelta(hours=23, minutes=59)

    assets = args.assets or LIVE_ASSETS
    periods = split_period(start, end, args.period_days)
    print(f"=== BACKTEST V12 MASSIVE PARALLEL ===", flush=True)
    print(f"Periode globale : {start} -> {end}", flush=True)
    print(f"Sous-periodes   : {len(periods)} x {args.period_days} jours", flush=True)
    print(f"Actifs          : {len(assets)}", flush=True)
    print(f"Tasks totales   : {len(assets) * len(periods)}", flush=True)
    print(f"Workers paralel : {args.workers}", flush=True)
    print(f"Step scan       : {args.step} min", flush=True)
    print(f"Cores dispo     : {os.cpu_count()}", flush=True)
    print()

    # Build tasks
    tasks = []
    task_id = 0
    for a in assets:
        for (sp_s, sp_e) in periods:
            tasks.append((a, str(sp_s), str(sp_e), args.step, task_id))
            task_id += 1
    print(f"Lancement de {len(tasks)} taches sur {args.workers} workers...", flush=True)
    print()

    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(backtest_task, t): t for t in tasks}
        done = 0
        for fut in as_completed(futures):
            done += 1
            try:
                r = fut.result()
                results.append(r)
                if r.get("ok"):
                    print(f"[{done}/{len(tasks)}] {r['asset']:8s} {r['start'][:10]}->{r['end'][:10]}: "
                          f"{r['n_trades']} trades en {r['elapsed_s']:.0f}s", flush=True)
                else:
                    print(f"[{done}/{len(tasks)}] FAIL {r['asset']} {r.get('start','?')[:10]}: {r.get('error','?')[:100]}", flush=True)
            except Exception as e:
                print(f"[{done}/{len(tasks)}] EXCEPTION : {e}", flush=True)

    elapsed = time.time() - t0
    print(f"\n=== TERMINE en {elapsed/60:.1f}min ===", flush=True)

    # Consolide tous les CSV
    import glob as _g
    all_csv = _g.glob(f"{ROOT}/data/bt_v12_*_*.csv")
    print(f"CSV trouves : {len(all_csv)}", flush=True)
    all_trades = []
    for c in all_csv:
        try:
            all_trades.append(pd.read_csv(c))
        except Exception:
            continue
    if not all_trades:
        print("Aucun trade trouve !", flush=True)
        return

    big = pd.concat(all_trades, ignore_index=True)
    # Dedup : un meme OB peut apparaitre dans 2 sous-periodes contigues si
    # son OB_ts est dans la fenetre warmup. On dedupe par (instrument, ob_ts, dir, place_ts).
    big = big.drop_duplicates(subset=["instrument", "ob_ts", "direction", "place_ts"])

    out_all = f"{ROOT}/data/bt_v12_ALL.csv"
    big.to_csv(out_all, index=False)
    print(f"Recap consolide : {out_all} ({len(big)} trades uniques)", flush=True)

    closed = big[big["outcome"].isin(["WIN", "LOSS"])]
    if len(closed):
        wr = (closed["outcome"] == "WIN").mean() * 100
        pnl_r = closed["pnl_r"].sum()
        print(f"\n=== STATS GLOBALES ===", flush=True)
        print(f"  Trades fermes : {len(closed)}", flush=True)
        print(f"  WR            : {wr:.1f}%", flush=True)
        print(f"  PnL (R)       : {pnl_r:.1f}", flush=True)
        print(f"  PnL/trade     : {pnl_r/len(closed):.3f} R", flush=True)
        print(f"\n  Par actif :", flush=True)
        for inst, sub in closed.groupby("instrument"):
            wr_a = (sub["outcome"] == "WIN").mean() * 100
            pnl_a = sub["pnl_r"].sum()
            print(f"    {inst:8s} : {len(sub):3d} trades, WR={wr_a:5.1f}%, PnL={pnl_a:+6.1f}R", flush=True)


if __name__ == "__main__":
    main()
