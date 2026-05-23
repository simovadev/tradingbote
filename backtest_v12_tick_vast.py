"""Backtest V12 TICK PAR TICK - version VAST (sans MT5).

Lit :
- les bougies depuis data_vantage/<ASSET>_<TF>.parquet (detection)
- les ticks depuis data_ticks/<ASSET>_ticks_<YYYYMMDD>.parquet (fill/SL/TP exact)

Detection sur M1 (= live), fill+exit sur vrais ticks bid/ask.
Parallelise par actif pour saturer Vast (regle saturation).

Usage Vast :
    cd /workspace/TradingBot
    nohup python3 -u backtest_v12_tick_vast.py --date 2026-05-19 \
        --workers 14 > /workspace/bt_tick.log 2>&1 & disown
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault("SWS_OVERRIDE", "1")
os.environ.setdefault("RR_OVERRIDE", "1.5")
os.environ.setdefault("BUILD_DATA_DIR", "data_vantage")

ROOT = "c:/Users/Shadow/TradingBot" if sys.platform == "win32" else "/workspace/TradingBot"
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

import multiprocessing as _mp
_mp.set_start_method("spawn", force=True)

LIVE_ASSETS = [
    "XAUUSD", "NAS100", "GER40", "BTCUSD",
    "EURUSD", "GBPUSD", "AUDUSD", "USDJPY",
    "SP500", "DJ30", "UK100", "FRA40",
    "USDCAD", "USDCHF",
]

CAP_AGE_OB_MIN = 30
EXPIRE_PENDING_MIN = 60
INITIAL_BALANCE = 150.0


@dataclass
class TradeRec:
    instrument: str
    direction: str
    ob_ts: str
    placed_ts: str
    ml: float
    entry: float
    sl: float
    tp: float
    rr: float
    fill_ts: str = ""
    exit_ts: str = ""
    outcome: str = "PENDING"
    pnl_r: float = 0.0


def simulate_on_ticks(rec, tick_times, tick_bid, tick_ask):
    placed = pd.Timestamp(rec.placed_ts).value
    expire = placed + EXPIRE_PENDING_MIN * 60 * 1_000_000_000
    n = len(tick_times)
    fill_idx = None
    for i in range(n):
        t = tick_times[i]
        if t <= placed:
            continue
        if t > expire:
            rec.outcome = "NO_FILL"; return
        if rec.direction == "bullish":
            if tick_ask[i] <= rec.entry:
                fill_idx = i; break
        else:
            if tick_bid[i] >= rec.entry:
                fill_idx = i; break
    if fill_idx is None:
        rec.outcome = "NO_FILL"; return
    rec.fill_ts = str(pd.Timestamp(tick_times[fill_idx], tz="UTC"))
    for i in range(fill_idx + 1, n):
        if rec.direction == "bullish":
            if tick_bid[i] <= rec.sl:
                rec.outcome = "LOSS"; rec.pnl_r = -1.0
                rec.exit_ts = str(pd.Timestamp(tick_times[i], tz="UTC")); return
            if tick_bid[i] >= rec.tp:
                rec.outcome = "WIN"; rec.pnl_r = rec.rr
                rec.exit_ts = str(pd.Timestamp(tick_times[i], tz="UTC")); return
        else:
            if tick_ask[i] >= rec.sl:
                rec.outcome = "LOSS"; rec.pnl_r = -1.0
                rec.exit_ts = str(pd.Timestamp(tick_times[i], tz="UTC")); return
            if tick_ask[i] <= rec.tp:
                rec.outcome = "WIN"; rec.pnl_r = rec.rr
                rec.exit_ts = str(pd.Timestamp(tick_times[i], tz="UTC")); return
    rec.outcome = "OPEN"


def backtest_asset(args_tuple):
    asset, date_str, step = args_tuple
    try:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["SWS_OVERRIDE"] = "1"
        os.environ["RR_OVERRIDE"] = "1.5"
        os.environ["BUILD_DATA_DIR"] = "data_vantage"
        import sys as _sys
        _sys.path.insert(0, ROOT)
        import pandas as _pd
        import numpy as _np
        from bot_v2.data_loader import load
        from bot_v2.live_runner_v2 import compute_asset, load_model
        from bot_v2.concepts.daily_bias import build_d1_from_h1

        day = _pd.Timestamp(date_str, tz="UTC")
        start = day
        end = day + _pd.Timedelta(hours=23, minutes=59)

        # Bougies depuis data_vantage (avec buffer 60j amont)
        ws = start - _pd.Timedelta(days=60)
        df_m1 = load(asset, "M1", start=ws, end=end + _pd.Timedelta(days=1))
        df_m15 = load(asset, "M15", start=ws, end=end + _pd.Timedelta(days=1))
        df_h1 = load(asset, "H1", start=ws - _pd.Timedelta(days=30), end=end + _pd.Timedelta(days=1))
        try:
            df_h4 = load(asset, "H4", start=ws - _pd.Timedelta(days=120), end=end + _pd.Timedelta(days=1))
        except Exception:
            df_h4 = None
        try:
            df_d1 = load(asset, "D1", start=ws - _pd.Timedelta(days=400), end=end + _pd.Timedelta(days=1))
        except Exception:
            df_d1 = build_d1_from_h1(df_h1)

        # Ticks depuis data_ticks
        tick_path = f"{ROOT}/data_ticks/{asset}_ticks_{date_str.replace('-','')}.parquet"
        if not os.path.exists(tick_path):
            return {"ok": False, "asset": asset, "error": f"ticks absents {tick_path}"}
        tdf = _pd.read_parquet(tick_path)
        tick_times = tdf["time_ns"].values.astype(_np.int64)
        tick_bid = tdf["bid"].values.astype(float)
        tick_ask = tdf["ask"].values.astype(float)

        loaded = load_model(asset)
        if loaded is None:
            return {"ok": False, "asset": asset, "error": "pas de modele"}

        # Scan boucle (detection M1)
        active = []
        evaluated = set()
        cur = start
        n_setups = 0
        while cur <= end:
            cut = cur - _pd.Timedelta(minutes=1)
            ie = df_m1.index.searchsorted(cut, side="right")
            sub_m1 = df_m1.iloc[max(0, ie - 88000):ie]
            if len(sub_m1) < 200:
                cur += _pd.Timedelta(minutes=step); continue
            i15 = df_m15.index.searchsorted(cut, side="right")
            i1 = df_h1.index.searchsorted(cut, side="right")
            id1 = df_d1.index.searchsorted(cut, side="right")
            payload = {
                "instrument": asset, "df_m1": sub_m1,
                "df_m15": df_m15.iloc[max(0, i15 - 11000):i15],
                "df_h1": df_h1.iloc[max(0, i1 - 2800):i1],
                "df_h4": (df_h4.iloc[:df_h4.index.searchsorted(cut, side="right")][-500:] if df_h4 is not None else None),
                "df_d1": df_d1.iloc[max(0, id1 - 120):id1],
                "correlated_dfs": {}, "balance": INITIAL_BALANCE, "debug_diag": False,
            }
            try:
                res = compute_asset(payload)
            except Exception:
                cur += _pd.Timedelta(minutes=step); continue
            for s in res.get("setups", []):
                ob = s["ob"]; ob_ts = s["ts"]; proba = s.get("proba", 0)
                key = (str(ob_ts), ob.direction)
                if key in evaluated:
                    continue
                if (cur - ob_ts).total_seconds() / 60 > CAP_AGE_OB_MIN:
                    continue
                t2 = s["r"].trade_setup
                active.append(TradeRec(
                    instrument=asset, direction=ob.direction, ob_ts=str(ob_ts),
                    placed_ts=str(cur), ml=proba,
                    entry=t2.entry_price, sl=t2.stop_loss, tp=t2.take_profit, rr=t2.rr))
                evaluated.add(key)
                n_setups += 1
            cur += _pd.Timedelta(minutes=step)

        for rec in active:
            simulate_on_ticks(rec, tick_times, tick_bid, tick_ask)

        closed = [t for t in active if t.outcome in ("WIN", "LOSS")]
        wr = (sum(1 for t in closed if t.outcome == "WIN") / len(closed) * 100) if closed else 0
        return {
            "ok": True, "asset": asset, "n_setups": n_setups,
            "n_closed": len(closed),
            "n_win": sum(1 for t in closed if t.outcome == "WIN"),
            "wr": wr, "pnl_r": sum(t.pnl_r for t in closed),
            "n_nofill": sum(1 for t in active if t.outcome == "NO_FILL"),
            "n_open": sum(1 for t in active if t.outcome == "OPEN"),
            "trades": [vars(t) for t in active],
        }
    except Exception as e:
        import traceback
        return {"ok": False, "asset": asset, "error": str(e)[:200],
                "trace": traceback.format_exc()[-500:]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--date", required=True, help="YYYY-MM-DD")
    p.add_argument("--assets", nargs="+", default=LIVE_ASSETS)
    p.add_argument("--step", type=int, default=5)
    p.add_argument("--workers", type=int, default=14)
    args = p.parse_args()

    print(f"=== BACKTEST TICK PAR TICK (VAST) ===", flush=True)
    print(f"Date    : {args.date}", flush=True)
    print(f"Actifs  : {len(args.assets)}", flush=True)
    print(f"Workers : {args.workers}", flush=True)
    print()

    tasks = [(a, args.date, args.step) for a in args.assets]
    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(backtest_asset, t): t[0] for t in tasks}
        done = 0
        for fut in as_completed(futs):
            done += 1
            r = fut.result()
            results.append(r)
            if r.get("ok"):
                print(f"[{done:2d}/{len(tasks)}] {r['asset']:8s} setups={r['n_setups']:3d} "
                      f"fermes={r['n_closed']:3d} WR={r['wr']:5.1f}% PnL={r['pnl_r']:+6.1f}R "
                      f"NF={r['n_nofill']} OPEN={r['n_open']}", flush=True)
            else:
                print(f"[{done:2d}/{len(tasks)}] FAIL {r['asset']} : {r.get('error','')[:90]}", flush=True)

    elapsed = time.time() - t0
    print(f"\n=== TERMINE en {elapsed/60:.1f}min ===")

    ok = [r for r in results if r.get("ok")]
    tot_closed = sum(r["n_closed"] for r in ok)
    tot_win = sum(r["n_win"] for r in ok)
    tot_pnl = sum(r["pnl_r"] for r in ok)
    wr = (tot_win / tot_closed * 100) if tot_closed else 0
    print(f"\n=== RECAP GLOBAL TICK ===")
    print(f"  Trades fermes : {tot_closed}")
    print(f"  WR            : {wr:.1f}%")
    print(f"  PnL (R)       : {tot_pnl:+.1f}")
    print(f"  NO_FILL       : {sum(r['n_nofill'] for r in ok)}")
    print(f"  OPEN          : {sum(r['n_open'] for r in ok)}")
    print(f"\n  Par actif :")
    for r in sorted(ok, key=lambda x: x["asset"]):
        if r["n_closed"]:
            print(f"    {r['asset']:8s} : {r['n_closed']:3d} trades, WR={r['wr']:5.1f}%, PnL={r['pnl_r']:+.1f}R")

    # Save journal
    import json
    all_tr = []
    for r in ok:
        all_tr.extend(r.get("trades", []))
    out = f"{ROOT}/bt_tick_{args.date.replace('-','')}_trades.csv"
    if all_tr:
        pd.DataFrame(all_tr).to_csv(out, index=False)
        print(f"\nJournal : {out}")


if __name__ == "__main__":
    main()
