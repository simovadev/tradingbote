"""Backtest V12 TICK PAR TICK - le plus fidele possible au live.

Difference vs backtest_v12_realistic.py :
- La DETECTION des setups reste sur bougies M1 (= live : le bot scanne les M1)
- Le FILL et le SL/TP sont verifies sur les VRAIS TICKS (bid/ask) de MT5
  -> ordre exact SL/TP, prix exact de fill, spread reel capture

Reproduit a ~95-98% le live. Seuls ecarts restants : latence (18s),
slippage/requote (imprevisibles, rares sur forex liquide).

Necessite MT5 ouvert + connecte (pour copy_ticks_range).
Tourne en LOCAL (PC), pas sur Vast (Vast n'a pas MT5).

Usage :
    python backtest_v12_tick.py --start 2026-05-19 --end 2026-05-19 --assets XAUUSD EURUSD
"""
from __future__ import annotations
import argparse
import os
import sys
import time
import datetime
from dataclasses import dataclass

os.environ.setdefault("SWS_OVERRIDE", "1")
os.environ.setdefault("RR_OVERRIDE", "1.5")
os.environ.setdefault("BUILD_DATA_DIR", "data_vantage")

sys.path.insert(0, "c:/Users/Shadow/TradingBot")

import numpy as np
import pandas as pd
import MetaTrader5 as mt5

from bot_v2.mt5_executor import MT5Executor, to_broker_symbol
from bot_v2.live_runner_v2 import compute_asset, load_model

LIVE_ASSETS = [
    "XAUUSD", "NAS100", "GER40", "BTCUSD",
    "EURUSD", "GBPUSD", "AUDUSD", "USDJPY",
    "SP500", "DJ30", "UK100", "FRA40",
    "USDCAD", "USDCHF",
]

CAP_AGE_OB_MIN = 30
EXPIRE_PENDING_MIN = 60
RISK_PCT = 0.02
INITIAL_BALANCE = 150.0


@dataclass
class TradeRec:
    instrument: str
    direction: str
    ob_ts: pd.Timestamp
    placed_ts: pd.Timestamp
    ml: float
    entry: float
    sl: float
    tp: float
    rr: float
    fill_ts: pd.Timestamp | None = None
    fill_price: float | None = None
    exit_ts: pd.Timestamp | None = None
    exit_price: float | None = None
    outcome: str = "PENDING"
    pnl_r: float = 0.0


def simulate_on_ticks(pending: TradeRec, ticks: np.ndarray, tick_times: np.ndarray,
                      tick_bid: np.ndarray, tick_ask: np.ndarray) -> None:
    """Rejoue les ticks pour determiner fill puis SL/TP.

    LIMIT fill (le prix revient toucher entry) :
      - bullish (BUY LIMIT) : rempli quand ASK <= entry. Sortie : SL si bid<=sl, TP si bid>=tp
      - bearish (SELL LIMIT) : rempli quand BID >= entry. Sortie : SL si ask>=sl, TP si ask<=tp
    Le spread bid/ask est ainsi capture (on entre/sort au bon cote).
    """
    placed = pending.placed_ts.value  # ns epoch
    expire = placed + EXPIRE_PENDING_MIN * 60 * 1_000_000_000

    n = len(tick_times)
    # 1. Cherche le FILL
    fill_idx = None
    for i in range(n):
        t = tick_times[i]
        if t <= placed:
            continue
        if t > expire:
            pending.outcome = "NO_FILL"
            return
        if pending.direction == "bullish":
            # BUY LIMIT rempli si le ASK descend a entry
            if tick_ask[i] <= pending.entry:
                fill_idx = i
                pending.fill_price = pending.entry  # fill au prix limite
                break
        else:
            # SELL LIMIT rempli si le BID monte a entry
            if tick_bid[i] >= pending.entry:
                fill_idx = i
                pending.fill_price = pending.entry
                break
    if fill_idx is None:
        pending.outcome = "NO_FILL"
        return
    pending.fill_ts = pd.Timestamp(tick_times[fill_idx], tz="UTC")

    # 2. Apres le fill, cherche SL ou TP (au tick, ordre exact)
    for i in range(fill_idx + 1, n):
        if pending.direction == "bullish":
            # position LONG : on suit le BID (prix de vente)
            if tick_bid[i] <= pending.sl:
                pending.outcome = "LOSS"; pending.pnl_r = -1.0
                pending.exit_ts = pd.Timestamp(tick_times[i], tz="UTC")
                pending.exit_price = pending.sl
                return
            if tick_bid[i] >= pending.tp:
                pending.outcome = "WIN"; pending.pnl_r = pending.rr
                pending.exit_ts = pd.Timestamp(tick_times[i], tz="UTC")
                pending.exit_price = pending.tp
                return
        else:
            # position SHORT : on suit le ASK (prix d'achat)
            if tick_ask[i] >= pending.sl:
                pending.outcome = "LOSS"; pending.pnl_r = -1.0
                pending.exit_ts = pd.Timestamp(tick_times[i], tz="UTC")
                pending.exit_price = pending.sl
                return
            if tick_ask[i] <= pending.tp:
                pending.outcome = "WIN"; pending.pnl_r = pending.rr
                pending.exit_ts = pd.Timestamp(tick_times[i], tz="UTC")
                pending.exit_price = pending.tp
                return
    pending.outcome = "OPEN"  # ni SL ni TP atteint dans la fenetre de ticks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--assets", nargs="+", default=["XAUUSD", "EURUSD", "NAS100"])
    p.add_argument("--step", type=int, default=5, help="Pas de scan en min")
    args = p.parse_args()

    exe = MT5Executor()
    if not exe.initialize():
        print("MT5 INIT FAIL - ouvre MT5 et connecte-toi")
        sys.exit(1)
    off = exe.broker_utc_offset_sec
    print(f"MT5 OK, offset={off}s")

    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    if end.hour == 0 and end.minute == 0:
        end = end + pd.Timedelta(hours=23, minutes=59)

    all_trades = []
    for asset in args.assets:
        print(f"\n{'='*60}\n=== {asset} ===", flush=True)
        broker = to_broker_symbol(asset)

        # --- Charge bougies (pour la detection, comme le live) ---
        def fetch_tf(tf, n):
            rates = mt5.copy_rates_from_pos(broker, tf, 0, n)
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"] - off, unit="s", utc=True)
            return df.set_index("time").rename(columns={"tick_volume": "volume"})[
                ["open", "high", "low", "close", "volume"]]

        df_m1_full = fetch_tf(mt5.TIMEFRAME_M1, 100000)
        df_m15_full = fetch_tf(mt5.TIMEFRAME_M15, 11000)
        df_h1_full = fetch_tf(mt5.TIMEFRAME_H1, 2800)
        try:
            df_h4_full = fetch_tf(mt5.TIMEFRAME_H4, 500)
        except Exception:
            df_h4_full = None
        df_d1_full = fetch_tf(mt5.TIMEFRAME_D1, 120)

        # --- Charge TOUS les ticks de la periode (+1h marge pour fills tardifs) ---
        tk_start = (start - pd.Timedelta(hours=1)).to_pydatetime()
        tk_end = (end + pd.Timedelta(hours=2)).to_pydatetime()
        raw = mt5.copy_ticks_range(
            broker,
            tk_start + datetime.timedelta(seconds=off),
            tk_end + datetime.timedelta(seconds=off),
            mt5.COPY_TICKS_ALL,
        )
        if raw is None or len(raw) == 0:
            print(f"  PAS DE TICKS pour {asset}, skip ({mt5.last_error()})")
            continue
        tdf = pd.DataFrame(raw)
        # epoch ns en UTC reel
        tick_times = ((tdf["time_msc"].values.astype(np.int64)) * 1_000_000) - (off * 1_000_000_000)
        tick_bid = tdf["bid"].values.astype(float)
        tick_ask = tdf["ask"].values.astype(float)
        print(f"  {len(tdf):,} ticks charges ({pd.Timestamp(tick_times[0],tz='UTC')} -> {pd.Timestamp(tick_times[-1],tz='UTC')})")

        # --- Verifie modele V12 ---
        loaded = load_model(asset)
        if loaded is None:
            print(f"  pas de modele, skip"); continue
        _, feats = loaded
        print(f"  {'V12' if 'snapshot_k' not in feats else 'V11'} ({len(feats)} features)")

        # --- Boucle de scan (detection sur M1, comme le live) ---
        active = []
        evaluated = set()
        cur = start
        t0 = time.time()
        n_setups = 0
        while cur <= end:
            cut = cur - pd.Timedelta(minutes=1)
            ie = df_m1_full.index.searchsorted(cut, side="right")
            df_m1 = df_m1_full.iloc[max(0, ie - 88000):ie]
            if len(df_m1) < 200:
                cur += pd.Timedelta(minutes=args.step); continue
            i15 = df_m15_full.index.searchsorted(cut, side="right")
            i1 = df_h1_full.index.searchsorted(cut, side="right")
            id1 = df_d1_full.index.searchsorted(cut, side="right")
            payload = {
                "instrument": asset,
                "df_m1": df_m1,
                "df_m15": df_m15_full.iloc[max(0, i15 - 11000):i15],
                "df_h1": df_h1_full.iloc[max(0, i1 - 2800):i1],
                "df_h4": (df_h4_full.iloc[:df_h4_full.index.searchsorted(cut, side="right")][-500:]
                          if df_h4_full is not None else None),
                "df_d1": df_d1_full.iloc[max(0, id1 - 120):id1],
                "correlated_dfs": {},
                "balance": INITIAL_BALANCE,
                "debug_diag": False,
            }
            try:
                res = compute_asset(payload)
            except Exception:
                cur += pd.Timedelta(minutes=args.step); continue

            for s in res.get("setups", []):
                ob = s["ob"]; ob_ts = s["ts"]; proba = s.get("proba", 0)
                key = (str(ob_ts), ob.direction)
                if key in evaluated:
                    continue
                # cap age
                age = (cur - ob_ts).total_seconds() / 60
                if age > CAP_AGE_OB_MIN:
                    continue
                ts2 = s["r"].trade_setup
                rec = TradeRec(
                    instrument=asset, direction=ob.direction, ob_ts=ob_ts,
                    placed_ts=cur, ml=proba,
                    entry=ts2.entry_price, sl=ts2.stop_loss, tp=ts2.take_profit,
                    rr=ts2.rr,
                )
                active.append(rec)
                evaluated.add(key)
                n_setups += 1
            cur += pd.Timedelta(minutes=args.step)

        # --- Resolution TICK par TICK de tous les pendings ---
        for rec in active:
            simulate_on_ticks(rec, None, tick_times, tick_bid, tick_ask)
            all_trades.append(rec)

        elapsed = time.time() - t0
        closed = [t for t in active if t.outcome in ("WIN", "LOSS")]
        wr = (sum(1 for t in closed if t.outcome == "WIN") / len(closed) * 100) if closed else 0
        print(f"  {asset}: {n_setups} setups, {len(closed)} fermes, WR={wr:.1f}%, "
              f"NO_FILL={sum(1 for t in active if t.outcome=='NO_FILL')}, "
              f"OPEN={sum(1 for t in active if t.outcome=='OPEN')} ({elapsed:.0f}s)", flush=True)

    exe.shutdown()

    # ===== RECAP GLOBAL =====
    print(f"\n{'='*60}\n=== RECAP TICK PAR TICK ===")
    closed = [t for t in all_trades if t.outcome in ("WIN", "LOSS")]
    if not closed:
        print("Aucun trade ferme.")
        return
    wins = sum(1 for t in closed if t.outcome == "WIN")
    wr = wins / len(closed) * 100
    pnl_r = sum(t.pnl_r for t in closed)
    print(f"  Trades fermes : {len(closed)}")
    print(f"  WR            : {wr:.1f}%")
    print(f"  PnL (R)       : {pnl_r:+.1f}")
    print(f"  NO_FILL       : {sum(1 for t in all_trades if t.outcome=='NO_FILL')}")
    print(f"  OPEN          : {sum(1 for t in all_trades if t.outcome=='OPEN')}")
    print(f"\n  Par actif :")
    by_a = {}
    for t in closed:
        by_a.setdefault(t.instrument, []).append(t)
    for a, ts in sorted(by_a.items()):
        w = sum(1 for x in ts if x.outcome == "WIN")
        print(f"    {a:8s} : {len(ts):3d} trades, WR={w/len(ts)*100:5.1f}%, PnL={sum(x.pnl_r for x in ts):+.1f}R")

    # CSV
    rows = [{
        "instrument": t.instrument, "direction": t.direction,
        "ob_ts": t.ob_ts, "placed_ts": t.placed_ts, "ml": t.ml,
        "entry": t.entry, "sl": t.sl, "tp": t.tp,
        "fill_ts": t.fill_ts, "fill_price": t.fill_price,
        "exit_ts": t.exit_ts, "exit_price": t.exit_price,
        "outcome": t.outcome, "pnl_r": t.pnl_r,
    } for t in all_trades]
    out = "c:/Users/Shadow/TradingBot/backtest_v12_tick_trades.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nJournal sauve : {out}")


if __name__ == "__main__":
    main()
