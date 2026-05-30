"""Analyse detaillee des 7 trades GER40 du 29/05/2026 :
- Heure exacte de chaque trade
- Direction, entry, SL, TP1R, TP2R
- Outcome
- Pour les LOSS : on regarde le contexte (les bougies entre entry et SL)
- Pour comprendre POURQUOI ca a touche SL
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

from bot_v2.concepts.safe import (
    find_obs_simple, daily_bias_safe, htf_value_at_t, atr_safe,
)

DATA = ROOT / "data_vantage"
ASSET = "GER40"
SPREAD = 0.8

SL_BUFFER_ATR = 0.1
TP_RR = 2.0
PARTIAL_R = 1.0
MAX_FILL_BARS = 20
MAX_HOLD_BARS = 50

TARGET_DAY = pd.Timestamp("2026-05-29", tz="UTC")


def get_filters_for_ob(df_m5, df_h1, df_d1, ob):
    val_ts = ob.validation_ts; val_idx = ob.validation_index
    d1 = daily_bias_safe(df_d1, val_ts)
    d1_a = False; d1_bias = None
    if d1["ok"]:
        d1_bias = d1["bias"]
        d1_h = d1["bias"] == "haussier"
        d1_a = (ob.direction == "bullish" and d1_h) or (ob.direction == "bearish" and not d1_h)
    h1_close = htf_value_at_t(df_h1, val_ts, "close")
    h1_a = False; h1_trend = None
    if h1_close is not None:
        pos = df_h1.index.searchsorted(val_ts, side="left") - 1 - 10
        if pos >= 0:
            h1_old = float(df_h1["close"].iloc[pos])
            h1_h = h1_close > h1_old
            h1_trend = "haussier" if h1_h else "baissier"
            h1_a = (ob.direction == "bullish" and h1_h) or (ob.direction == "bearish" and not h1_h)
    pd_a = False; pd_zone = None
    if df_d1 is not None and len(df_d1) >= 2:
        target_day = pd.Timestamp(val_ts).normalize()
        df_d1_past = df_d1[df_d1.index < target_day]
        if len(df_d1_past) >= 1:
            pdh = float(df_d1_past["high"].iloc[-1])
            pdl = float(df_d1_past["low"].iloc[-1])
            mid = (pdh + pdl) / 2
            price = float(df_m5["close"].iloc[val_idx])
            pd_zone = "discount" if price < mid else ("premium" if price > mid else "equilibrium")
            pd_a = (
                (ob.direction == "bullish" and price < mid) or
                (ob.direction == "bearish" and price > mid)
            )
    return {"d1": d1_a, "h1": h1_a, "pd": pd_a, "d1_bias": d1_bias,
            "h1_trend": h1_trend, "pd_zone": pd_zone}


def simulate_with_details(df_m5, ob, atr, spread=0.0):
    val_idx = ob.validation_index
    if val_idx + 1 >= len(df_m5): return None
    hs = spread / 2
    if ob.direction == "bullish":
        sl = ob.ob_low - SL_BUFFER_ATR * atr; entry = ob.ob_high
        entry_eff = entry + hs; sl_eff = sl - hs
    else:
        sl = ob.ob_high + SL_BUFFER_ATR * atr; entry = ob.ob_low
        entry_eff = entry - hs; sl_eff = sl + hs
    if abs(entry_eff - sl_eff) < 1e-9: return None
    risk = abs(entry_eff - sl_eff)
    if ob.direction == "bullish":
        tp_1r = entry_eff + PARTIAL_R * risk; tp_2r = entry_eff + TP_RR * risk
    else:
        tp_1r = entry_eff - PARTIAL_R * risk; tp_2r = entry_eff - TP_RR * risk

    fh = df_m5["high"].values[val_idx + 1:val_idx + 1 + MAX_FILL_BARS]
    fl = df_m5["low"].values[val_idx + 1:val_idx + 1 + MAX_FILL_BARS]
    if len(fh) == 0: return None
    entry_idx_rel = None
    for i in range(len(fh)):
        if ob.direction == "bullish":
            if fl[i] <= entry:
                if fl[i] <= sl: return {"status": "INVALIDATED", "entry": entry, "sl": sl, "tp_1r": tp_1r, "tp_2r": tp_2r}
                entry_idx_rel = i; break
        else:
            if fh[i] >= entry:
                if fh[i] >= sl: return {"status": "INVALIDATED", "entry": entry, "sl": sl, "tp_1r": tp_1r, "tp_2r": tp_2r}
                entry_idx_rel = i; break
    if entry_idx_rel is None:
        return {"status": "NO_FILL", "entry": entry, "sl": sl, "tp_1r": tp_1r, "tp_2r": tp_2r}
    entry_global = val_idx + 1 + entry_idx_rel
    entry_ts = df_m5.index[entry_global]
    fe = entry_global + 1
    en = min(fe + MAX_HOLD_BARS, len(df_m5))
    if en - fe < 5: return None
    pfh = df_m5["high"].values[fe:en]; pfl = df_m5["low"].values[fe:en]
    half_locked = False; stop = sl_eff
    n_bars_in_trade = 0
    max_favorable = 0.0
    for i in range(len(pfh)):
        n_bars_in_trade = i + 1
        if ob.direction == "bullish":
            hit_sl = pfl[i] <= stop; hit_1r = pfh[i] >= tp_1r; hit_2r = pfh[i] >= tp_2r
            fav = (pfh[i] - entry_eff) / risk
        else:
            hit_sl = pfh[i] >= stop; hit_1r = pfl[i] <= tp_1r; hit_2r = pfl[i] <= tp_2r
            fav = (entry_eff - pfl[i]) / risk
        max_favorable = max(max_favorable, fav)
        if hit_sl and hit_2r:
            status = "WIN_BE" if half_locked else "LOSS"
            return {"status": status, "entry": entry, "sl": sl, "tp_1r": tp_1r, "tp_2r": tp_2r,
                    "entry_ts": entry_ts, "exit_ts": df_m5.index[fe + i],
                    "n_bars": n_bars_in_trade, "max_favorable_R": max_favorable,
                    "half_locked": half_locked}
        if hit_sl:
            status = "WIN_BE" if half_locked else "LOSS"
            return {"status": status, "entry": entry, "sl": sl, "tp_1r": tp_1r, "tp_2r": tp_2r,
                    "entry_ts": entry_ts, "exit_ts": df_m5.index[fe + i],
                    "n_bars": n_bars_in_trade, "max_favorable_R": max_favorable,
                    "half_locked": half_locked}
        if not half_locked and hit_1r:
            half_locked = True; stop = entry_eff
            if hit_2r:
                return {"status": "WIN_FULL", "entry": entry, "sl": sl, "tp_1r": tp_1r, "tp_2r": tp_2r,
                        "entry_ts": entry_ts, "exit_ts": df_m5.index[fe + i],
                        "n_bars": n_bars_in_trade, "max_favorable_R": max_favorable,
                        "half_locked": True}
        elif half_locked and hit_2r:
            return {"status": "WIN_FULL", "entry": entry, "sl": sl, "tp_1r": tp_1r, "tp_2r": tp_2r,
                    "entry_ts": entry_ts, "exit_ts": df_m5.index[fe + i],
                    "n_bars": n_bars_in_trade, "max_favorable_R": max_favorable,
                    "half_locked": True}
    status = "TIME_BE" if half_locked else "TIME_FLAT"
    return {"status": status, "entry": entry, "sl": sl, "tp_1r": tp_1r, "tp_2r": tp_2r,
            "entry_ts": entry_ts, "exit_ts": df_m5.index[en - 1],
            "n_bars": n_bars_in_trade, "max_favorable_R": max_favorable,
            "half_locked": half_locked}


def to_fr(ts):
    return ts.tz_convert("Europe/Paris").strftime("%H:%M FR")


def main():
    print(f"=" * 90)
    print(f"ANALYSE DETAILLEE : {ASSET} 29/05/2026")
    print(f"=" * 90)

    df_m5 = pd.read_parquet(DATA / f"{ASSET}_M5.parquet")[["open","high","low","close"]]
    df_h1 = pd.read_parquet(DATA / f"{ASSET}_H1.parquet")[["open","high","low","close"]]
    df_d1 = pd.read_parquet(DATA / f"{ASSET}_D1.parquet")[["open","high","low","close"]]

    ctx_start = TARGET_DAY - pd.Timedelta(days=5)
    day_end = TARGET_DAY + pd.Timedelta(days=1)
    df_m5 = df_m5[(df_m5.index >= ctx_start) & (df_m5.index < day_end)]
    df_h1 = df_h1[(df_h1.index >= ctx_start) & (df_h1.index < day_end)]
    df_d1 = df_d1[(df_d1.index >= TARGET_DAY - pd.Timedelta(days=30)) & (df_d1.index < day_end)]

    # Context daily
    target_day = TARGET_DAY.normalize()
    df_d1_past = df_d1[df_d1.index < target_day]
    pdh = float(df_d1_past["high"].iloc[-1]) if len(df_d1_past) else None
    pdl = float(df_d1_past["low"].iloc[-1]) if len(df_d1_past) else None
    mid_daily = (pdh + pdl) / 2 if pdh and pdl else None
    print(f"\nContexte D1 (jour PRECEDENT, 28/05/2026) :")
    print(f"  PDH 28/05 : {pdh:.2f}")
    print(f"  PDL 28/05 : {pdl:.2f}")
    print(f"  MID daily : {mid_daily:.2f}  (au-dessus = premium, en-dessous = discount)")
    d1_bias = daily_bias_safe(df_d1, TARGET_DAY + pd.Timedelta(hours=1))
    print(f"  D1 bias   : {d1_bias['bias']}  (close J-1 {d1_bias['j1_close']:.2f} vs J-2 {d1_bias['j2_close']:.2f})")

    # Detect OBs et trades
    obs = find_obs_simple(df_m5, as_of_index=len(df_m5) - 1)
    obs_day = [ob for ob in obs if TARGET_DAY <= ob.validation_ts < day_end]
    obs_day.sort(key=lambda x: x.validation_ts)

    trades_detail = []
    for ob in obs_day:
        f = get_filters_for_ob(df_m5, df_h1, df_d1, ob)
        if not (f["d1"] and f["h1"] and f["pd"]): continue
        atr = atr_safe(df_m5, ob.validation_index)
        if atr <= 0: continue
        r = simulate_with_details(df_m5, ob, atr, spread=SPREAD)
        if r is None: continue
        trades_detail.append({"ob": ob, "filters": f, "result": r, "atr": atr})

    print(f"\n{len(trades_detail)} trades pris sur GER40 ce jour.\n")
    print(f"=" * 90)

    for i, t in enumerate(trades_detail, 1):
        ob = t["ob"]; r = t["result"]; f = t["filters"]; atr = t["atr"]
        print(f"\n--- TRADE {i} : {ob.direction.upper()} {ASSET} ---")
        print(f"  Validation OB : {to_fr(ob.validation_ts)} | OB zone [{ob.ob_low:.2f} - {ob.ob_high:.2f}]")
        print(f"  Contexte : D1={f['d1_bias']}  H1={f['h1_trend']}  Zone={f['pd_zone']}")
        print(f"  Plan : Entry {r['entry']:.2f}  SL {r['sl']:.2f}  TP1R {r['tp_1r']:.2f}  TP2R {r['tp_2r']:.2f}")
        print(f"  Risk size = {abs(r['entry'] - r['sl']):.2f} pts (= {abs(r['entry']-r['sl'])/atr:.2f} ATR)")
        if r.get("entry_ts"):
            print(f"  Entry effective : {to_fr(r['entry_ts'])}")
        if r.get("exit_ts"):
            print(f"  Exit : {to_fr(r['exit_ts'])} apres {r.get('n_bars', '?')} bougies M5")
        print(f"  STATUS : {r['status']}", end="")
        if r.get("max_favorable_R") is not None:
            print(f"  | Best max favorable = +{r['max_favorable_R']:.2f}R")
        else:
            print()

        # Pour les LOSS : analyse contextuelle
        if r["status"] == "LOSS":
            entry_idx_in_df = df_m5.index.searchsorted(r["entry_ts"])
            exit_idx_in_df = df_m5.index.searchsorted(r["exit_ts"])
            trade_bars = df_m5.iloc[entry_idx_in_df:exit_idx_in_df + 1]
            n_bars = len(trade_bars)
            print(f"  >>> ANALYSE LOSS :")
            print(f"      Trade a dure {n_bars} bougies M5 (~{n_bars*5} min)")
            if n_bars > 0:
                max_h = trade_bars["high"].max()
                min_l = trade_bars["low"].min()
                if ob.direction == "bullish":
                    fav_pts = max_h - r["entry"]
                    print(f"      Max favorable atteint : {max_h:.2f} (entry {r['entry']:.2f}, +{fav_pts:.2f} pts = +{fav_pts/abs(r['entry']-r['sl']):.2f}R)")
                else:
                    fav_pts = r["entry"] - min_l
                    print(f"      Max favorable atteint : {min_l:.2f} (entry {r['entry']:.2f}, +{fav_pts:.2f} pts = +{fav_pts/abs(r['entry']-r['sl']):.2f}R)")
                # Volatilite pendant le trade
                trade_atr = (trade_bars["high"] - trade_bars["low"]).mean()
                print(f"      ATR moyen pendant le trade : {trade_atr:.2f} (vs ATR M5 historique : {atr:.2f})")
                # Cas typique : prix part dans le bon sens, retrace, casse SL
                if r["max_favorable_R"] < 0.5:
                    print(f"      -> Le prix N'a JAMAIS bouge dans notre sens (max +{r['max_favorable_R']:.2f}R)")
                    print(f"         => Setup invalide tres rapidement, ATR sl trop serre, ou move directionnel contre nous")
                elif r["max_favorable_R"] < 1.0:
                    print(f"      -> Le prix est parti dans notre sens mais a retrace avant 1R")
                    print(f"         => Pas de partial declenche, SL touche apres retracement")

    # Resume final
    print(f"\n{'=' * 90}")
    print(f"RESUME : {len(trades_detail)} trades")
    wins = sum(1 for t in trades_detail if t["result"]["status"].startswith("WIN"))
    losses = sum(1 for t in trades_detail if t["result"]["status"] == "LOSS")
    print(f"  WIN_FULL : {sum(1 for t in trades_detail if t['result']['status'] == 'WIN_FULL')}")
    print(f"  WIN_BE   : {sum(1 for t in trades_detail if t['result']['status'] == 'WIN_BE')}")
    print(f"  LOSS     : {losses}")
    # Calcul PnL
    pnl_map = {"WIN_FULL": 1.5, "WIN_BE": 0.5, "LOSS": -1.0, "TIME_BE": 0.5, "TIME_FLAT": 0.0}
    total = sum(pnl_map.get(t["result"]["status"], 0) for t in trades_detail)
    print(f"  PnL total : {total:+.2f}R")


if __name__ == "__main__":
    main()
