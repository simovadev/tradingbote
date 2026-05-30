"""v22_test_pd_post_asia.py - Tester PD jour-en-cours UNIQUEMENT apres l'Asie.

Modes testes :
A - BASELINE : PD J-1 (regle actuelle)
B - PD jour en cours apres 6h FR (Asia tardive)
C - PD jour en cours apres 8h FR (London open)
D - PD jour en cours OU PD J-1 (le + restrictif), apres 8h FR

Pour chaque mode :
- Le filtre PD daily est applique selon le mode
- Si mode != A, on n'autorise les trades qu'apres l'heure seuil
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

ASSETS_MAP = {
    "EURUSD": "EURUSD", "XAUUSD": "XAUUSD", "NAS100": "NAS100",
    "GER40":  "GER40",  "BTCUSD": "BTCUSD", "USOUSD": "CL-OIL",
}
SPREADS = {
    "EURUSD": 0.00016, "XAUUSD": 0.14, "NAS100": 1.0,
    "GER40":  0.8,     "BTCUSD": 6.0,  "USOUSD": 0.06,
}

SL_BUFFER_ATR = 0.1; TP_RR = 2.0; PARTIAL_R = 1.0
MAX_FILL_BARS = 20; MAX_HOLD_BARS = 50


def pd_zone_current_day(df_m5, val_idx, val_ts):
    """PD du jour en cours : high/low entre 00h UTC et val_ts."""
    day_start = pd.Timestamp(val_ts).normalize()
    mask = (df_m5.index >= day_start) & (df_m5.index < val_ts)
    sub = df_m5[mask]
    if len(sub) < 6:
        return None
    high = float(sub["high"].max())
    low = float(sub["low"].min())
    return (high + low) / 2


def pd_zone_prev_day(df_d1, val_ts):
    """PD J-1."""
    target_day = pd.Timestamp(val_ts).normalize()
    past = df_d1[df_d1.index < target_day]
    if len(past) < 1: return None
    pdh = float(past["high"].iloc[-1]); pdl = float(past["low"].iloc[-1])
    return (pdh + pdl) / 2


def passes_user_rule(df_m5, df_h1, df_d1, ob, pd_mode, min_hour_fr=0):
    val_ts = ob.validation_ts; val_idx = ob.validation_index
    # Filtre heure FR mini
    hour_fr = val_ts.tz_convert("Europe/Paris").hour + val_ts.tz_convert("Europe/Paris").minute / 60
    if hour_fr < min_hour_fr:
        return False
    # D1 bias
    d1 = daily_bias_safe(df_d1, val_ts)
    if not d1["ok"]: return False
    d1_h = d1["bias"] == "haussier"
    d1_a = (ob.direction == "bullish" and d1_h) or (ob.direction == "bearish" and not d1_h)
    if not d1_a: return False
    # H1 trend
    h1_close = htf_value_at_t(df_h1, val_ts, "close")
    if h1_close is None: return False
    pos = df_h1.index.searchsorted(val_ts, side="left") - 1 - 10
    if pos < 0: return False
    h1_old = float(df_h1["close"].iloc[pos])
    h1_h = h1_close > h1_old
    h1_a = (ob.direction == "bullish" and h1_h) or (ob.direction == "bearish" and not h1_h)
    if not h1_a: return False
    # PD daily selon mode
    price = float(df_m5["close"].iloc[val_idx])
    if pd_mode == "prev":
        mid = pd_zone_prev_day(df_d1, val_ts)
        if mid is None: return False
        pd_a = (ob.direction == "bullish" and price < mid) or (ob.direction == "bearish" and price > mid)
        return pd_a
    elif pd_mode == "curr":
        mid = pd_zone_current_day(df_m5, val_idx, val_ts)
        if mid is None: return False
        pd_a = (ob.direction == "bullish" and price < mid) or (ob.direction == "bearish" and price > mid)
        return pd_a
    elif pd_mode == "both":
        mid_prev = pd_zone_prev_day(df_d1, val_ts)
        mid_curr = pd_zone_current_day(df_m5, val_idx, val_ts)
        if mid_prev is None or mid_curr is None: return False
        pd_prev_a = (ob.direction == "bullish" and price < mid_prev) or (ob.direction == "bearish" and price > mid_prev)
        pd_curr_a = (ob.direction == "bullish" and price < mid_curr) or (ob.direction == "bearish" and price > mid_curr)
        return pd_prev_a and pd_curr_a   # les 2 doivent etre OK
    return False


def simulate_ob_trade(df_m5, ob, atr, spread=0.0):
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
                if fl[i] <= sl: return None
                entry_idx_rel = i; break
        else:
            if fh[i] >= entry:
                if fh[i] >= sl: return None
                entry_idx_rel = i; break
    if entry_idx_rel is None: return None
    entry_global = val_idx + 1 + entry_idx_rel
    fe = entry_global + 1
    en = min(fe + MAX_HOLD_BARS, len(df_m5))
    if en - fe < 5: return None
    pfh = df_m5["high"].values[fe:en]; pfl = df_m5["low"].values[fe:en]
    half_locked = False; stop = sl_eff
    for i in range(len(pfh)):
        if ob.direction == "bullish":
            hit_sl = pfl[i] <= stop; hit_1r = pfh[i] >= tp_1r; hit_2r = pfh[i] >= tp_2r
        else:
            hit_sl = pfh[i] >= stop; hit_1r = pfl[i] <= tp_1r; hit_2r = pfl[i] <= tp_2r
        if hit_sl and hit_2r:
            if half_locked: return 0.5 * PARTIAL_R
            return -1.0
        if hit_sl:
            if half_locked: return 0.5 * PARTIAL_R
            return -1.0
        if not half_locked and hit_1r:
            half_locked = True; stop = entry_eff
            if hit_2r: return 0.5 * PARTIAL_R + 0.5 * TP_RR
        elif half_locked and hit_2r:
            return 0.5 * PARTIAL_R + 0.5 * TP_RR
    if half_locked: return 0.5 * PARTIAL_R
    return 0.0


def run_asset(asset, file_name, pd_mode, min_hour_fr, sp):
    df_m5 = pd.read_parquet(DATA / f"{file_name}_M5.parquet")[["open","high","low","close"]]
    df_h1 = pd.read_parquet(DATA / f"{file_name}_H1.parquet")[["open","high","low","close"]]
    df_d1 = pd.read_parquet(DATA / f"{file_name}_D1.parquet")[["open","high","low","close"]]
    start = pd.Timestamp("2023-01-01", tz="UTC"); end = pd.Timestamp("2026-04-01", tz="UTC")
    ctx = start - pd.Timedelta(days=10)
    df_m5 = df_m5[(df_m5.index >= ctx) & (df_m5.index < end)]
    df_h1 = df_h1[(df_h1.index >= ctx) & (df_h1.index < end)]
    df_d1 = df_d1[(df_d1.index >= start - pd.Timedelta(days=30)) & (df_d1.index < end)]
    if len(df_m5) < 1000: return []
    obs = find_obs_simple(df_m5, as_of_index=len(df_m5) - 1)
    obs = [ob for ob in obs if start <= ob.validation_ts < end]
    pnls = []
    for ob in obs:
        if not passes_user_rule(df_m5, df_h1, df_d1, ob, pd_mode=pd_mode, min_hour_fr=min_hour_fr): continue
        atr = atr_safe(df_m5, ob.validation_index)
        if atr <= 0: continue
        p = simulate_ob_trade(df_m5, ob, atr, spread=sp)
        if p is not None: pnls.append(p)
    return pnls


def stats(pnls):
    if not pnls: return None
    pnls = np.array(pnls)
    wins = (pnls > 0).sum(); losses = (pnls < 0).sum()
    n = len(pnls); wr = wins / n * 100; exp = pnls.mean(); total = pnls.sum()
    eq = np.cumsum(pnls); peak = np.maximum.accumulate(eq); dd = (peak - eq).max()
    return {"n": n, "wr": wr, "exp": exp, "total": total, "dd": float(dd)}


def main():
    print("=" * 90)
    print("TEST PD DAILY POST-ASIE (commence apres 8h FR / London open)")
    print("=" * 90)
    configs = [
        ("A - BASELINE (PD J-1, sans filtre horaire)", "prev", 0),
        ("B - PD J-1 + apres 8h FR", "prev", 8),
        ("C - PD jour-en-cours + apres 6h FR (Asia tardive)", "curr", 6),
        ("D - PD jour-en-cours + apres 8h FR (London open)", "curr", 8),
        ("E - PD J-1 ET jour-en-cours + apres 8h FR", "both", 8),
        ("F - PD J-1 ET jour-en-cours + apres 6h FR", "both", 6),
    ]

    for name, pd_mode, min_hour in configs:
        print(f"\n{'=' * 70}")
        print(f"CONFIG : {name}")
        print(f"{'=' * 70}")
        print(f"{'actif':<10} {'n':<6} {'WR%':<7} {'exp_R':<8} {'total':<9} {'DD':<7}")
        print("-" * 60)
        total_n = 0; total_exp = []; pos = 0
        for asset, file_name in ASSETS_MAP.items():
            sp = SPREADS[asset]
            print(f"  {asset}...", end=" ", flush=True)
            try:
                pnls = run_asset(asset, file_name, pd_mode, min_hour, sp)
            except Exception as e:
                print(f"err {e}"); continue
            if not pnls: print("0"); continue
            s = stats(pnls)
            v = "OK" if s["exp"] > 0.05 else ("=" if s["exp"] > 0 else "NEG")
            if s["exp"] > 0: pos += 1
            total_n += s["n"]; total_exp.append(s["exp"])
            print(f"{s['n']:<6} {s['wr']:<7.1f} {s['exp']:<+8.3f} {s['total']:<+9.1f} {s['dd']:<7.1f} {v}")
        avg = np.mean(total_exp) if total_exp else 0
        print(f"\n  Moyenne exp_R : {avg:+.3f}R | profitables : {pos}/6 | trades/an : {total_n / 3.25:.0f}")


if __name__ == "__main__":
    main()
