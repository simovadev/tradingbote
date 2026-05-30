"""v22_test_sans_pd.py - Tester impact du filtre PD daily.

Configs comparees :
A - Baseline COMPLET (D1+H1+PD daily) - Option A choisie
B - SANS PD daily (D1+H1 seulement)
C - SANS H1 (D1+PD seulement)
D - SANS D1 (H1+PD seulement)

Pour chaque config on applique aussi le filtre Option A
(London+NY + H1 momentum > 0.3%).

On regarde quel filtre est le plus important.
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


def check_filters(df_m5, df_h1, df_d1, ob):
    """Retourne (d1_a, h1_a, pd_a, h1_mom, hour_fr) pour chaque OB."""
    val_ts = ob.validation_ts; val_idx = ob.validation_index
    # D1 bias
    d1 = daily_bias_safe(df_d1, val_ts)
    d1_a = False
    if d1["ok"]:
        d1_h = d1["bias"] == "haussier"
        d1_a = (ob.direction == "bullish" and d1_h) or (ob.direction == "bearish" and not d1_h)
    # H1 trend
    h1_close = htf_value_at_t(df_h1, val_ts, "close")
    h1_a = False; h1_mom = 0
    if h1_close is not None:
        pos = df_h1.index.searchsorted(val_ts, side="left") - 1 - 10
        if pos >= 0:
            h1_old = float(df_h1["close"].iloc[pos])
            if h1_old > 0:
                h1_mom = abs((h1_close - h1_old) / h1_old * 100)
            h1_h = h1_close > h1_old
            h1_a = (ob.direction == "bullish" and h1_h) or (ob.direction == "bearish" and not h1_h)
    # PD daily
    pd_a = False
    if df_d1 is not None and len(df_d1) >= 2:
        target_day = pd.Timestamp(val_ts).normalize()
        df_d1_past = df_d1[df_d1.index < target_day]
        if len(df_d1_past) >= 1:
            pdh = float(df_d1_past["high"].iloc[-1])
            pdl = float(df_d1_past["low"].iloc[-1])
            mid = (pdh + pdl) / 2
            price = float(df_m5["close"].iloc[val_idx])
            pd_a = (
                (ob.direction == "bullish" and price < mid) or
                (ob.direction == "bearish" and price > mid)
            )
    # Hour FR
    hour_fr = val_ts.tz_convert("Europe/Paris").hour + val_ts.tz_convert("Europe/Paris").minute / 60
    return d1_a, h1_a, pd_a, h1_mom, hour_fr


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


def run_config(asset, file_name, sp, use_d1=True, use_h1=True, use_pd=True, use_option_a=True):
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
        d1_a, h1_a, pd_a, h1_mom, hour_fr = check_filters(df_m5, df_h1, df_d1, ob)
        # Filtres requis
        if use_d1 and not d1_a: continue
        if use_h1 and not h1_a: continue
        if use_pd and not pd_a: continue
        # Option A : London+NY + H1 momentum > 0.3%
        if use_option_a:
            if not (8 <= hour_fr < 17): continue
            if h1_mom <= 0.3: continue
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
    print("TEST IMPACT DU FILTRE PD DAILY (et autres)")
    print("=" * 90)
    print("Tous les tests incluent Option A (London+NY + H1 mom > 0.3%)\n")

    configs = [
        ("1. COMPLET (D1+H1+PD) [Option A actuelle]", True, True, True),
        ("2. SANS PD daily (D1+H1)", True, True, False),
        ("3. SANS H1 trend (D1+PD)", True, False, True),
        ("4. SANS D1 bias (H1+PD)", False, True, True),
        ("5. D1 seul", True, False, False),
        ("6. H1 seul", False, True, False),
        ("7. PD seul", False, False, True),
        ("8. AUCUN filtre (juste Option A horaire/momentum)", False, False, False),
    ]

    for name, use_d1, use_h1, use_pd in configs:
        print(f"\n{'=' * 80}")
        print(f"CONFIG : {name}")
        print(f"{'=' * 80}")
        print(f"{'actif':<10} {'n':<6} {'WR%':<7} {'exp_R':<8} {'total':<9} {'DD':<7}")
        print("-" * 60)
        total_n = 0; total_exp = []; pos = 0
        for asset, file_name in ASSETS_MAP.items():
            sp = SPREADS[asset]
            print(f"  {asset}...", end=" ", flush=True)
            try:
                pnls = run_config(asset, file_name, sp, use_d1, use_h1, use_pd, use_option_a=True)
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
