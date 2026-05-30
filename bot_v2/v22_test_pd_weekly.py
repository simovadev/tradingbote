"""v22_test_pd_weekly.py - Tester si ajouter le PD Weekly ameliore l'edge.

3 configs compares :
A. Baseline : D1 + H1 + PD daily (la regle actuelle)
B. + PD weekly : D1 + H1 + PD daily + PD weekly
C. PD weekly seul : D1 + H1 + PD weekly (sans daily)

PD weekly : on prend high/low de la semaine PRECEDENTE (lundi-vendredi precedent).
Mid week = (high_w-1 + low_w-1) / 2
- Bullish OK : price < mid_w-1 (discount weekly)
- Bearish OK : price > mid_w-1 (premium weekly)
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


def pd_zone_weekly_safe(df_d1, target_ts):
    """Retourne dict avec {high_w, low_w, mid_w} de la semaine PRECEDENTE.
    Une semaine = lundi 00h -> samedi 00h.
    SAFE : on prend uniquement les D1 STRICTEMENT avant le lundi de la semaine en cours.
    """
    if df_d1 is None or len(df_d1) == 0:
        return None
    target_day = pd.Timestamp(target_ts).normalize()
    # Trouver le lundi de la semaine en cours
    dow = target_day.weekday()   # 0 = lundi
    monday_curr = target_day - pd.Timedelta(days=dow)
    # Semaine precedente = monday_curr - 7 jours -> monday_curr - 1 second
    monday_prev = monday_curr - pd.Timedelta(days=7)
    # On veut les D1 dont l'index est entre monday_prev et monday_curr (strictement <)
    df_week = df_d1[(df_d1.index >= monday_prev) & (df_d1.index < monday_curr)]
    if len(df_week) == 0:
        return None
    high_w = float(df_week["high"].max())
    low_w = float(df_week["low"].min())
    mid_w = (high_w + low_w) / 2
    return {"high": high_w, "low": low_w, "mid": mid_w}


def get_filters_for_ob(df_m5, df_h1, df_d1, ob):
    """Retourne dict avec d1_a, h1_a, pd_a (daily), pd_weekly_a."""
    val_ts = ob.validation_ts; val_idx = ob.validation_index

    # D1 bias
    d1 = daily_bias_safe(df_d1, val_ts)
    d1_a = False
    if d1["ok"]:
        d1_h = d1["bias"] == "haussier"
        d1_a = (ob.direction == "bullish" and d1_h) or (ob.direction == "bearish" and not d1_h)

    # H1 trend
    h1_close = htf_value_at_t(df_h1, val_ts, "close")
    h1_a = False
    if h1_close is not None:
        pos = df_h1.index.searchsorted(val_ts, side="left") - 1 - 10
        if pos >= 0:
            h1_old = float(df_h1["close"].iloc[pos])
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

    # PD weekly
    pd_weekly_a = False
    zw = pd_zone_weekly_safe(df_d1, val_ts)
    if zw is not None:
        price = float(df_m5["close"].iloc[val_idx])
        pd_weekly_a = (
            (ob.direction == "bullish" and price < zw["mid"]) or
            (ob.direction == "bearish" and price > zw["mid"])
        )

    return {"d1": d1_a, "h1": h1_a, "pd_daily": pd_a, "pd_weekly": pd_weekly_a}


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


def stats_pnls(pnls):
    if not pnls: return None
    pnls = np.array(pnls)
    wins = (pnls > 0).sum(); losses = (pnls < 0).sum()
    n = len(pnls); wr = wins / n * 100 if n else 0
    exp = pnls.mean(); total = pnls.sum()
    eq = np.cumsum(pnls); peak = np.maximum.accumulate(eq); dd = (peak - eq).max()
    return {"n": n, "wr": wr, "exp": exp, "total": total, "dd": float(dd)}


def run_config(asset_name, file_name, period_start, period_end, spread, config):
    """config = dict like {'d1': True, 'h1': True, 'pd_daily': True, 'pd_weekly': False}"""
    df_m5 = pd.read_parquet(DATA / f"{file_name}_M5.parquet")[["open","high","low","close"]]
    df_h1 = pd.read_parquet(DATA / f"{file_name}_H1.parquet")[["open","high","low","close"]]
    df_d1 = pd.read_parquet(DATA / f"{file_name}_D1.parquet")[["open","high","low","close"]]
    ctx = period_start - pd.Timedelta(days=20)
    df_m5 = df_m5[(df_m5.index >= ctx) & (df_m5.index < period_end)]
    df_h1 = df_h1[(df_h1.index >= ctx) & (df_h1.index < period_end)]
    df_d1 = df_d1[(df_d1.index >= period_start - pd.Timedelta(days=60)) & (df_d1.index < period_end)]
    if len(df_m5) < 1000: return None
    obs = find_obs_simple(df_m5, as_of_index=len(df_m5) - 1)
    obs = [ob for ob in obs if period_start <= ob.validation_ts < period_end]
    pnls = []
    for ob in obs:
        f = get_filters_for_ob(df_m5, df_h1, df_d1, ob)
        if not all(f[k] for k, v in config.items() if v):
            continue
        atr = atr_safe(df_m5, ob.validation_index)
        if atr <= 0: continue
        p = simulate_ob_trade(df_m5, ob, atr, spread=spread)
        if p is not None: pnls.append(p)
    return pnls


CONFIGS = {
    "A_baseline (D1+H1+PDday)":   {"d1": True, "h1": True, "pd_daily": True, "pd_weekly": False},
    "B_avec_PDweek (D1+H1+PDd+PDw)": {"d1": True, "h1": True, "pd_daily": True, "pd_weekly": True},
    "C_PDweek_seul (D1+H1+PDw)":   {"d1": True, "h1": True, "pd_daily": False, "pd_weekly": True},
}


def main():
    print("=" * 90)
    print("V22 - TEST AJOUT PD WEEKLY")
    print("=" * 90)
    print("\n3 configs comparees :")
    print("  A = ta regle actuelle (D1+H1+PD daily)")
    print("  B = ta regle + PD weekly en plus (encore plus selectif)")
    print("  C = D1+H1+PD weekly (sans daily)")
    full_start = pd.Timestamp("2023-01-01", tz="UTC")
    full_end = pd.Timestamp("2026-04-01", tz="UTC")

    for cfg_name, cfg in CONFIGS.items():
        print(f"\n{'=' * 70}")
        print(f"CONFIG : {cfg_name}")
        print(f"{'=' * 70}")
        print(f"{'actif':<10} {'n':<6} {'WR%':<7} {'exp_R':<8} {'total_R':<9} {'DD_R':<7} {'trades/an'}")
        print("-" * 70)
        pos_assets = 0
        total_n = 0; total_exp = []
        for user_name, file_name in ASSETS_MAP.items():
            sp = SPREADS[user_name]
            pnls = run_config(user_name, file_name, full_start, full_end, sp, cfg)
            if pnls is None or len(pnls) == 0:
                print(f"{user_name:<10} 0 trades"); continue
            s = stats_pnls(pnls)
            nb_an = s["n"] / 3.25
            verdict = "OK" if s["exp"] > 0.05 else ("=" if s["exp"] > 0 else "NEG")
            if s["exp"] > 0: pos_assets += 1
            total_n += s["n"]
            total_exp.append(s["exp"])
            print(f"{user_name:<10} {s['n']:<6} {s['wr']:<7.1f} {s['exp']:<+8.3f} {s['total']:<+9.1f} {s['dd']:<7.1f} {nb_an:.0f} {verdict}")
        if total_exp:
            avg_exp = np.mean(total_exp)
            print(f"\nMoyenne exp_R sur les actifs : {avg_exp:+.3f}R")
            print(f"Actifs profitables : {pos_assets}/{len(ASSETS_MAP)}")
            print(f"Total trades/an sur les 6 actifs : {total_n / 3.25:.0f}")


if __name__ == "__main__":
    main()
