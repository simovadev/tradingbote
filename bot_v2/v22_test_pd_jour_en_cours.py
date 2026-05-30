"""v22_test_pd_jour_en_cours.py - Tester PD daily JOUR EN COURS.

Au lieu d'utiliser PDH/PDL du jour PRECEDENT, on utilise high/low
du jour EN COURS (depuis 00h UTC jusqu'au moment du trade).

Avantage : le mid evolue avec le marche, donc on peut acheter en
"discount du jour" meme si le marche est sorti du range de la veille.

3 configs comparees :
A - PD daily J-1 (regle actuelle baseline)
B - PD daily jour en cours (au moins 6 bougies M5)
C - PD daily jour en cours (au moins 12 bougies M5)
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


def pd_zone_current_day(df_m5, val_idx, val_ts, min_bars=6):
    """PD daily du JOUR EN COURS : high/low entre 00h UTC et val_ts strict.

    Retourne (mid, zone) ou (None, None) si pas assez de bougies.
    """
    # 00h UTC du jour de val_ts
    day_start = pd.Timestamp(val_ts).normalize()
    # Bougies du jour en cours STRICTEMENT avant val_ts (anti-leak : on ne prend pas la bougie val_idx elle-meme)
    # En realite val_idx est le moment T, on prend les bougies depuis day_start jusqu'a val_idx exclus
    mask_today = (df_m5.index >= day_start) & (df_m5.index < val_ts)
    sub = df_m5[mask_today]
    if len(sub) < min_bars:
        return None, None
    high_today = float(sub["high"].max())
    low_today = float(sub["low"].min())
    mid = (high_today + low_today) / 2
    return mid, (high_today, low_today)


def pd_zone_previous_day(df_d1, val_ts):
    """PD daily J-1 : high/low du jour precedent."""
    target_day = pd.Timestamp(val_ts).normalize()
    df_d1_past = df_d1[df_d1.index < target_day]
    if len(df_d1_past) < 1:
        return None, None
    pdh = float(df_d1_past["high"].iloc[-1])
    pdl = float(df_d1_past["low"].iloc[-1])
    mid = (pdh + pdl) / 2
    return mid, (pdh, pdl)


def passes_user_rule(df_m5, df_h1, df_d1, ob, pd_mode="prev", min_bars_today=6):
    """Filtre user avec choix du mode PD daily.

    pd_mode :
      'prev' : PDH/PDL du jour precedent (regle actuelle)
      'curr_6' : PD du jour en cours (min 6 bougies M5 = 30 min)
      'curr_12' : PD du jour en cours (min 12 bougies = 1h)
    """
    val_ts = ob.validation_ts; val_idx = ob.validation_index
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
    if pd_mode == "prev":
        mid, _ = pd_zone_previous_day(df_d1, val_ts)
    elif pd_mode == "curr_6":
        mid, _ = pd_zone_current_day(df_m5, val_idx, val_ts, min_bars=6)
    elif pd_mode == "curr_12":
        mid, _ = pd_zone_current_day(df_m5, val_idx, val_ts, min_bars=12)
    else:
        return False
    if mid is None: return False
    price = float(df_m5["close"].iloc[val_idx])
    pd_a = (
        (ob.direction == "bullish" and price < mid) or
        (ob.direction == "bearish" and price > mid)
    )
    return pd_a


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


def run_config(asset, file_name, pd_mode, sp):
    """Backtest 1 actif avec mode PD donne. Retourne pnls."""
    df_m5 = pd.read_parquet(DATA / f"{file_name}_M5.parquet")[["open","high","low","close"]]
    df_h1 = pd.read_parquet(DATA / f"{file_name}_H1.parquet")[["open","high","low","close"]]
    df_d1 = pd.read_parquet(DATA / f"{file_name}_D1.parquet")[["open","high","low","close"]]
    start = pd.Timestamp("2023-01-01", tz="UTC")
    end = pd.Timestamp("2026-04-01", tz="UTC")
    ctx = start - pd.Timedelta(days=10)
    df_m5 = df_m5[(df_m5.index >= ctx) & (df_m5.index < end)]
    df_h1 = df_h1[(df_h1.index >= ctx) & (df_h1.index < end)]
    df_d1 = df_d1[(df_d1.index >= start - pd.Timedelta(days=30)) & (df_d1.index < end)]
    if len(df_m5) < 1000: return []
    obs = find_obs_simple(df_m5, as_of_index=len(df_m5) - 1)
    obs = [ob for ob in obs if start <= ob.validation_ts < end]
    pnls = []
    for ob in obs:
        if not passes_user_rule(df_m5, df_h1, df_d1, ob, pd_mode=pd_mode): continue
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
    print("TEST PD DAILY JOUR EN COURS vs JOUR PRECEDENT")
    print("=" * 90)
    print("\n3 modes testes :")
    print("  PREV     = PDH/PDL du jour PRECEDENT (regle actuelle baseline)")
    print("  CURR_6   = high/low du jour EN COURS (min 6 bougies M5 = 30 min apres 00h)")
    print("  CURR_12  = high/low du jour EN COURS (min 12 bougies M5 = 1h apres 00h)")

    for pd_mode in ["prev", "curr_6", "curr_12"]:
        print(f"\n{'=' * 70}")
        print(f"MODE : {pd_mode.upper()}")
        print(f"{'=' * 70}")
        print(f"{'actif':<10} {'n':<6} {'WR%':<7} {'exp_R':<8} {'total_R':<9} {'DD_R':<7}")
        print("-" * 60)
        total_n = 0; total_exp = []; pos_count = 0
        for user_name, file_name in ASSETS_MAP.items():
            sp = SPREADS[user_name]
            print(f"  Loading {user_name}...", end=" ", flush=True)
            try:
                pnls = run_config(user_name, file_name, pd_mode, sp)
            except Exception as e:
                print(f"ERR {e}"); continue
            if not pnls:
                print("0 trades"); continue
            s = stats(pnls)
            verdict = "OK" if s["exp"] > 0.05 else ("=" if s["exp"] > 0 else "NEG")
            if s["exp"] > 0: pos_count += 1
            total_n += s["n"]; total_exp.append(s["exp"])
            print(f"{s['n']:<6} {s['wr']:<7.1f} {s['exp']:<+8.3f} {s['total']:<+9.1f} {s['dd']:<7.1f} {verdict}")
        avg = np.mean(total_exp) if total_exp else 0
        print(f"\n  Moyenne exp_R : {avg:+.3f}R")
        print(f"  Actifs profitables : {pos_count}/{len(ASSETS_MAP)}")
        print(f"  Total trades/an (somme) : {total_n / 3.25:.0f}")


if __name__ == "__main__":
    main()
