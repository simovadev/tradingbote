"""v22_mesure_concurrence.py - Mesure le nombre max et moyen de trades simultanes.

Pour chaque actif on simule les trades comme avant, puis on calcule :
- Combien de trades sont OUVERTS en meme temps (max + moyenne)
- Distribution : combien de temps on a 0, 1, 2 .. N positions ouvertes
- Risque cumule max (= nb positions x 1% pseudo-risque)
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


def get_filters_for_ob(df_m5, df_h1, df_d1, ob):
    val_ts = ob.validation_ts; val_idx = ob.validation_index
    d1 = daily_bias_safe(df_d1, val_ts)
    if not d1["ok"]: return False, False, False
    d1_h = d1["bias"] == "haussier"
    d1_a = (ob.direction == "bullish" and d1_h) or (ob.direction == "bearish" and not d1_h)
    h1_close = htf_value_at_t(df_h1, val_ts, "close")
    h1_a = False
    if h1_close is not None:
        pos = df_h1.index.searchsorted(val_ts, side="left") - 1 - 10
        if pos >= 0:
            h1_old = float(df_h1["close"].iloc[pos])
            h1_h = h1_close > h1_old
            h1_a = (ob.direction == "bullish" and h1_h) or (ob.direction == "bearish" and not h1_h)
    pd_a = False
    if df_d1 is not None and len(df_d1) >= 2:
        target_day = pd.Timestamp(val_ts).normalize()
        df_d1_past = df_d1[df_d1.index < target_day]
        if len(df_d1_past) >= 1:
            pdh = float(df_d1_past["high"].iloc[-1]); pdl = float(df_d1_past["low"].iloc[-1])
            mid = (pdh + pdl) / 2; price = float(df_m5["close"].iloc[val_idx])
            pd_a = (ob.direction == "bullish" and price < mid) or (ob.direction == "bearish" and price > mid)
    return d1_a, h1_a, pd_a


def simulate_ob_trade_get_window(df_m5, ob, atr, spread=0.0):
    """Comme simulate_ob_trade mais retourne (status, entry_ts, exit_ts) au lieu du pnl seul."""
    val_idx = ob.validation_index
    if val_idx + 1 >= len(df_m5):
        return None
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
    entry_ts = df_m5.index[entry_global]
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
        exit_global = fe + i
        if hit_sl or hit_2r:
            return (entry_ts, df_m5.index[exit_global])
        if not half_locked and hit_1r:
            half_locked = True; stop = entry_eff
            if hit_2r: return (entry_ts, df_m5.index[exit_global])
    # Time exit
    return (entry_ts, df_m5.index[en - 1])


def get_trade_windows(asset_name, file_name, spread):
    """Retourne liste de (entry_ts, exit_ts, asset) pour cet actif."""
    try:
        df_m5 = pd.read_parquet(DATA / f"{file_name}_M5.parquet")[["open","high","low","close"]]
        df_h1 = pd.read_parquet(DATA / f"{file_name}_H1.parquet")[["open","high","low","close"]]
        df_d1 = pd.read_parquet(DATA / f"{file_name}_D1.parquet")[["open","high","low","close"]]
    except Exception as e:
        return []
    start = pd.Timestamp("2023-01-01", tz="UTC"); end = pd.Timestamp("2026-04-01", tz="UTC")
    ctx = start - pd.Timedelta(days=10)
    df_m5 = df_m5[(df_m5.index >= ctx) & (df_m5.index < end)]
    df_h1 = df_h1[(df_h1.index >= ctx) & (df_h1.index < end)]
    df_d1 = df_d1[(df_d1.index >= start - pd.Timedelta(days=30)) & (df_d1.index < end)]
    if len(df_m5) < 1000: return []
    obs = find_obs_simple(df_m5, as_of_index=len(df_m5) - 1)
    obs = [ob for ob in obs if start <= ob.validation_ts < end]
    windows = []
    for ob in obs:
        d1a, h1a, pda = get_filters_for_ob(df_m5, df_h1, df_d1, ob)
        if not (d1a and h1a and pda): continue
        atr = atr_safe(df_m5, ob.validation_index)
        if atr <= 0: continue
        w = simulate_ob_trade_get_window(df_m5, ob, atr, spread=spread)
        if w is not None:
            windows.append((w[0], w[1], asset_name))
    return windows


def main():
    print("=" * 90)
    print("MESURE DE CONCURRENCE - Combien de trades ouverts en MEME TEMPS ?")
    print("=" * 90)

    # Recupere tous les trades
    all_windows = []
    for user_name, file_name in ASSETS_MAP.items():
        sp = SPREADS[user_name]
        print(f"  Calcul {user_name} ...", end=" ", flush=True)
        windows = get_trade_windows(user_name, file_name, sp)
        print(f"{len(windows)} trades")
        all_windows.extend(windows)

    print(f"\nTotal trades sur 6 actifs : {len(all_windows)}")
    if not all_windows: return

    # Construit une serie d'evenements (+1 entry, -1 exit) et calcule
    # le nb de positions ouvertes a chaque instant
    events = []
    for entry_ts, exit_ts, asset in all_windows:
        events.append((entry_ts, +1, asset))
        events.append((exit_ts, -1, asset))
    events.sort(key=lambda e: e[0])

    open_count = 0
    max_open = 0; max_open_ts = None
    # On enregistre le timeseries de open_count avec duration entre events
    timeseries = []   # list of (start_ts, end_ts, n_open)
    prev_ts = None
    for ts, delta, asset in events:
        if prev_ts is not None:
            timeseries.append((prev_ts, ts, open_count))
        open_count += delta
        if open_count > max_open:
            max_open = open_count; max_open_ts = ts
        prev_ts = ts

    # Statistiques pondérées par durée
    total_time_sec = 0
    time_by_n = {}
    for s, e, n in timeseries:
        duration = (e - s).total_seconds()
        total_time_sec += duration
        time_by_n[n] = time_by_n.get(n, 0) + duration

    # Pourcentages
    pct_by_n = {n: t / total_time_sec * 100 for n, t in time_by_n.items()} if total_time_sec else {}

    print(f"\n=== RESULTATS GLOBAUX ===")
    print(f"Max positions ouvertes simultanees : {max_open}")
    print(f"  -> a {max_open_ts}")
    print()
    print(f"Distribution du temps passe avec N positions ouvertes :")
    print(f"{'N':<5} {'% temps':<10} {'cumul'}")
    print("-" * 40)
    cumul = 0
    for n in sorted(pct_by_n.keys()):
        cumul += pct_by_n[n]
        print(f"{n:<5} {pct_by_n[n]:<10.1f} {cumul:.1f}%")

    # Mediane et moyenne
    weighted_sum = sum(n * t for n, t in time_by_n.items())
    avg_open = weighted_sum / total_time_sec if total_time_sec else 0
    print(f"\nMoyenne positions ouvertes (ponderee par duree) : {avg_open:.2f}")

    # Pourcentage du temps avec PLUS de 5, 10, 15 ouvertes
    for threshold in [5, 10, 15, 20]:
        pct = sum(t for n, t in time_by_n.items() if n > threshold) / total_time_sec * 100 if total_time_sec else 0
        print(f"% temps avec > {threshold} positions ouvertes : {pct:.2f}%")

    # Decomposition par actif - combien d'actifs ouverts en moyenne
    print(f"\n=== MAX SIMULTANE PAR ACTIF ===")
    # On compte par actif quel max il atteint
    open_by_asset = {a: 0 for a in ASSETS_MAP.keys()}
    max_by_asset = {a: 0 for a in ASSETS_MAP.keys()}
    open_count = 0
    for ts, delta, asset in events:
        open_by_asset[asset] += delta
        if open_by_asset[asset] > max_by_asset[asset]:
            max_by_asset[asset] = open_by_asset[asset]
    for a, mx in max_by_asset.items():
        print(f"  {a:<10} max simultane : {mx}")


if __name__ == "__main__":
    main()
