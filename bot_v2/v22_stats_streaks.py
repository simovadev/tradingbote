"""v22_stats_streaks.py - Statistique des series de pertes consecutives.

Permet de savoir : si je perds N trades de suite en live, est-ce normal ou anormal ?
On calcule sur 3 ans :
- Pire serie de pertes consecutives par actif et globale
- Distribution des series (combien de fois on a perdu 1, 2, 3, 5, 10 fois de suite)
- Seuil d'alerte (95e percentile = au-dela = probleme suspect)
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


def get_pnls_chronological(asset_name, file_name, spread):
    """Retourne la liste des pnls en ordre chrono pour cet actif."""
    try:
        df_m5 = pd.read_parquet(DATA / f"{file_name}_M5.parquet")[["open","high","low","close"]]
        df_h1 = pd.read_parquet(DATA / f"{file_name}_H1.parquet")[["open","high","low","close"]]
        df_d1 = pd.read_parquet(DATA / f"{file_name}_D1.parquet")[["open","high","low","close"]]
    except Exception:
        return []
    start = pd.Timestamp("2023-01-01", tz="UTC"); end = pd.Timestamp("2026-04-01", tz="UTC")
    ctx = start - pd.Timedelta(days=10)
    df_m5 = df_m5[(df_m5.index >= ctx) & (df_m5.index < end)]
    df_h1 = df_h1[(df_h1.index >= ctx) & (df_h1.index < end)]
    df_d1 = df_d1[(df_d1.index >= start - pd.Timedelta(days=30)) & (df_d1.index < end)]
    if len(df_m5) < 1000: return []
    obs = find_obs_simple(df_m5, as_of_index=len(df_m5) - 1)
    obs = [ob for ob in obs if start <= ob.validation_ts < end]
    obs.sort(key=lambda x: x.validation_ts)   # ordre chrono
    pnls = []
    for ob in obs:
        d1a, h1a, pda = get_filters_for_ob(df_m5, df_h1, df_d1, ob)
        if not (d1a and h1a and pda): continue
        atr = atr_safe(df_m5, ob.validation_index)
        if atr <= 0: continue
        p = simulate_ob_trade(df_m5, ob, atr, spread=spread)
        if p is not None: pnls.append(p)
    return pnls


def analyze_streaks(pnls):
    """Retourne (max_loss_streak, distribution, count_runs)."""
    streaks_loss = []  # liste des longueurs de series LOSS
    streaks_win = []
    cur = 0; cur_sign = None
    for p in pnls:
        sign = -1 if p < 0 else (1 if p > 0 else 0)
        if sign == 0:   # neutre = on continue ce qu'on faisait, mais on l'ignore
            continue
        if cur_sign is None or sign == cur_sign:
            cur += 1
            cur_sign = sign
        else:
            if cur_sign == -1: streaks_loss.append(cur)
            elif cur_sign == 1: streaks_win.append(cur)
            cur = 1; cur_sign = sign
    if cur > 0:
        if cur_sign == -1: streaks_loss.append(cur)
        elif cur_sign == 1: streaks_win.append(cur)
    return streaks_loss, streaks_win


def main():
    print("=" * 90)
    print("STATISTIQUE DES SERIES DE PERTES CONSECUTIVES")
    print("=" * 90)
    print("\nSur quoi te repondre : combien de pertes d'affilee est-ce NORMAL ?")
    print("Au-dela du percentile 99 = anormal, probleme suspect en live.\n")

    all_pnls = []
    per_asset = {}
    for user_name, file_name in ASSETS_MAP.items():
        sp = SPREADS[user_name]
        print(f"  Calcul {user_name} ...", end=" ", flush=True)
        pnls = get_pnls_chronological(user_name, file_name, sp)
        print(f"{len(pnls)} trades")
        per_asset[user_name] = pnls
        all_pnls.extend(pnls)

    print(f"\n=== PAR ACTIF ===\n")
    print(f"{'actif':<10} {'trades':<8} {'max_loss':<10} {'p95_loss':<10} {'p99_loss':<10} {'max_win'}")
    print("-" * 70)
    for asset, pnls in per_asset.items():
        if not pnls: print(f"{asset:<10} (vide)"); continue
        sl, sw = analyze_streaks(pnls)
        max_l = max(sl) if sl else 0
        p95 = int(np.percentile(sl, 95)) if sl else 0
        p99 = int(np.percentile(sl, 99)) if sl else 0
        max_w = max(sw) if sw else 0
        print(f"{asset:<10} {len(pnls):<8} {max_l:<10} {p95:<10} {p99:<10} {max_w}")

    # Global
    print(f"\n=== GLOBAL (tous actifs confondus) ===\n")
    if all_pnls:
        sl_all, sw_all = analyze_streaks(all_pnls)
        max_l = max(sl_all) if sl_all else 0
        p95 = int(np.percentile(sl_all, 95)) if sl_all else 0
        p99 = int(np.percentile(sl_all, 99)) if sl_all else 0
        max_w = max(sw_all) if sw_all else 0
        print(f"  Total trades : {len(all_pnls)}")
        print(f"  Total series LOSS : {len(sl_all)}")
        print(f"  Pire serie LOSS jamais vue : {max_l} pertes consecutives")
        print(f"  95e percentile : {p95} pertes consecutives (= au-dela = top 5% des cas)")
        print(f"  99e percentile : {p99} pertes consecutives (= au-dela = top 1%)")
        print(f"  Pire serie WIN : {max_w} wins consecutifs")

        # Distribution
        print(f"\n  Distribution des series LOSS :")
        from collections import Counter
        c = Counter(sl_all)
        print(f"  {'n_pertes_d_affilee':<22} {'occurrences':<12} {'%cumul'}")
        cumul = 0
        total = sum(c.values())
        for n in sorted(c.keys()):
            cumul += c[n]
            pct = cumul / total * 100
            print(f"  {n:<22} {c[n]:<12} {pct:.1f}%")

    # Interpretation
    print(f"\n=== INTERPRETATION ===\n")
    print(f"Si en live tu fais X pertes consecutives, c'est :")
    print(f"  <= p95 : NORMAL, simple variance")
    print(f"  entre p95 et p99 : INHABITUEL mais possible")
    print(f"  > p99 (>= {p99 + 1 if all_pnls else 0}) : ANORMAL, regarder ce qui se passe")
    print(f"  > max ({max_l if all_pnls else 0}) : JAMAIS vu en backtest => probleme serieux probable")


if __name__ == "__main__":
    main()
