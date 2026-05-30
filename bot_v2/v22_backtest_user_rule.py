"""v22_backtest_user_rule.py - Backtest la VRAIE strategie user :

OBs (definition simple) + Filtre TA REGLE : D1 + H1 alignes + Discount/Premium daily
Plan trade : Entry MARKET sur retour zone OB, SL sous OB, TP 2R partial 1R
INVALIDATION : si low casse ob_low avant entry -> skip

On mesure :
- nb OBs detectes
- nb OBs traades (apres filtre regle)
- nb OBs fillables (retour toucher la zone)
- WR + esperance R + drawdown
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

ASSET = "XAUUSD"
START_DATE = "2023-01-01"
END_DATE = "2026-04-01"

# Plan trade
SL_BUFFER_ATR = 0.1       # SL = ob_low - 0.1 ATR (sous OB)
TP_RR = 2.0
PARTIAL_R = 1.0
MAX_FILL_BARS = 20        # max 20 bougies M5 pour que le prix revienne toucher
MAX_HOLD_BARS = 50        # max 50 bougies en position (~4h)

# Filtres user
APPLY_USER_FILTER = True   # D1+H1 alignes + PD daily


def get_filters_for_ob(df_m5, df_h1, df_d1, ob):
    """Retourne (d1_aligned, h1_aligned, pd_aligned)."""
    val_ts = ob.validation_ts
    val_idx = ob.validation_index

    # D1 bias
    d1 = daily_bias_safe(df_d1, val_ts)
    if not d1["ok"]:
        return False, False, False
    d1_haussier = d1["bias"] == "haussier"
    d1_aligned = (
        (ob.direction == "bullish" and d1_haussier) or
        (ob.direction == "bearish" and not d1_haussier)
    )

    # H1 trend (close vs 10h plus tot)
    h1_close = htf_value_at_t(df_h1, val_ts, "close")
    h1_aligned = False
    if h1_close is not None:
        pos = df_h1.index.searchsorted(val_ts, side="left") - 1 - 10
        if pos >= 0:
            h1_close_old = float(df_h1["close"].iloc[pos])
            h1_haussier = h1_close > h1_close_old
            h1_aligned = (
                (ob.direction == "bullish" and h1_haussier) or
                (ob.direction == "bearish" and not h1_haussier)
            )

    # PD daily : pour bullish il faut etre en discount (sous mid PDH-PDL)
    pd_aligned = False
    if df_d1 is not None and len(df_d1) >= 2:
        target_day = pd.Timestamp(val_ts).normalize()
        df_d1_past = df_d1[df_d1.index < target_day]
        if len(df_d1_past) >= 1:
            pdh = float(df_d1_past["high"].iloc[-1])
            pdl = float(df_d1_past["low"].iloc[-1])
            mid = (pdh + pdl) / 2
            price = float(df_m5["close"].iloc[val_idx])
            in_discount = price < mid
            in_premium = price > mid
            pd_aligned = (
                (ob.direction == "bullish" and in_discount) or
                (ob.direction == "bearish" and in_premium)
            )

    return d1_aligned, h1_aligned, pd_aligned


def simulate_ob_trade(df_m5, ob, atr):
    """Simule un trade sur cet OB.

    Retourne dict:
    - status : 'FILLED_WIN_FULL' / 'FILLED_WIN_BE' / 'FILLED_LOSS' / 'NO_FILL' / 'INVALIDATED'
    - pnl_r : pnl en R (None si pas trade)
    """
    val_idx = ob.validation_index
    if val_idx + 1 >= len(df_m5):
        return {"status": "NO_DATA", "pnl_r": None}

    # SL = sous ob_low pour bullish, au-dessus ob_high pour bearish
    if ob.direction == "bullish":
        sl = ob.ob_low - SL_BUFFER_ATR * atr
        # Entry market quand le prix RETOUCHE la zone OB
        entry = ob.ob_high   # on entre des que le prix retombe a ob_high
    else:
        sl = ob.ob_high + SL_BUFFER_ATR * atr
        entry = ob.ob_low

    if abs(entry - sl) < 1e-9:
        return {"status": "INVALID_RR", "pnl_r": None}

    risk = abs(entry - sl)
    if ob.direction == "bullish":
        tp_1r = entry + PARTIAL_R * risk
        tp_2r = entry + TP_RR * risk
    else:
        tp_1r = entry - PARTIAL_R * risk
        tp_2r = entry - TP_RR * risk

    # Phase 1 : attendre que le prix retouche la zone OB (entry)
    # Limite : MAX_FILL_BARS bougies
    fh = df_m5["high"].values[val_idx + 1:val_idx + 1 + MAX_FILL_BARS]
    fl = df_m5["low"].values[val_idx + 1:val_idx + 1 + MAX_FILL_BARS]

    entry_idx_rel = None
    for i in range(len(fh)):
        if ob.direction == "bullish":
            # Pour LONG : on attend que le prix REDESCENDE toucher la zone OB
            # Entry triggered si low de la bougie <= ob_high (entree dans la zone)
            # Invalidation si low < ob_low SANS d'abord avoir touche entry
            # (i.e. si une seule bougie casse l'OB tout droit)
            if fl[i] <= entry:
                # Entry touchee. Mais si dans la MEME bougie le low casse aussi sl
                # = invalidate (gap-down brutal)
                if fl[i] <= sl:
                    return {"status": "INVALIDATED", "pnl_r": None}
                entry_idx_rel = i; break
        else:
            # Pour SHORT : on attend que le prix REMONTE toucher la zone OB
            # Entry triggered si high >= ob_low
            if fh[i] >= entry:
                if fh[i] >= sl:
                    return {"status": "INVALIDATED", "pnl_r": None}
                entry_idx_rel = i; break

    if entry_idx_rel is None:
        return {"status": "NO_FILL", "pnl_r": None}

    # Phase 2 : position ouverte, on cherche SL ou TP
    entry_global = val_idx + 1 + entry_idx_rel
    future_start = entry_global + 1
    future_end = min(future_start + MAX_HOLD_BARS, len(df_m5))
    if future_end - future_start < 5:
        return {"status": "NO_FUTURE", "pnl_r": None}

    pf_h = df_m5["high"].values[future_start:future_end]
    pf_l = df_m5["low"].values[future_start:future_end]
    pf_o = df_m5["open"].values[future_start:future_end]

    half_locked = False
    stop = sl
    for i in range(len(pf_h)):
        if ob.direction == "bullish":
            hit_sl = pf_l[i] <= stop
            hit_1r = pf_h[i] >= tp_1r
            hit_2r = pf_h[i] >= tp_2r
        else:
            hit_sl = pf_h[i] >= stop
            hit_1r = pf_l[i] <= tp_1r
            hit_2r = pf_l[i] <= tp_2r

        # Conservative si sl et 2r touches dans meme bougie
        if hit_sl and hit_2r:
            if half_locked:
                return {"status": "FILLED_WIN_BE", "pnl_r": 0.5 * PARTIAL_R + 0.5 * 0.0}
            return {"status": "FILLED_LOSS", "pnl_r": -1.0}
        if hit_sl:
            if half_locked:
                return {"status": "FILLED_WIN_BE", "pnl_r": 0.5 * PARTIAL_R + 0.5 * 0.0}
            return {"status": "FILLED_LOSS", "pnl_r": -1.0}
        if not half_locked and hit_1r:
            half_locked = True
            stop = entry  # BE sur le reste
            if hit_2r:
                return {"status": "FILLED_WIN_FULL", "pnl_r": 0.5 * PARTIAL_R + 0.5 * TP_RR}
        elif half_locked and hit_2r:
            return {"status": "FILLED_WIN_FULL", "pnl_r": 0.5 * PARTIAL_R + 0.5 * TP_RR}

    # Time exit : reste a BE si half_locked
    if half_locked:
        return {"status": "FILLED_TIME_BE", "pnl_r": 0.5 * PARTIAL_R}
    return {"status": "FILLED_TIME_FLAT", "pnl_r": 0.0}


def main():
    print(f"\n=== V22 BACKTEST REGLE USER (OB simple + D1+H1+PD) ===")
    print(f"Actif : {ASSET}")
    print(f"Periode : {START_DATE} -> {END_DATE}")
    print(f"Plan : Entry market sur retour zone OB, SL sous OB, TP 2R partial 1R")
    print(f"Fill max : {MAX_FILL_BARS} bougies | Hold max : {MAX_HOLD_BARS} bougies")
    print()

    df_m5 = pd.read_parquet(DATA / f"{ASSET}_M5.parquet")[["open", "high", "low", "close"]]
    df_h1 = pd.read_parquet(DATA / f"{ASSET}_H1.parquet")[["open", "high", "low", "close"]]
    df_d1 = pd.read_parquet(DATA / f"{ASSET}_D1.parquet")[["open", "high", "low", "close"]]

    start = pd.Timestamp(START_DATE, tz="UTC")
    end = pd.Timestamp(END_DATE, tz="UTC")
    ctx = start - pd.Timedelta(days=10)
    df_m5 = df_m5[(df_m5.index >= ctx) & (df_m5.index < end)]
    df_h1 = df_h1[(df_h1.index >= ctx) & (df_h1.index < end)]
    df_d1 = df_d1[(df_d1.index >= start - pd.Timedelta(days=30)) & (df_d1.index < end)]
    print(f"M5 : {len(df_m5)}, H1 : {len(df_h1)}, D1 : {len(df_d1)}")

    print("\nDetect OBs ...")
    obs = find_obs_simple(df_m5, as_of_index=len(df_m5) - 1)
    obs = [ob for ob in obs if start <= ob.validation_ts < end]
    print(f"OBs detectes : {len(obs)}")

    # Filtre user regle
    obs_user = []
    obs_nofilter = obs
    for ob in obs:
        d1_a, h1_a, pd_a = get_filters_for_ob(df_m5, df_h1, df_d1, ob)
        if d1_a and h1_a and pd_a:
            obs_user.append(ob)
    print(f"OBs apres filtre user (D1+H1+PD) : {len(obs_user)} ({len(obs_user)/len(obs)*100:.1f}%)")

    # Pour la suite : on backteste avec ET sans filtre
    for label, obs_list in [("BASELINE (tous OBs)", obs_nofilter[:10000]),  # limit pour speed
                              ("USER FILTER", obs_user)]:
        print(f"\n--- {label} ({len(obs_list)} OBs) ---")
        outcomes = {}
        pnls = []
        for ob in obs_list:
            atr = atr_safe(df_m5, ob.validation_index)
            if atr <= 0: continue
            r = simulate_ob_trade(df_m5, ob, atr)
            outcomes[r["status"]] = outcomes.get(r["status"], 0) + 1
            if r["pnl_r"] is not None:
                pnls.append(r["pnl_r"])

        print(f"  Total trades simulees : {len(obs_list)}")
        for status in sorted(outcomes.keys(), key=lambda s: -outcomes[s]):
            print(f"    {status:<25} : {outcomes[status]:>6}")

        if pnls:
            pnls = np.array(pnls)
            wins = (pnls > 0).sum()
            losses = (pnls < 0).sum()
            neutrals = (pnls == 0).sum()
            n_trade = len(pnls)
            wr = wins / n_trade * 100 if n_trade > 0 else 0
            exp = pnls.mean()
            total = pnls.sum()
            eq = np.cumsum(pnls); peak = np.maximum.accumulate(eq)
            dd = (peak - eq).max()
            print(f"\n  Trades filles : {n_trade}")
            print(f"  WIN : {wins} | LOSS : {losses} | NEUTRE : {neutrals}")
            print(f"  WR : {wr:.1f}%")
            print(f"  Esperance : {exp:+.3f}R par trade")
            print(f"  Total R   : {total:+.1f}R")
            print(f"  Drawdown  : -{dd:.1f}R")
            # Simu compte 100 EUR risque FIXE 1 EUR par trade (pas compounding)
            # Compounding sur 4000+ trades = explosion irrealiste
            cap = 100.0; risk_eur = 1.0
            for p in pnls:
                cap += risk_eur * p
            yearly_trades = len(pnls) / 3  # periode 3 ans
            print(f"  Compte 100 EUR (risk FIXE 1 EUR/trade, ~{yearly_trades:.0f} trades/an) -> {cap:.2f} EUR")
            print(f"    Total {total:+.1f}R = {total:+.1f} EUR (1R = 1 EUR)")


if __name__ == "__main__":
    main()
