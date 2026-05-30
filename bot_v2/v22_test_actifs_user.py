"""v22_test_actifs_user.py - Test V22 sur les 6 actifs principaux user.

Actifs : EURUSD, XAUUSD, NAS100, GER40, BTCUSD, USOUSD (= CL-OIL dans nos data)

Pour chaque actif on fait :
1. Backtest complet 2023-2026 avec ta regle (D1+H1+PD) - AVEC FRAIS
2. Walk-forward par annee (2023, 2024, 2025-2026)
3. Baseline sans filtre (pour comparer)

Verdict : actif profitable si exp_R > 0 sur la periode complete + sur 2/3 annees.
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

# Mapping user_name -> filename
ASSETS_MAP = {
    "EURUSD": "EURUSD",
    "XAUUSD": "XAUUSD",
    "NAS100": "NAS100",
    "GER40":  "GER40",
    "BTCUSD": "BTCUSD",
    "USOUSD": "CL-OIL",     # USOUSD = WTI Crude Oil
}

# Spreads + commission (en unite prix)
# Spread (broker) + commission ~= spread broker (compte ECN)
SPREADS = {
    "EURUSD": 0.00016,    # 0.8 pip + 0.8 pip commission
    "XAUUSD": 0.14,       # 0.07 + 0.07
    "NAS100": 1.0,        # 0.5 + 0.5
    "GER40":  0.8,        # 0.4 + 0.4
    "BTCUSD": 6.0,        # 3 + 3
    "USOUSD": 0.06,       # 0.03 + 0.03
}

SL_BUFFER_ATR = 0.1
TP_RR = 2.0
PARTIAL_R = 1.0
MAX_FILL_BARS = 20
MAX_HOLD_BARS = 50


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
            pdh = float(df_d1_past["high"].iloc[-1])
            pdl = float(df_d1_past["low"].iloc[-1])
            mid = (pdh + pdl) / 2
            price = float(df_m5["close"].iloc[val_idx])
            pd_a = (
                (ob.direction == "bullish" and price < mid) or
                (ob.direction == "bearish" and price > mid)
            )
    return d1_a, h1_a, pd_a


def simulate_ob_trade(df_m5, ob, atr, spread=0.0):
    val_idx = ob.validation_index
    if val_idx + 1 >= len(df_m5):
        return {"status": "NO_DATA", "pnl_r": None}

    hs = spread / 2
    if ob.direction == "bullish":
        sl = ob.ob_low - SL_BUFFER_ATR * atr
        entry = ob.ob_high
        entry_eff = entry + hs
        sl_eff = sl - hs
    else:
        sl = ob.ob_high + SL_BUFFER_ATR * atr
        entry = ob.ob_low
        entry_eff = entry - hs
        sl_eff = sl + hs

    if abs(entry_eff - sl_eff) < 1e-9:
        return {"status": "INVALID_RR", "pnl_r": None}

    risk = abs(entry_eff - sl_eff)
    if ob.direction == "bullish":
        tp_1r = entry_eff + PARTIAL_R * risk
        tp_2r = entry_eff + TP_RR * risk
    else:
        tp_1r = entry_eff - PARTIAL_R * risk
        tp_2r = entry_eff - TP_RR * risk

    fh = df_m5["high"].values[val_idx + 1:val_idx + 1 + MAX_FILL_BARS]
    fl = df_m5["low"].values[val_idx + 1:val_idx + 1 + MAX_FILL_BARS]
    if len(fh) == 0:
        return {"status": "NO_DATA", "pnl_r": None}

    entry_idx_rel = None
    for i in range(len(fh)):
        if ob.direction == "bullish":
            if fl[i] <= entry:
                if fl[i] <= sl:
                    return {"status": "INVALIDATED", "pnl_r": None}
                entry_idx_rel = i; break
        else:
            if fh[i] >= entry:
                if fh[i] >= sl:
                    return {"status": "INVALIDATED", "pnl_r": None}
                entry_idx_rel = i; break

    if entry_idx_rel is None:
        return {"status": "NO_FILL", "pnl_r": None}

    entry_global = val_idx + 1 + entry_idx_rel
    fe = entry_global + 1
    en = min(fe + MAX_HOLD_BARS, len(df_m5))
    if en - fe < 5:
        return {"status": "NO_FUTURE", "pnl_r": None}

    pfh = df_m5["high"].values[fe:en]; pfl = df_m5["low"].values[fe:en]
    half_locked = False; stop = sl_eff
    for i in range(len(pfh)):
        if ob.direction == "bullish":
            hit_sl = pfl[i] <= stop; hit_1r = pfh[i] >= tp_1r; hit_2r = pfh[i] >= tp_2r
        else:
            hit_sl = pfh[i] >= stop; hit_1r = pfl[i] <= tp_1r; hit_2r = pfl[i] <= tp_2r
        if hit_sl and hit_2r:
            if half_locked: return {"status": "FILLED_WIN_BE", "pnl_r": 0.5 * PARTIAL_R}
            return {"status": "FILLED_LOSS", "pnl_r": -1.0}
        if hit_sl:
            if half_locked: return {"status": "FILLED_WIN_BE", "pnl_r": 0.5 * PARTIAL_R}
            return {"status": "FILLED_LOSS", "pnl_r": -1.0}
        if not half_locked and hit_1r:
            half_locked = True; stop = entry_eff
            if hit_2r: return {"status": "FILLED_WIN_FULL", "pnl_r": 0.5 * PARTIAL_R + 0.5 * TP_RR}
        elif half_locked and hit_2r:
            return {"status": "FILLED_WIN_FULL", "pnl_r": 0.5 * PARTIAL_R + 0.5 * TP_RR}
    if half_locked: return {"status": "FILLED_TIME_BE", "pnl_r": 0.5 * PARTIAL_R}
    return {"status": "FILLED_TIME_FLAT", "pnl_r": 0.0}


def stats_pnls(pnls):
    if not pnls: return None
    pnls = np.array(pnls)
    wins = (pnls > 0).sum(); losses = (pnls < 0).sum()
    n = len(pnls); wr = wins / n * 100 if n else 0
    exp = pnls.mean(); total = pnls.sum()
    eq = np.cumsum(pnls); peak = np.maximum.accumulate(eq); dd = (peak - eq).max()
    return {"n": n, "wins": int(wins), "losses": int(losses),
            "wr": wr, "exp": exp, "total": total, "dd": float(dd)}


def run_backtest(asset_name, file_name, period_start, period_end, apply_filter, spread):
    try:
        df_m5 = pd.read_parquet(DATA / f"{file_name}_M5.parquet")[["open","high","low","close"]]
        df_h1 = pd.read_parquet(DATA / f"{file_name}_H1.parquet")[["open","high","low","close"]]
        df_d1 = pd.read_parquet(DATA / f"{file_name}_D1.parquet")[["open","high","low","close"]]
    except Exception as e:
        return None, f"data error : {e}"

    start = period_start; end = period_end
    ctx = start - pd.Timedelta(days=10)
    df_m5 = df_m5[(df_m5.index >= ctx) & (df_m5.index < end)]
    df_h1 = df_h1[(df_h1.index >= ctx) & (df_h1.index < end)]
    df_d1 = df_d1[(df_d1.index >= start - pd.Timedelta(days=30)) & (df_d1.index < end)]
    if len(df_m5) < 1000: return None, f"trop peu data ({len(df_m5)})"

    obs = find_obs_simple(df_m5, as_of_index=len(df_m5) - 1)
    obs = [ob for ob in obs if start <= ob.validation_ts < end]

    pnls = []
    for ob in obs:
        if apply_filter:
            d1a, h1a, pda = get_filters_for_ob(df_m5, df_h1, df_d1, ob)
            if not (d1a and h1a and pda): continue
        atr = atr_safe(df_m5, ob.validation_index)
        if atr <= 0: continue
        r = simulate_ob_trade(df_m5, ob, atr, spread=spread)
        if r["pnl_r"] is not None:
            pnls.append(r["pnl_r"])

    return pnls, None


def main():
    print("=" * 90)
    print("V22 - TEST SUR LES 6 ACTIFS PRINCIPAUX USER")
    print("Avec frais reels + ta regle (D1+H1+PD daily)")
    print("=" * 90)

    full_start = pd.Timestamp("2023-01-01", tz="UTC")
    full_end = pd.Timestamp("2026-04-01", tz="UTC")

    # ============= TABLEAU GENERAL =============
    print(f"\n{'actif':<10} {'spread':<9} {'OBs':<6} {'WR%':<6} {'exp_R':<8} {'total_R':<9} {'DD_R':<7} {'~trades/an':<12} {'verdict'}")
    print("-" * 100)
    summary = []
    for user_name, file_name in ASSETS_MAP.items():
        sp = SPREADS[user_name]
        pnls, err = run_backtest(user_name, file_name, full_start, full_end, True, sp)
        if err:
            print(f"{user_name:<10} ERREUR : {err}"); continue
        if pnls is None or len(pnls) == 0:
            print(f"{user_name:<10} 0 trades"); continue
        s = stats_pnls(pnls)
        nb_an = s["n"] / 3.25   # ~3.25 ans
        verdict = "OK" if s["exp"] > 0.05 else ("faible" if s["exp"] > 0 else "NEG")
        print(f"{user_name:<10} {sp:<9.5f} {s['n']:<6} {s['wr']:<6.1f} {s['exp']:<+8.3f} {s['total']:<+9.1f} {s['dd']:<7.1f} {nb_an:<12.0f} {verdict}")
        summary.append((user_name, s, nb_an))

    # ============= WALK-FORWARD PAR ANNEE =============
    print("\n" + "=" * 90)
    print("WALK-FORWARD PAR ANNEE (avec ta regle + frais)")
    print("=" * 90)
    years = [
        ("2023", pd.Timestamp("2023-01-01",tz="UTC"), pd.Timestamp("2024-01-01",tz="UTC")),
        ("2024", pd.Timestamp("2024-01-01",tz="UTC"), pd.Timestamp("2025-01-01",tz="UTC")),
        ("2025", pd.Timestamp("2025-01-01",tz="UTC"), pd.Timestamp("2026-01-01",tz="UTC")),
        ("2026-Q1", pd.Timestamp("2026-01-01",tz="UTC"), pd.Timestamp("2026-04-01",tz="UTC")),
    ]
    print(f"\n{'actif':<10}", end="")
    for ylabel, _, _ in years:
        print(f"{ylabel:>14}", end="")
    print()
    print("-" * (10 + 14*len(years)))
    for user_name, file_name in ASSETS_MAP.items():
        sp = SPREADS[user_name]
        print(f"{user_name:<10}", end="")
        for ylabel, ys, ye in years:
            pnls, err = run_backtest(user_name, file_name, ys, ye, True, sp)
            if pnls is None or len(pnls) == 0:
                print(f"{'-':>14}", end="")
                continue
            s = stats_pnls(pnls)
            tag = "+" if s["exp"] > 0 else ("=" if s["exp"] == 0 else "-")
            print(f" {s['exp']:+.2f}R({s['n']:>3}){tag} ", end="")
        print()

    # ============= COMPARAISON SANS FILTRE =============
    print("\n" + "=" * 90)
    print("COMPARAISON BASELINE (sans filtre user) vs USER (avec D1+H1+PD)")
    print("=" * 90)
    print(f"\n{'actif':<10} {'BL_exp':<10} {'BL_WR':<8} {'USER_exp':<10} {'USER_WR':<8} {'GAIN':<8} {'ratio_kept'}")
    print("-" * 80)
    for user_name, file_name in ASSETS_MAP.items():
        sp = SPREADS[user_name]
        pnls_bl, _ = run_backtest(user_name, file_name, full_start, full_end, False, sp)
        pnls_user, _ = run_backtest(user_name, file_name, full_start, full_end, True, sp)
        if not pnls_bl or not pnls_user:
            print(f"{user_name:<10} data manquante"); continue
        s_bl = stats_pnls(pnls_bl); s_u = stats_pnls(pnls_user)
        ratio = len(pnls_user) / len(pnls_bl) * 100 if pnls_bl else 0
        gain = s_u['exp'] - s_bl['exp']
        print(f"{user_name:<10} {s_bl['exp']:<+10.3f} {s_bl['wr']:<8.1f} {s_u['exp']:<+10.3f} {s_u['wr']:<8.1f} {gain:<+8.3f} {ratio:.0f}%")

    # ============= SYNTHESE FINALE =============
    print("\n" + "=" * 90)
    print("SYNTHESE FINALE")
    print("=" * 90)
    profitable = [s for n, s, _ in summary if s["exp"] > 0]
    very_profitable = [s for n, s, _ in summary if s["exp"] > 0.10]
    print(f"Actifs profitables : {len(profitable)}/{len(summary)}")
    print(f"Actifs avec edge solide (>+0.10R) : {len(very_profitable)}/{len(summary)}")
    if summary:
        avg_exp = np.mean([s["exp"] for _, s, _ in summary])
        avg_wr = np.mean([s["wr"] for _, s, _ in summary])
        total_trades_yr = sum(nb for _, _, nb in summary)
        print(f"Esperance moyenne (par actif) : {avg_exp:+.3f}R")
        print(f"WR moyen (par actif) : {avg_wr:.1f}%")
        print(f"Total trades/an (somme actifs) : {total_trades_yr:.0f}")


if __name__ == "__main__":
    main()
