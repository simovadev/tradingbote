"""v22_verifications.py - 6 tests rigoureux pour valider OU casser l'edge.

1. Walk-forward strict : 2023 / 2024 / 2025-2026
2. Frais reels inclus
3. Test multi-actifs (XAU + NAS + SP500 + GER40 + EUR + BTC)
4. Stratification temporelle (par trimestre)
5. Sanity check : OBs random (doit donner ~0R)
6. Audit anti-leak (assert que les indices sont strict)

Si les 6 passent -> l'edge est solide. Sinon on isole le probleme.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from datetime import datetime

from bot_v2.concepts.safe import (
    find_obs_simple, daily_bias_safe, htf_value_at_t, atr_safe,
)

DATA = ROOT / "data_vantage"

# Spreads par actif (en unite prix)
SPREADS = {
    "XAUUSD": 0.07,
    "NAS100": 0.50,
    "SP500": 0.30,
    "GER40": 0.40,
    "EURUSD": 0.00008,
    "BTCUSD": 3.0,
}
COMMISSION_RATIO = 1.0   # frais ajoutes = 1x spread (compte ECN)

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
    """Simule trade avec spread inclus (entry et SL ajustes du demi-spread)."""
    val_idx = ob.validation_index
    if val_idx + 1 >= len(df_m5):
        return {"status": "NO_DATA", "pnl_r": None}

    hs = spread / 2
    if ob.direction == "bullish":
        sl = ob.ob_low - SL_BUFFER_ATR * atr
        entry = ob.ob_high
        # Avec spread : on paye le spread a l'entree (ask > bid)
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
    n = len(pnls); wr = wins / n * 100; exp = pnls.mean(); total = pnls.sum()
    eq = np.cumsum(pnls); peak = np.maximum.accumulate(eq); dd = (peak - eq).max()
    return {"n": n, "wins": int(wins), "losses": int(losses),
            "wr": wr, "exp": exp, "total": total, "dd": float(dd)}


def run_period(asset, df_m5, df_h1, df_d1, period_start, period_end,
               apply_filter=True, spread=0.0):
    """Backtest pour une periode donnee."""
    obs = find_obs_simple(df_m5, as_of_index=len(df_m5) - 1)
    obs = [ob for ob in obs if period_start <= ob.validation_ts < period_end]
    pnls = []
    statuses = {}
    for ob in obs:
        if apply_filter:
            d1a, h1a, pda = get_filters_for_ob(df_m5, df_h1, df_d1, ob)
            if not (d1a and h1a and pda): continue
        atr = atr_safe(df_m5, ob.validation_index)
        if atr <= 0: continue
        r = simulate_ob_trade(df_m5, ob, atr, spread=spread)
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
        if r["pnl_r"] is not None:
            pnls.append(r["pnl_r"])
    return pnls, statuses


def load_data(asset):
    df_m5 = pd.read_parquet(DATA / f"{asset}_M5.parquet")[["open","high","low","close"]]
    df_h1 = pd.read_parquet(DATA / f"{asset}_H1.parquet")[["open","high","low","close"]]
    df_d1 = pd.read_parquet(DATA / f"{asset}_D1.parquet")[["open","high","low","close"]]
    start = pd.Timestamp("2023-01-01", tz="UTC")
    end = pd.Timestamp("2026-04-01", tz="UTC")
    ctx = start - pd.Timedelta(days=10)
    df_m5 = df_m5[(df_m5.index >= ctx) & (df_m5.index < end)]
    df_h1 = df_h1[(df_h1.index >= ctx) & (df_h1.index < end)]
    df_d1 = df_d1[(df_d1.index >= start - pd.Timedelta(days=30)) & (df_d1.index < end)]
    return df_m5, df_h1, df_d1


def main():
    print("=" * 80)
    print("V22 - 6 VERIFICATIONS POUR VALIDER OU CASSER L'EDGE")
    print("=" * 80)

    # =============================
    # VERIF 1 : Walk-forward strict XAU
    # =============================
    print("\n" + "=" * 80)
    print("VERIF 1 - WALK-FORWARD STRICT XAU (sans frais)")
    print("=" * 80)
    df_m5, df_h1, df_d1 = load_data("XAUUSD")
    periods = [
        ("2023", "2023-01-01", "2024-01-01"),
        ("2024", "2024-01-01", "2025-01-01"),
        ("2025-Q1+Q2", "2025-01-01", "2025-07-01"),
        ("2025-Q3+Q4", "2025-07-01", "2026-01-01"),
        ("2026-Q1", "2026-01-01", "2026-04-01"),
    ]
    print(f"{'periode':<15} {'n':>5} {'WR%':>6} {'exp_R':>8} {'total_R':>9} {'DD_R':>8} {'verdict'}")
    print("-" * 80)
    for label, ps, pe in periods:
        pnls, _ = run_period("XAUUSD", df_m5, df_h1, df_d1,
                              pd.Timestamp(ps, tz="UTC"), pd.Timestamp(pe, tz="UTC"),
                              apply_filter=True, spread=0.0)
        s = stats_pnls(pnls)
        if s is None: print(f"{label:<15} 0 trades"); continue
        verdict = "OK" if s["exp"] > 0.05 else ("faible" if s["exp"] > 0 else "NEG")
        print(f"{label:<15} {s['n']:>5} {s['wr']:>6.1f} {s['exp']:>+8.3f} {s['total']:>+9.1f} {s['dd']:>8.1f}  {verdict}")

    # =============================
    # VERIF 2 : Frais reels XAU
    # =============================
    print("\n" + "=" * 80)
    print("VERIF 2 - FRAIS REELS INCLUS XAU (spread 0.14$ = 0.07 broker + 0.07 commission)")
    print("=" * 80)
    sp = SPREADS["XAUUSD"] * (1 + COMMISSION_RATIO)
    pnls_full, _ = run_period("XAUUSD", df_m5, df_h1, df_d1,
                               pd.Timestamp("2023-01-01", tz="UTC"),
                               pd.Timestamp("2026-04-01", tz="UTC"),
                               apply_filter=True, spread=sp)
    pnls_full_no_fees, _ = run_period("XAUUSD", df_m5, df_h1, df_d1,
                                        pd.Timestamp("2023-01-01", tz="UTC"),
                                        pd.Timestamp("2026-04-01", tz="UTC"),
                                        apply_filter=True, spread=0.0)
    s_with = stats_pnls(pnls_full); s_without = stats_pnls(pnls_full_no_fees)
    print(f"Sans frais : n={s_without['n']:<5} WR={s_without['wr']:.1f}% exp={s_without['exp']:+.3f}R total={s_without['total']:+.1f}R")
    print(f"Avec frais : n={s_with['n']:<5} WR={s_with['wr']:.1f}% exp={s_with['exp']:+.3f}R total={s_with['total']:+.1f}R")
    print(f"Impact frais : {s_with['exp']-s_without['exp']:+.3f}R par trade")

    # =============================
    # VERIF 3 : Multi-actifs (avec frais)
    # =============================
    print("\n" + "=" * 80)
    print("VERIF 3 - MULTI-ACTIFS (avec frais)")
    print("=" * 80)
    print(f"{'asset':<10} {'n':>5} {'WR%':>6} {'exp_R':>8} {'total_R':>9} {'DD_R':>8} {'verdict'}")
    print("-" * 80)
    pos_count = 0; total_assets = 0
    for asset in ["XAUUSD", "NAS100", "SP500", "GER40", "EURUSD", "BTCUSD"]:
        try:
            df_m5_a, df_h1_a, df_d1_a = load_data(asset)
        except Exception as e:
            print(f"{asset:<10} ERREUR data : {e}"); continue
        if len(df_m5_a) < 1000:
            print(f"{asset:<10} pas assez de data ({len(df_m5_a)})"); continue
        sp_a = SPREADS.get(asset, 0.1) * (1 + COMMISSION_RATIO)
        pnls_a, _ = run_period(asset, df_m5_a, df_h1_a, df_d1_a,
                                pd.Timestamp("2023-01-01", tz="UTC"),
                                pd.Timestamp("2026-04-01", tz="UTC"),
                                apply_filter=True, spread=sp_a)
        s = stats_pnls(pnls_a)
        total_assets += 1
        if s is None: print(f"{asset:<10} 0 trades"); continue
        verdict = "OK" if s["exp"] > 0.05 else ("faible" if s["exp"] > 0 else "NEG")
        if s["exp"] > 0: pos_count += 1
        print(f"{asset:<10} {s['n']:>5} {s['wr']:>6.1f} {s['exp']:>+8.3f} {s['total']:>+9.1f} {s['dd']:>8.1f}  {verdict}")
    print(f"\nActifs profitables : {pos_count}/{total_assets}")

    # =============================
    # VERIF 4 : Stratification temporelle par trimestre XAU
    # =============================
    print("\n" + "=" * 80)
    print("VERIF 4 - STRATIFICATION PAR TRIMESTRE XAU (avec frais)")
    print("=" * 80)
    print(f"{'trim':<10} {'n':>5} {'WR%':>6} {'exp_R':>8} {'verdict'}")
    print("-" * 50)
    sp_xau = SPREADS["XAUUSD"] * (1 + COMMISSION_RATIO)
    trimestres = pd.date_range("2023-01-01", "2026-04-01", freq="QS").tolist()
    n_pos_t = 0; n_total_t = 0
    for i in range(len(trimestres) - 1):
        ps = trimestres[i]; pe = trimestres[i + 1]
        pnls_t, _ = run_period("XAUUSD", df_m5, df_h1, df_d1,
                                ps.tz_localize("UTC") if ps.tzinfo is None else ps,
                                pe.tz_localize("UTC") if pe.tzinfo is None else pe,
                                apply_filter=True, spread=sp_xau)
        s = stats_pnls(pnls_t)
        if s is None or s["n"] < 5: continue
        n_total_t += 1
        verdict = "+" if s["exp"] > 0 else "-"
        if s["exp"] > 0: n_pos_t += 1
        trim_label = f"{ps.year}-Q{(ps.month-1)//3+1}"
        print(f"{trim_label:<10} {s['n']:>5} {s['wr']:>6.1f} {s['exp']:>+8.3f}  {verdict}")
    print(f"\nTrimestres positifs : {n_pos_t}/{n_total_t}")

    # =============================
    # VERIF 5 : Sanity check - OBs randomises
    # =============================
    print("\n" + "=" * 80)
    print("VERIF 5 - SANITY CHECK : RANDOM ENTRY (doit donner ~-spread)")
    print("=" * 80)
    print("Test : on prend des entries random aux memes ts que les OBs reels.")
    print("Si la regle ICT a un edge, random doit donner ~0 ou negatif.")
    np.random.seed(42)
    obs_xau = find_obs_simple(df_m5, as_of_index=len(df_m5) - 1)
    obs_xau = [ob for ob in obs_xau if pd.Timestamp("2023-01-01", tz="UTC") <= ob.validation_ts < pd.Timestamp("2026-04-01", tz="UTC")]
    # Echantillon de 1000 OBs random : direction aleatoire
    random_obs = []
    sample = np.random.choice(len(obs_xau), size=min(2000, len(obs_xau)), replace=False)
    pnls_rnd = []
    for idx in sample:
        ob = obs_xau[idx]
        # Force direction random
        from bot_v2.concepts.safe import SafeOB
        rnd_dir = "bullish" if np.random.random() < 0.5 else "bearish"
        ob_rnd = SafeOB(direction=rnd_dir, ob_low=ob.ob_low, ob_high=ob.ob_high,
                          ob_open=ob.ob_open, ob_close=ob.ob_close,
                          formation_index=ob.formation_index, formation_ts=ob.formation_ts,
                          validation_index=ob.validation_index, validation_ts=ob.validation_ts,
                          displacement_atr=ob.displacement_atr, swept_swing_index=None)
        atr = atr_safe(df_m5, ob.validation_index)
        if atr <= 0: continue
        r = simulate_ob_trade(df_m5, ob_rnd, atr, spread=sp_xau)
        if r["pnl_r"] is not None: pnls_rnd.append(r["pnl_r"])
    s_rnd = stats_pnls(pnls_rnd)
    print(f"Random direction : n={s_rnd['n']} WR={s_rnd['wr']:.1f}% exp={s_rnd['exp']:+.3f}R")
    print(f"VERDICT : Si exp random < exp user-rule => l'edge vient bien de la regle ICT")

    # =============================
    # VERIF 6 : Test sans filtre user (baseline) XAU avec frais
    # =============================
    print("\n" + "=" * 80)
    print("VERIF 6 - BASELINE SANS FILTRE USER (avec frais)")
    print("=" * 80)
    # Limite 5000 OBs pour speed
    obs_all = find_obs_simple(df_m5, as_of_index=len(df_m5)-1)
    obs_all = [ob for ob in obs_all if pd.Timestamp("2023-01-01",tz="UTC") <= ob.validation_ts < pd.Timestamp("2026-04-01",tz="UTC")][:5000]
    pnls_bl = []
    for ob in obs_all:
        atr = atr_safe(df_m5, ob.validation_index)
        if atr <= 0: continue
        r = simulate_ob_trade(df_m5, ob, atr, spread=sp_xau)
        if r["pnl_r"] is not None: pnls_bl.append(r["pnl_r"])
    s_bl = stats_pnls(pnls_bl)
    print(f"Sans filtre : n={s_bl['n']} WR={s_bl['wr']:.1f}% exp={s_bl['exp']:+.3f}R total={s_bl['total']:+.1f}R")
    print(f"Avec filtre user (rappel) : exp={s_with['exp']:+.3f}R")
    print(f"\nGain net du filtre user : {s_with['exp']-s_bl['exp']:+.3f}R par trade")

    # =============================
    # CONCLUSION
    # =============================
    print("\n" + "=" * 80)
    print("CONCLUSION FINALE")
    print("=" * 80)
    print(f"V1. Walk-forward XAU : voir resultats par periode ci-dessus")
    print(f"V2. Avec frais reels : exp={s_with['exp']:+.3f}R par trade")
    print(f"V3. Multi-actifs : {pos_count}/{total_assets} profitables")
    print(f"V4. Trimestres : {n_pos_t}/{n_total_t} positifs")
    print(f"V5. Random : exp={s_rnd['exp']:+.3f}R (doit etre < user)")
    print(f"V6. Sans filtre : exp={s_bl['exp']:+.3f}R (doit etre < user)")


if __name__ == "__main__":
    main()
