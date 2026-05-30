"""v22_position_sizing.py - Scoring qualite trade + position sizing dynamique.

Concept : on classe chaque trade en tier S/A/B/C selon plusieurs criteres
de qualite, et on adapte le risque par trade en fonction du tier.

Criteres de scoring :
+3 si NY AM (14h-17h FR)
+2 si London (8h-11h FR)
+1 si hors mais dans Asia tardive (5h-7h)
-2 si hors fenetres (nuit, week-end)

+3 si H1 momentum > 1.0%
+2 si H1 momentum > 0.5%
+1 si H1 momentum > 0.3%

+2 si displacement_atr > 1.0
+1 si displacement_atr > 0.5

+1 si ob_size_atr > 0.5 et < 3.0 (taille raisonnable)

Tiers :
  S : score >= 7
  A : score 5-6
  B : score 3-4
  C : score 1-2
  D : score <= 0

Puis on simule 1 mois de bot avec position sizing :
  - Strategie 1 : risque FIXE 1% sur tous tiers
  - Strategie 2 : risque FIXE 5% sur tous tiers
  - Strategie 3 : risque FIXE 10% sur tous tiers
  - Strategie 4 : sizing DYNAMIQUE conservateur (S=5%, A=3%, B=1%, C=0.5%, D=skip)
  - Strategie 5 : sizing DYNAMIQUE agressif (S=10%, A=7%, B=3%, C=1%, D=skip)
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


def passes_user_rule(df_m5, df_h1, df_d1, ob):
    val_ts = ob.validation_ts; val_idx = ob.validation_index
    d1 = daily_bias_safe(df_d1, val_ts)
    if not d1["ok"]: return False
    d1_h = d1["bias"] == "haussier"
    d1_a = (ob.direction == "bullish" and d1_h) or (ob.direction == "bearish" and not d1_h)
    if not d1_a: return False
    h1_close = htf_value_at_t(df_h1, val_ts, "close")
    if h1_close is None: return False
    pos = df_h1.index.searchsorted(val_ts, side="left") - 1 - 10
    if pos < 0: return False
    h1_old = float(df_h1["close"].iloc[pos])
    h1_h = h1_close > h1_old
    h1_a = (ob.direction == "bullish" and h1_h) or (ob.direction == "bearish" and not h1_h)
    if not h1_a: return False
    if df_d1 is None or len(df_d1) < 2: return False
    target_day = pd.Timestamp(val_ts).normalize()
    df_d1_past = df_d1[df_d1.index < target_day]
    if len(df_d1_past) < 1: return False
    pdh = float(df_d1_past["high"].iloc[-1])
    pdl = float(df_d1_past["low"].iloc[-1])
    mid = (pdh + pdl) / 2
    price = float(df_m5["close"].iloc[val_idx])
    pd_a = (
        (ob.direction == "bullish" and price < mid) or
        (ob.direction == "bearish" and price > mid)
    )
    return pd_a


def compute_quality_score(df_m5, df_h1, df_d1, ob, atr):
    """Retourne (score, tier) selon les criteres de qualite."""
    val_ts = ob.validation_ts; val_idx = ob.validation_index
    score = 0

    # ---- TIMING (heures FR) ----
    hour_fr = val_ts.tz_convert("Europe/Paris").hour + val_ts.tz_convert("Europe/Paris").minute / 60
    if 14 <= hour_fr < 17:
        score += 3   # NY AM (best window)
    elif 8 <= hour_fr < 11:
        score += 2   # London
    elif 17 <= hour_fr < 21:
        score += 2   # NY PM
    elif 11 <= hour_fr < 14:
        score += 1   # Pre-NY
    elif 5 <= hour_fr < 8:
        score += 1   # Asia tardive
    else:
        score -= 2   # nuit (faible liquidite)

    # ---- H1 MOMENTUM ----
    h1_close = htf_value_at_t(df_h1, val_ts, "close")
    if h1_close is not None:
        pos = df_h1.index.searchsorted(val_ts, side="left") - 1 - 10
        if pos >= 0:
            h1_old = float(df_h1["close"].iloc[pos])
            if h1_old > 0:
                h1_mom = abs((h1_close - h1_old) / h1_old * 100)
                if h1_mom > 1.0:
                    score += 3
                elif h1_mom > 0.5:
                    score += 2
                elif h1_mom > 0.3:
                    score += 1

    # ---- DISPLACEMENT ----
    disp = abs(ob.displacement_atr)
    if disp > 1.0:
        score += 2
    elif disp > 0.5:
        score += 1

    # ---- OB SIZE ----
    ob_size_atr = (ob.ob_high - ob.ob_low) / atr if atr > 0 else 0
    if 0.5 < ob_size_atr < 3.0:
        score += 1

    # ---- TIER ----
    if score >= 7: tier = "S"
    elif score >= 5: tier = "A"
    elif score >= 3: tier = "B"
    elif score >= 1: tier = "C"
    else: tier = "D"

    return score, tier


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


def collect_dataset_with_scores():
    """Construit un dataset avec scoring qualite."""
    dataset = []
    for user_name, file_name in ASSETS_MAP.items():
        sp = SPREADS[user_name]
        print(f"  Loading {user_name}...", end=" ", flush=True)
        try:
            df_m5 = pd.read_parquet(DATA / f"{file_name}_M5.parquet")[["open","high","low","close"]]
            df_h1 = pd.read_parquet(DATA / f"{file_name}_H1.parquet")[["open","high","low","close"]]
            df_d1 = pd.read_parquet(DATA / f"{file_name}_D1.parquet")[["open","high","low","close"]]
        except Exception as e:
            print(f"err : {e}"); continue
        start = pd.Timestamp("2023-01-01", tz="UTC")
        end = pd.Timestamp("2026-04-01", tz="UTC")
        ctx = start - pd.Timedelta(days=10)
        df_m5 = df_m5[(df_m5.index >= ctx) & (df_m5.index < end)]
        df_h1 = df_h1[(df_h1.index >= ctx) & (df_h1.index < end)]
        df_d1 = df_d1[(df_d1.index >= start - pd.Timedelta(days=30)) & (df_d1.index < end)]
        obs = find_obs_simple(df_m5, as_of_index=len(df_m5) - 1)
        obs = [ob for ob in obs if start <= ob.validation_ts < end]
        obs.sort(key=lambda x: x.validation_ts)
        n_pre = len(dataset)
        for ob in obs:
            if not passes_user_rule(df_m5, df_h1, df_d1, ob): continue
            atr = atr_safe(df_m5, ob.validation_index)
            if atr <= 0: continue
            score, tier = compute_quality_score(df_m5, df_h1, df_d1, ob, atr)
            p = simulate_ob_trade(df_m5, ob, atr, spread=sp)
            if p is None: continue
            dataset.append({
                "asset": user_name, "pnl_r": p, "score": score, "tier": tier,
                "ts": ob.validation_ts,
            })
        print(f"{len(dataset) - n_pre} trades")
    print(f"\nTotal dataset : {len(dataset)} trades")
    return dataset


def stats_by_tier(dataset):
    """Stats par tier."""
    from collections import defaultdict
    tiers = defaultdict(list)
    for t in dataset:
        tiers[t["tier"]].append(t["pnl_r"])
    out = {}
    for tier, pnls in tiers.items():
        pnls = np.array(pnls)
        wins = (pnls > 0).sum()
        n = len(pnls)
        out[tier] = {
            "n": n, "wr": wins / n * 100 if n else 0,
            "exp": pnls.mean() if n else 0, "total": pnls.sum(),
        }
    return out


def simulate_account(dataset, sizing_fn, initial=200.0, max_pct_total=30.0):
    """Simule un compte avec sizing_fn(tier, pnl_r) -> pct_risk.
    Retourne historique du compte.
    """
    cap = initial
    cap_history = [initial]
    peak = initial
    max_dd_pct = 0.0
    for t in dataset:
        pct_risk = sizing_fn(t["tier"])
        if pct_risk <= 0:
            cap_history.append(cap); continue
        risk_eur = cap * pct_risk / 100
        cap += risk_eur * t["pnl_r"]
        if cap <= 0:
            cap_history.extend([0] * (len(dataset) - len(cap_history) + 1))
            break
        cap_history.append(cap)
        if cap > peak: peak = cap
        dd_pct = (peak - cap) / peak * 100 if peak > 0 else 0
        if dd_pct > max_dd_pct: max_dd_pct = dd_pct
    return cap_history, max_dd_pct


def main():
    print("=" * 90)
    print("POSITION SIZING DYNAMIQUE - Scoring qualite par tier")
    print("=" * 90)
    print(">>> Loading dataset...")
    dataset = collect_dataset_with_scores()

    # === STATS PAR TIER ===
    tiers_stats = stats_by_tier(dataset)
    print(f"\n=== PERFORMANCE PAR TIER ===\n")
    print(f"{'tier':<6} {'n':<7} {'WR%':<7} {'exp_R':<9} {'total_R':<10} {'% du dataset'}")
    print("-" * 55)
    for tier in ["S", "A", "B", "C", "D"]:
        if tier not in tiers_stats: print(f"{tier:<6} (vide)"); continue
        s = tiers_stats[tier]
        pct = s["n"] / len(dataset) * 100
        print(f"{tier:<6} {s['n']:<7} {s['wr']:<7.1f} {s['exp']:<+9.3f} {s['total']:<+10.1f} {pct:.1f}%")

    # === SIMULATION SUR 1 MOIS RECENT (180 derniers trades) ===
    print(f"\n=== SIMULATION 1 MOIS (180 derniers trades) ===")
    print(f"Initial : 200 EUR\n")
    dataset_recent = dataset[-180:] if len(dataset) >= 180 else dataset

    sizings = [
        ("Risque fixe 0.5%", lambda tier: 0.5),
        ("Risque fixe 1%", lambda tier: 1.0),
        ("Risque fixe 2%", lambda tier: 2.0),
        ("Risque fixe 5%", lambda tier: 5.0),
        ("Risque fixe 10%", lambda tier: 10.0),
        ("DYN conservateur (S=5 A=3 B=1 C=0.5 D=skip)",
         lambda tier: {"S": 5, "A": 3, "B": 1, "C": 0.5, "D": 0}.get(tier, 0)),
        ("DYN modere (S=7 A=5 B=2 C=1 D=skip)",
         lambda tier: {"S": 7, "A": 5, "B": 2, "C": 1, "D": 0}.get(tier, 0)),
        ("DYN agressif (S=10 A=7 B=3 C=1 D=skip)",
         lambda tier: {"S": 10, "A": 7, "B": 3, "C": 1, "D": 0}.get(tier, 0)),
        ("DYN ultra-selectif (S=15 A=5 B=skip C=skip D=skip)",
         lambda tier: {"S": 15, "A": 5, "B": 0, "C": 0, "D": 0}.get(tier, 0)),
    ]

    print(f"{'sizing':<55} {'fin EUR':<12} {'%':<10} {'DD max'}")
    print("-" * 90)
    for name, fn in sizings:
        cap_hist, dd = simulate_account(dataset_recent, fn, initial=200.0)
        end_cap = cap_hist[-1]
        pct_change = (end_cap - 200) / 200 * 100
        print(f"{name:<55} {end_cap:<12.2f} {pct_change:+<10.1f} {dd:.1f}%")

    # === SUR 12 MOIS COMPLETS (les 2160 derniers trades) ===
    print(f"\n=== SIMULATION 12 MOIS (2160 derniers trades) ===")
    print(f"Initial : 200 EUR\n")
    dataset_12m = dataset[-2160:] if len(dataset) >= 2160 else dataset

    print(f"{'sizing':<55} {'fin EUR':<14} {'%':<14} {'DD max'}")
    print("-" * 95)
    for name, fn in sizings:
        cap_hist, dd = simulate_account(dataset_12m, fn, initial=200.0)
        end_cap = cap_hist[-1]
        pct_change = (end_cap - 200) / 200 * 100
        end_str = f"{end_cap:,.2f}" if end_cap > 0 else "RUINE"
        pct_str = f"{pct_change:+.1f}%" if end_cap > 0 else "-100%"
        print(f"{name:<55} {end_str:<14} {pct_str:<14} {dd:.1f}%")


if __name__ == "__main__":
    main()
