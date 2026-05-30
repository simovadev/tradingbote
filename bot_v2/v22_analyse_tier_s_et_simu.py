"""v22_analyse_tier_s_et_simu.py - Deux missions :

PARTIE 1 - Tier S : pourquoi 27.4% perdent ?
  - On isole le tier S
  - On compare WIN vs LOSS sur des features supplementaires
  - On cherche le filtre qui ferait passer le tier S de 72.6% a 80-90% WR

PARTIE 2 - Simu 1 mois Option A avec 200 EUR
  - On filtre les trades par Option A (London+NY + H1 momentum > 0.3%)
  - On simule 1 mois (30 derniers jours du dataset) avec sizing minimum 5%
  - On mesure profit / DD / pire serie de pertes
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
    """Score qualite + tier."""
    val_ts = ob.validation_ts; val_idx = ob.validation_index
    score = 0
    hour_fr = val_ts.tz_convert("Europe/Paris").hour + val_ts.tz_convert("Europe/Paris").minute / 60
    if 14 <= hour_fr < 17: score += 3
    elif 8 <= hour_fr < 11: score += 2
    elif 17 <= hour_fr < 21: score += 2
    elif 11 <= hour_fr < 14: score += 1
    elif 5 <= hour_fr < 8: score += 1
    else: score -= 2
    h1_close = htf_value_at_t(df_h1, val_ts, "close")
    h1_mom = 0
    if h1_close is not None:
        pos = df_h1.index.searchsorted(val_ts, side="left") - 1 - 10
        if pos >= 0:
            h1_old = float(df_h1["close"].iloc[pos])
            if h1_old > 0:
                h1_mom = abs((h1_close - h1_old) / h1_old * 100)
                if h1_mom > 1.0: score += 3
                elif h1_mom > 0.5: score += 2
                elif h1_mom > 0.3: score += 1
    disp = abs(ob.displacement_atr)
    if disp > 1.0: score += 2
    elif disp > 0.5: score += 1
    ob_size_atr = (ob.ob_high - ob.ob_low) / atr if atr > 0 else 0
    if 0.5 < ob_size_atr < 3.0: score += 1
    if score >= 7: tier = "S"
    elif score >= 5: tier = "A"
    elif score >= 3: tier = "B"
    elif score >= 1: tier = "C"
    else: tier = "D"
    return score, tier, hour_fr, h1_mom, disp, ob_size_atr


def passes_option_a(meta):
    """Option A : London+NY + H1 momentum > 0.3%."""
    return 8 <= meta["hour_fr"] < 17 and meta["h1_momentum"] > 0.3


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


def collect_dataset_with_meta():
    """Construit un dataset complet avec metadata pour analyse + simu."""
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
            score, tier, hour_fr, h1_mom, disp, ob_size_atr = compute_quality_score(df_m5, df_h1, df_d1, ob, atr)
            p = simulate_ob_trade(df_m5, ob, atr, spread=sp)
            if p is None: continue
            # Heure UTC + day of week pour features
            val_ts = ob.validation_ts
            dow = val_ts.weekday()
            dataset.append({
                "asset": user_name, "pnl_r": p, "score": score, "tier": tier,
                "ts": val_ts, "direction": ob.direction,
                "hour_fr": hour_fr, "h1_momentum": h1_mom,
                "displacement_atr": disp, "ob_size_atr": ob_size_atr,
                "dow": dow,
            })
        print(f"{len(dataset) - n_pre} trades")
    print(f"\nTotal : {len(dataset)} trades")
    return dataset


# ====================================
# PARTIE 1 : Analyse tier S WIN vs LOSS
# ====================================

def analyse_tier_s(dataset):
    print("\n" + "=" * 90)
    print("PARTIE 1 - ANALYSE TIER S : qu'est-ce qui differencie WIN vs LOSS ?")
    print("=" * 90)

    tier_s = [t for t in dataset if t["tier"] == "S"]
    n_s = len(tier_s)
    wins_s = [t for t in tier_s if t["pnl_r"] > 0]
    losses_s = [t for t in tier_s if t["pnl_r"] < 0]
    print(f"\nTier S total : {n_s} trades")
    print(f"  WIN : {len(wins_s)} ({len(wins_s)/n_s*100:.1f}%)")
    print(f"  LOSS : {len(losses_s)} ({len(losses_s)/n_s*100:.1f}%)")

    # ====================================
    # Comparer features sur WIN vs LOSS
    # ====================================
    print(f"\n{'feature':<25} {'WIN moy':<12} {'LOSS moy':<12} {'diff':<10} {'discrim'}")
    print("-" * 70)
    features_num = ["score", "hour_fr", "h1_momentum", "displacement_atr", "ob_size_atr"]
    discrim_scores = []
    for f in features_num:
        w_vals = [t[f] for t in wins_s]
        l_vals = [t[f] for t in losses_s]
        w_mean = np.mean(w_vals); l_mean = np.mean(l_vals)
        # discriminant : abs(diff) / std combine
        all_vals = [t[f] for t in tier_s]
        std = np.std(all_vals)
        discrim = abs(w_mean - l_mean) / std if std > 1e-9 else 0
        discrim_scores.append((f, discrim, w_mean, l_mean))
        print(f"{f:<25} {w_mean:<12.3f} {l_mean:<12.3f} {w_mean - l_mean:<+10.3f} {discrim:.3f}")

    # Compare par actif
    print(f"\nWR tier S par actif :")
    print(f"{'actif':<10} {'n':<5} {'WR':<7} {'exp_R'}")
    for asset in ASSETS_MAP.keys():
        asset_trades = [t for t in tier_s if t["asset"] == asset]
        if not asset_trades: continue
        n = len(asset_trades)
        wins = sum(1 for t in asset_trades if t["pnl_r"] > 0)
        wr = wins / n * 100
        exp = np.mean([t["pnl_r"] for t in asset_trades])
        print(f"  {asset:<10} {n:<5} {wr:<7.1f} {exp:+.3f}R")

    # Direction
    print(f"\nWR tier S par direction :")
    for d in ["bullish", "bearish"]:
        sub = [t for t in tier_s if t["direction"] == d]
        if not sub: continue
        wr = sum(1 for t in sub if t["pnl_r"] > 0) / len(sub) * 100
        exp = np.mean([t["pnl_r"] for t in sub])
        print(f"  {d:<10} n={len(sub):<5} WR={wr:.1f}% exp={exp:+.3f}R")

    # Par heure FR
    print(f"\nWR tier S par fenetre horaire (FR) :")
    hour_buckets = [(8, 11, "London"), (11, 14, "Pre-NY"), (14, 17, "NY AM"), (17, 21, "NY PM")]
    for h0, h1, label in hour_buckets:
        sub = [t for t in tier_s if h0 <= t["hour_fr"] < h1]
        if not sub: continue
        wr = sum(1 for t in sub if t["pnl_r"] > 0) / len(sub) * 100
        exp = np.mean([t["pnl_r"] for t in sub])
        print(f"  {label:<10} ({h0:02d}h-{h1:02d}h) n={len(sub):<5} WR={wr:.1f}% exp={exp:+.3f}R")

    # Combinaisons proposees pour faire monter le WR tier S
    print(f"\n=== TENTATIVES POUR REMONTER LE WR TIER S ===\n")
    combos = [
        ("S baseline", lambda t: True),
        ("S + hour 14h-17h (NY AM)", lambda t: 14 <= t["hour_fr"] < 17),
        ("S + h1_momentum > 0.5", lambda t: t["h1_momentum"] > 0.5),
        ("S + h1_momentum > 1.0", lambda t: t["h1_momentum"] > 1.0),
        ("S + displacement > 1.0", lambda t: t["displacement_atr"] > 1.0),
        ("S + score >= 9", lambda t: t["score"] >= 9),
        ("S + score >= 10", lambda t: t["score"] >= 10),
        ("S + NY AM + h1_mom > 0.5", lambda t: 14 <= t["hour_fr"] < 17 and t["h1_momentum"] > 0.5),
        ("S + NY AM + h1_mom > 1.0", lambda t: 14 <= t["hour_fr"] < 17 and t["h1_momentum"] > 1.0),
        ("S + h1_mom > 0.5 + disp > 1.0", lambda t: t["h1_momentum"] > 0.5 and t["displacement_atr"] > 1.0),
    ]
    print(f"{'config':<45} {'n':<6} {'WR':<7} {'exp_R':<8} {'vs S'}")
    print("-" * 80)
    base_wr = sum(1 for t in tier_s if t["pnl_r"] > 0) / n_s * 100
    base_exp = np.mean([t["pnl_r"] for t in tier_s])
    for name, fn in combos:
        sub = [t for t in tier_s if fn(t)]
        if not sub:
            print(f"{name:<45} 0 trades"); continue
        wr = sum(1 for t in sub if t["pnl_r"] > 0) / len(sub) * 100
        exp = np.mean([t["pnl_r"] for t in sub])
        d_wr = wr - base_wr
        d_exp = exp - base_exp
        marker = " ***" if d_exp > 0.1 else (" *" if d_exp > 0.05 else "")
        print(f"{name:<45} {len(sub):<6} {wr:<7.1f} {exp:<+8.3f} WR{d_wr:+.1f}%  exp{d_exp:+.3f}R{marker}")


# ====================================
# PARTIE 2 : Simulation 1 mois 200 EUR Option A
# ====================================

def simulate_account(dataset_filtered, sizing_fn, initial=200.0):
    """Simu compte. sizing_fn(tier) -> pct_risk."""
    cap = initial; cap_history = [initial]
    peak = initial; max_dd_pct = 0.0
    cur_streak_loss = 0; max_streak_loss = 0
    for t in dataset_filtered:
        pct_risk = sizing_fn(t["tier"])
        if pct_risk <= 0:
            cap_history.append(cap); continue
        risk_eur = cap * pct_risk / 100
        cap += risk_eur * t["pnl_r"]
        if cap <= 0:
            cap_history.append(0); break
        cap_history.append(cap)
        if cap > peak: peak = cap
        dd_pct = (peak - cap) / peak * 100 if peak > 0 else 0
        if dd_pct > max_dd_pct: max_dd_pct = dd_pct
        if t["pnl_r"] < 0:
            cur_streak_loss += 1
            max_streak_loss = max(max_streak_loss, cur_streak_loss)
        else:
            cur_streak_loss = 0
    return cap_history, max_dd_pct, max_streak_loss


def simu_1_mois(dataset):
    print("\n" + "=" * 90)
    print("PARTIE 2 - SIMULATION 1 MOIS Option A + 200 EUR + sizing min 5%")
    print("=" * 90)

    # Filtre Option A
    dataset_opt_a = [t for t in dataset if passes_option_a(t)]
    print(f"\nDataset Option A : {len(dataset_opt_a)} trades / {len(dataset)} baseline")
    avg_per_day = len(dataset_opt_a) / ((dataset[-1]["ts"] - dataset[0]["ts"]).days)
    print(f"  ~{avg_per_day:.1f} trades / jour")

    # Prendre les 1 mois (30 derniers jours du dataset)
    last_ts = dataset[-1]["ts"]
    month_ago = last_ts - pd.Timedelta(days=30)
    dataset_1m = [t for t in dataset_opt_a if t["ts"] >= month_ago]
    print(f"\nTrades dernier mois Option A : {len(dataset_1m)}")
    n_tier = {}
    for t in dataset_1m:
        n_tier[t["tier"]] = n_tier.get(t["tier"], 0) + 1
    print(f"  Repartition : {n_tier}")

    # Sizings a tester
    sizings = [
        ("Risque fixe 5%", lambda tier: 5.0),
        ("Risque fixe 7%", lambda tier: 7.0),
        ("Risque fixe 10%", lambda tier: 10.0),
        ("DYN min 5% (S=10 A=7 B=5 C=5 D=skip)",
         lambda tier: {"S": 10, "A": 7, "B": 5, "C": 5, "D": 0}.get(tier, 0)),
        ("DYN min 5% (S=15 A=10 B=5 C=5 D=skip)",
         lambda tier: {"S": 15, "A": 10, "B": 5, "C": 5, "D": 0}.get(tier, 0)),
    ]

    print(f"\n{'sizing':<55} {'fin EUR':<12} {'%':<10} {'DD%':<8} {'max_loss_streak'}")
    print("-" * 100)
    for name, fn in sizings:
        cap_hist, dd, streak = simulate_account(dataset_1m, fn, initial=200.0)
        end_cap = cap_hist[-1]
        pct = (end_cap - 200) / 200 * 100 if end_cap > 0 else -100
        end_str = f"{end_cap:.2f}" if end_cap > 0 else "RUINE"
        pct_str = f"{pct:+.1f}%" if end_cap > 0 else "-100%"
        print(f"{name:<55} {end_str:<12} {pct_str:<10} {dd:<8.1f} {streak}")

    # Simu trajectoire
    print(f"\n=== TRAJECTOIRE DETAILLEE - DYN min 5% (S=10 A=7 B=5 C=5 D=skip) ===")
    sizing = lambda tier: {"S": 10, "A": 7, "B": 5, "C": 5, "D": 0}.get(tier, 0)
    cap_hist, dd, streak = simulate_account(dataset_1m, sizing, initial=200.0)
    print(f"  Demarrage : 200 EUR")
    print(f"  Fin de mois : {cap_hist[-1]:.2f} EUR ({(cap_hist[-1]-200)/200*100:+.1f}%)")
    print(f"  DD max : -{dd:.1f}%")
    print(f"  Pire serie pertes consecutives : {streak}")
    # Quelques milestones
    milestones = [0.25, 0.5, 0.75, 1.0]
    for m in milestones:
        idx = int(len(cap_hist) * m)
        if idx < len(cap_hist):
            print(f"  Trade #{idx}/{len(cap_hist)-1} : {cap_hist[idx]:.2f} EUR")


def main():
    print(">>> Loading dataset...")
    dataset = collect_dataset_with_meta()
    analyse_tier_s(dataset)
    simu_1_mois(dataset)


if __name__ == "__main__":
    main()
