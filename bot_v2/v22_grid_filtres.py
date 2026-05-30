"""v22_grid_filtres.py - Grid search de filtres additionnels pour ameliorer le WR.

On part de la baseline (D1+H1+PD daily) et on teste UN PAR UN des filtres
additionnels pour voir lesquels ameliorent reellement la performance.

Filtres testes :
1. atr_ratio_max : skip si ATR M5 actuel > X * ATR moyen 50 bougies (anti-spike vol)
2. hour_max : skip si heure UTC > X (anti-fin de seance / faible liquidite)
3. hour_min : skip si heure UTC < X (anti-debut Asia / faible liquidite)
4. ob_size_atr_min : skip si zone OB < X ATR (OB trop petit = SL ecrase)
5. ob_size_atr_max : skip si zone OB > X ATR (OB trop grand = RR pourri)
6. dow_skip_monday : skip lundi
7. dow_skip_friday_pm : skip vendredi apres 19h UTC
8. h1_momentum_min : skip si momentum H1 trop faible (= pas vraiment en trend)
9. ob_displacement_min : skip si pas de displacement minimum apres OB
10. dist_to_target_max : skip si trop loin du PDH/PDL cible
11. dist_to_target_min : skip si trop pres du PDH/PDL (deja epuise)

Sortie : tableau qui montre, pour chaque filtre, l'exp_R + WR + n_trades vs baseline.
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
    """D1 + H1 + PD daily aligned."""
    val_ts = ob.validation_ts; val_idx = ob.validation_index
    d1 = daily_bias_safe(df_d1, val_ts)
    if not d1["ok"]: return False
    d1_h = d1["bias"] == "haussier"
    d1_a = (ob.direction == "bullish" and d1_h) or (ob.direction == "bearish" and not d1_h)
    if not d1_a: return False
    h1_close = htf_value_at_t(df_h1, val_ts, "close")
    h1_a = False
    if h1_close is not None:
        pos = df_h1.index.searchsorted(val_ts, side="left") - 1 - 10
        if pos >= 0:
            h1_old = float(df_h1["close"].iloc[pos])
            h1_h = h1_close > h1_old
            h1_a = (ob.direction == "bullish" and h1_h) or (ob.direction == "bearish" and not h1_h)
    if not h1_a: return False
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
            return pd_a
    return False


def compute_meta(df_m5, df_h1, df_d1, ob, atr):
    """Retourne meta-features pour appliquer les filtres."""
    val_ts = ob.validation_ts; val_idx = ob.validation_index
    meta = {}
    # ATR ratio : ATR actuel (sur 14) / ATR moyen 50 dernieres bougies
    atr_50 = atr_safe(df_m5, val_idx, period=50)
    meta["atr_ratio"] = atr / atr_50 if atr_50 > 0 else 1.0
    # Heure UTC
    meta["hour_utc"] = val_ts.hour + val_ts.minute / 60
    # Heure FR
    meta["hour_fr"] = val_ts.tz_convert("Europe/Paris").hour + val_ts.tz_convert("Europe/Paris").minute / 60
    # Day of week (0=lun, 4=ven)
    meta["dow"] = val_ts.weekday()
    # Taille OB en ATR
    ob_size = ob.ob_high - ob.ob_low
    meta["ob_size_atr"] = ob_size / atr if atr > 0 else 0
    # Momentum H1
    h1_close = htf_value_at_t(df_h1, val_ts, "close")
    h1_mom = 0
    if h1_close is not None:
        pos = df_h1.index.searchsorted(val_ts, side="left") - 1 - 10
        if pos >= 0:
            h1_old = float(df_h1["close"].iloc[pos])
            if h1_old > 0:
                h1_mom = (h1_close - h1_old) / h1_old * 100
    meta["h1_momentum_pct"] = h1_mom
    # Dist to target PDH/PDL
    dist_atr = 0
    if df_d1 is not None and len(df_d1) >= 2:
        target_day = pd.Timestamp(val_ts).normalize()
        df_d1_past = df_d1[df_d1.index < target_day]
        if len(df_d1_past) >= 1:
            pdh = float(df_d1_past["high"].iloc[-1])
            pdl = float(df_d1_past["low"].iloc[-1])
            price = float(df_m5["close"].iloc[val_idx])
            if ob.direction == "bullish":
                dist_atr = (pdh - price) / atr if atr > 0 else 0
            else:
                dist_atr = (price - pdl) / atr if atr > 0 else 0
    meta["dist_to_target_atr"] = dist_atr
    # Displacement
    meta["displacement_atr"] = float(ob.displacement_atr)
    return meta


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


def collect_dataset():
    """Construit un dataset {trades x features} sur les 6 actifs."""
    dataset = []   # liste de dicts {asset, meta, pnl}
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
        n_pre = len(dataset)
        for ob in obs:
            if not passes_user_rule(df_m5, df_h1, df_d1, ob): continue
            atr = atr_safe(df_m5, ob.validation_index)
            if atr <= 0: continue
            p = simulate_ob_trade(df_m5, ob, atr, spread=sp)
            if p is None: continue
            meta = compute_meta(df_m5, df_h1, df_d1, ob, atr)
            dataset.append({"asset": user_name, "pnl": p, "meta": meta})
        print(f"{len(dataset) - n_pre} trades")
    print(f"\nTotal dataset : {len(dataset)} trades")
    return dataset


def stats(pnls):
    if not pnls: return None
    pnls = np.array(pnls)
    wins = (pnls > 0).sum(); losses = (pnls < 0).sum()
    n = len(pnls); wr = wins / n * 100; exp = pnls.mean(); total = pnls.sum()
    return {"n": n, "wr": wr, "exp": exp, "total": total}


def apply_filter(dataset, filter_fn):
    """Garde uniquement les trades pour lesquels filter_fn(meta) == True."""
    return [t["pnl"] for t in dataset if filter_fn(t["meta"])]


def main():
    print("=" * 90)
    print("GRID SEARCH FILTRES ADDITIONNELS - 6 actifs 2023-2026")
    print("=" * 90)
    print("\n>>> Loading dataset...")
    dataset = collect_dataset()
    if not dataset: return

    pnls_baseline = [t["pnl"] for t in dataset]
    s_bl = stats(pnls_baseline)
    print(f"\nBASELINE (D1+H1+PD daily) : n={s_bl['n']}  WR={s_bl['wr']:.1f}%  exp={s_bl['exp']:+.3f}R  total={s_bl['total']:+.1f}R")
    print()
    print("=" * 90)
    print("TEST FILTRES (1 par 1)")
    print("=" * 90)
    print(f"{'filtre':<45} {'n':<7} {'-vs_BL':<8} {'WR%':<7} {'+vs_BL':<8} {'exp_R':<9} {'+vs_BL':<8}")
    print("-" * 100)

    # Liste de filtres (nom, fonction)
    filters = []

    # 1. ATR ratio max
    for thr in [1.5, 2.0, 2.5, 3.0]:
        filters.append((f"atr_ratio < {thr}", lambda m, t=thr: m["atr_ratio"] < t))

    # 2. Hour FR window
    for thr in [22, 21, 20]:
        filters.append((f"hour_fr < {thr} (skip fin)", lambda m, t=thr: m["hour_fr"] < t))
    for thr in [3, 5, 7]:
        filters.append((f"hour_fr > {thr} (skip Asia)", lambda m, t=thr: m["hour_fr"] > t))

    # 3. OB size min/max
    for thr in [0.3, 0.5, 0.7]:
        filters.append((f"ob_size_atr > {thr}", lambda m, t=thr: m["ob_size_atr"] > t))
    for thr in [3.0, 4.0]:
        filters.append((f"ob_size_atr < {thr}", lambda m, t=thr: m["ob_size_atr"] < t))

    # 4. Day of week
    filters.append(("dow != monday", lambda m: m["dow"] != 0))
    filters.append(("dow != friday", lambda m: m["dow"] != 4))
    filters.append(("dow in [tue, wed, thu]", lambda m: m["dow"] in (1, 2, 3)))
    filters.append(("dow != friday_pm",
                     lambda m: not (m["dow"] == 4 and m["hour_utc"] > 19)))

    # 5. H1 momentum minimum
    for thr in [0.1, 0.3, 0.5, 1.0]:
        filters.append((f"|h1_momentum_pct| > {thr}",
                         lambda m, t=thr: abs(m["h1_momentum_pct"]) > t))

    # 6. Distance to target
    for thr_min in [2.0, 3.0, 5.0]:
        filters.append((f"dist_to_target_atr > {thr_min}",
                         lambda m, t=thr_min: m["dist_to_target_atr"] > t))
    for thr_max in [20.0, 30.0, 50.0]:
        filters.append((f"dist_to_target_atr < {thr_max}",
                         lambda m, t=thr_max: m["dist_to_target_atr"] < t))

    # 7. Displacement minimum
    for thr in [0.5, 1.0, 1.5]:
        filters.append((f"displacement_atr > {thr}",
                         lambda m, t=thr: m["displacement_atr"] > t))

    # 8. Hour FR : zone NY AM (14h-17h)
    filters.append(("hour_fr in [14, 17] (NY AM)",
                     lambda m: 14 <= m["hour_fr"] < 17))
    filters.append(("hour_fr in [8, 11] (London)",
                     lambda m: 8 <= m["hour_fr"] < 11))
    filters.append(("hour_fr in [8, 17] (London+NY)",
                     lambda m: 8 <= m["hour_fr"] < 17))

    # Run all
    results = []
    for name, fn in filters:
        pnls = apply_filter(dataset, fn)
        if not pnls:
            print(f"{name:<45} 0 trades")
            continue
        s = stats(pnls)
        d_n = s['n'] - s_bl['n']
        d_wr = s['wr'] - s_bl['wr']
        d_exp = s['exp'] - s_bl['exp']
        marker = ""
        if d_exp > 0.02: marker = " *"
        if d_exp > 0.05: marker = " **"
        if d_exp > 0.10: marker = " ***"
        print(f"{name:<45} {s['n']:<7} {d_n:<+8} {s['wr']:<7.1f} {d_wr:<+8.1f} {s['exp']:<+9.3f} {d_exp:<+8.3f}{marker}")
        results.append({"name": name, "stats": s, "d_exp": d_exp, "d_wr": d_wr, "d_n": d_n})

    # TOP par gain d'exp_R
    print("\n" + "=" * 90)
    print("TOP 10 FILTRES (par gain d'esperance R)")
    print("=" * 90)
    results.sort(key=lambda x: -x["d_exp"])
    for r in results[:10]:
        s = r["stats"]
        print(f"  {r['name']:<45} n={s['n']:<7} WR={s['wr']:<5.1f}% exp={s['exp']:+.3f}R  (gain {r['d_exp']:+.3f}R, WR {r['d_wr']:+.1f}%)")

    # ============================
    # COMBINAISONS TOP 3
    # ============================
    print("\n" + "=" * 90)
    print("TEST DES COMBINAISONS (top filtres combines)")
    print("=" * 90)
    # On prend les meilleurs filtres et on essaie des combos
    top_filters_to_combine = [
        ("hour_fr in [8, 17] (London+NY)", lambda m: 8 <= m["hour_fr"] < 17),
        ("dow != monday", lambda m: m["dow"] != 0),
        ("atr_ratio < 2.5", lambda m: m["atr_ratio"] < 2.5),
        ("dist_to_target_atr > 3", lambda m: m["dist_to_target_atr"] > 3),
        ("|h1_momentum_pct| > 0.3", lambda m: abs(m["h1_momentum_pct"]) > 0.3),
        ("ob_size_atr > 0.5", lambda m: m["ob_size_atr"] > 0.5),
    ]

    combos_to_test = [
        ["hour_fr in [8, 17] (London+NY)"],
        ["hour_fr in [8, 17] (London+NY)", "dow != monday"],
        ["hour_fr in [8, 17] (London+NY)", "atr_ratio < 2.5"],
        ["hour_fr in [8, 17] (London+NY)", "dow != monday", "atr_ratio < 2.5"],
        ["hour_fr in [8, 17] (London+NY)", "dist_to_target_atr > 3"],
        ["hour_fr in [8, 17] (London+NY)", "dow != monday", "dist_to_target_atr > 3"],
        ["hour_fr in [8, 17] (London+NY)", "dow != monday", "atr_ratio < 2.5", "dist_to_target_atr > 3"],
        ["dow != monday", "atr_ratio < 2.5", "|h1_momentum_pct| > 0.3"],
        ["hour_fr in [8, 17] (London+NY)", "|h1_momentum_pct| > 0.3"],
    ]
    name_to_fn = {n: f for n, f in top_filters_to_combine}

    print(f"\n{'combo':<70} {'n':<7} {'WR%':<7} {'exp_R':<9} {'gain'}")
    print("-" * 100)
    for combo_names in combos_to_test:
        def combined(m, names=combo_names):
            return all(name_to_fn[n](m) for n in names)
        pnls = apply_filter(dataset, combined)
        if not pnls: print(f"{' + '.join(combo_names)[:65]:<70} 0 trades"); continue
        s = stats(pnls)
        gain = s["exp"] - s_bl["exp"]
        label = " + ".join(combo_names)[:65]
        print(f"{label:<70} {s['n']:<7} {s['wr']:<7.1f} {s['exp']:<+9.3f} {gain:+.3f}R")


if __name__ == "__main__":
    main()
