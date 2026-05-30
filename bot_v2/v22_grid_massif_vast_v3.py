"""v22_grid_massif_vast_v3.py - Grid massif sur 28 actifs + contrainte 1+ trade/jour.

Differences vs v2 :
- 28 actifs (au lieu de 6) -> vrai test anti-overfit
- Contrainte n_trades >= 1000 (= ~1/jour sur 3 ans) au lieu de >= 100
- Stockage spread par actif via dict (avec defaut)
- Logs detail par actif a Phase 1

Resultat attendu : seules les combos qui generent du volume sur la majorite
des 28 actifs survivront. Si une combo ne marche que sur 1-2 actifs avec
+0.7R sur 200 trades, elle sera EJECTEE (pas assez de trades).
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

from bot_v2.concepts.safe import (
    find_obs_simple, daily_bias_safe, htf_value_at_t, atr_safe,
)

DATA = ROOT / "data_vantage"

# 28 actifs : tous ceux qu'on a en local
ASSETS = [
    "AUDUSD", "BTCUSD", "BVSPX", "CL-OIL", "Cocoa-C", "Coffee-C", "DJ30",
    "ETHUSD", "EURUSD", "FRA40", "GAS-C", "GBPUSD", "GER40", "HK50",
    "NAS100", "NZDUSD", "Nikkei225", "SP500", "Sugar-C", "UK100",
    "USDCAD", "USDCHF", "USDJPY", "USDMXN", "USDZAR", "Wheat-C",
    "XAGUSD", "XAUUSD",
]

# Spreads + commissions (en unite prix). Estimes par actif.
SPREADS = {
    "AUDUSD": 0.00010, "BTCUSD": 6.0, "BVSPX": 8.0, "CL-OIL": 0.06,
    "Cocoa-C": 5.0, "Coffee-C": 0.4, "DJ30": 2.0, "ETHUSD": 1.5,
    "EURUSD": 0.00016, "FRA40": 1.5, "GAS-C": 0.005, "GBPUSD": 0.0002,
    "GER40": 0.8, "HK50": 5.0, "NAS100": 1.0, "NZDUSD": 0.00018,
    "Nikkei225": 10.0, "SP500": 0.3, "Sugar-C": 0.04, "UK100": 1.0,
    "USDCAD": 0.0002, "USDCHF": 0.0002, "USDJPY": 0.025,
    "USDMXN": 0.002, "USDZAR": 0.004, "Wheat-C": 0.5,
    "XAGUSD": 0.025, "XAUUSD": 0.14,
}

TP_RR = 2.0; PARTIAL_R = 1.0
MAX_FILL_BARS = 20; MAX_HOLD_BARS = 50
SL_BUFS = [0.0, 0.05, 0.1, 0.15, 0.2]
MIN_CONS_LIST = [2, 3]

# CONTRAINTE : minimum trades pour qu'une combo soit consideree.
# 28 actifs x ~250 jours x 3 ans = 21000 jours-actif. 1 trade/jour = trop strict.
# On veut ~1 trade/jour MOYEN -> ~1000 trades sur les 3 ans tous actifs confondus.
MIN_TRADES = 1000

# Et au moins X actifs profitables
MIN_PROFITABLE_ASSETS = 14   # > 50% des 28 actifs doivent etre profitables


def passes_user_rule_d1_h1_pd(df_m5, df_h1, df_d1, ob):
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
    pd_a = (ob.direction == "bullish" and price < mid) or (ob.direction == "bearish" and price > mid)
    return pd_a


def simulate_ob_trade(df_m5, ob, atr, spread, sl_buffer_atr):
    val_idx = ob.validation_index
    if val_idx + 1 >= len(df_m5): return None
    hs = spread / 2
    if ob.direction == "bullish":
        sl = ob.ob_low - sl_buffer_atr * atr; entry = ob.ob_high
        entry_eff = entry + hs; sl_eff = sl - hs
    else:
        sl = ob.ob_high + sl_buffer_atr * atr; entry = ob.ob_low
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


def build_dataset_for_asset(args):
    asset_name, sp, asset_idx = args
    try:
        df_m5 = pd.read_parquet(DATA / f"{asset_name}_M5.parquet")[["open","high","low","close"]]
        df_h1 = pd.read_parquet(DATA / f"{asset_name}_H1.parquet")[["open","high","low","close"]]
        df_d1 = pd.read_parquet(DATA / f"{asset_name}_D1.parquet")[["open","high","low","close"]]
    except Exception as e:
        return asset_name, []
    start = pd.Timestamp("2023-01-01", tz="UTC"); end = pd.Timestamp("2026-04-01", tz="UTC")
    ctx = start - pd.Timedelta(days=10)
    df_m5 = df_m5[(df_m5.index >= ctx) & (df_m5.index < end)]
    df_h1 = df_h1[(df_h1.index >= ctx) & (df_h1.index < end)]
    df_d1 = df_d1[(df_d1.index >= start - pd.Timedelta(days=30)) & (df_d1.index < end)]
    if len(df_m5) < 1000: return asset_name, []

    rows = []
    for min_cons in MIN_CONS_LIST:
        obs = find_obs_simple(df_m5, as_of_index=len(df_m5) - 1, min_consecutive=min_cons)
        obs = [ob for ob in obs if start <= ob.validation_ts < end]
        for ob in obs:
            if not passes_user_rule_d1_h1_pd(df_m5, df_h1, df_d1, ob): continue
            atr = atr_safe(df_m5, ob.validation_index)
            if atr <= 0: continue
            atr_50 = atr_safe(df_m5, ob.validation_index, period=50)
            atr_ratio = atr / atr_50 if atr_50 > 0 else 1.0
            val_ts = ob.validation_ts
            hour_fr = val_ts.tz_convert("Europe/Paris").hour + val_ts.tz_convert("Europe/Paris").minute / 60
            h1_close = htf_value_at_t(df_h1, val_ts, "close")
            h1_mom = 0.0
            if h1_close is not None:
                pos = df_h1.index.searchsorted(val_ts, side="left") - 1 - 10
                if pos >= 0:
                    h1_old = float(df_h1["close"].iloc[pos])
                    if h1_old > 0:
                        h1_mom = abs((h1_close - h1_old) / h1_old * 100)
            disp = abs(ob.displacement_atr)
            pnls = []
            for sl_buf in SL_BUFS:
                p = simulate_ob_trade(df_m5, ob, atr, sp, sl_buffer_atr=sl_buf)
                pnls.append(p if p is not None else np.nan)
            if all(np.isnan(p) for p in pnls): continue
            row = [asset_idx, min_cons, atr_ratio, hour_fr, h1_mom, disp] + pnls
            rows.append(row)
    return asset_name, rows


# Globals pour le pool (initialises apres fork)
_G_arr = None
_G_asset_idx = None
_G_min_cons = None
_G_atr_ratio = None
_G_hour_fr = None
_G_h1_mom = None
_G_disp = None
_G_pnls = None
_N_ASSETS = None


def init_worker(arr, n_assets):
    global _G_arr, _G_asset_idx, _G_min_cons, _G_atr_ratio, _G_hour_fr, _G_h1_mom, _G_disp, _G_pnls, _N_ASSETS
    _G_arr = arr
    _G_asset_idx = arr[:, 0].astype(np.int16)
    _G_min_cons = arr[:, 1].astype(np.int8)
    _G_atr_ratio = arr[:, 2]
    _G_hour_fr = arr[:, 3]
    _G_h1_mom = arr[:, 4]
    _G_disp = arr[:, 5]
    _G_pnls = arr[:, 6:]
    _N_ASSETS = n_assets


def evaluate_combo_fast(combo_packed):
    h1m, atr_max, h_min, h_max, dm, sl_idx, mc = combo_packed
    mask = (
        (_G_min_cons == mc) &
        (_G_h1_mom >= h1m) &
        (_G_hour_fr >= h_min) &
        (_G_hour_fr < h_max) &
        (_G_disp >= dm)
    )
    if atr_max is not None:
        mask &= (_G_atr_ratio < atr_max)
    if not mask.any(): return None
    sub_pnls = _G_pnls[mask, sl_idx]
    sub_assets = _G_asset_idx[mask]
    valid = ~np.isnan(sub_pnls)
    sub_pnls = sub_pnls[valid]
    sub_assets = sub_assets[valid]
    n = len(sub_pnls)
    if n < MIN_TRADES: return None
    wins = int((sub_pnls > 0).sum())
    wr = wins / n * 100
    exp = float(sub_pnls.mean())
    total = float(sub_pnls.sum())
    eq = np.cumsum(sub_pnls); peak = np.maximum.accumulate(eq); dd = float((peak - eq).max())
    # Combien d'actifs avec exp_R positif (au moins 30 trades par actif)
    n_profitable = 0
    for ai in range(_N_ASSETS):
        a_mask = sub_assets == ai
        a_n = a_mask.sum()
        if a_n >= 30:
            a_exp = sub_pnls[a_mask].mean()
            if a_exp > 0: n_profitable += 1
    if n_profitable < MIN_PROFITABLE_ASSETS: return None
    return (combo_packed, n, wr, exp, total, dd, n_profitable)


def main():
    print("=" * 90)
    print(f"V22 GRID MASSIF V3 - 28 ACTIFS - CONTRAINTE n>={MIN_TRADES} & profitable>={MIN_PROFITABLE_ASSETS}/28")
    print("=" * 90)

    print(f"\n>>> Phase 1 : pre-calcule dataset OBs + pnls sur {len(ASSETS)} actifs...")
    sys.stdout.flush()
    t0 = time.time()

    args_assets = [(name, SPREADS.get(name, 0.001), idx) for idx, name in enumerate(ASSETS)]
    all_rows = []
    actifs_n = {}
    with ProcessPoolExecutor(max_workers=28) as ex:
        futures = {ex.submit(build_dataset_for_asset, a): a for a in args_assets}
        n_done = 0
        for f in as_completed(futures):
            name, rows = f.result()
            all_rows.extend(rows)
            actifs_n[name] = len(rows)
            n_done += 1
            elapsed = time.time() - t0
            print(f"  [{n_done:>2}/{len(ASSETS)}] +{elapsed:5.0f}s  {name:<10} : {len(rows):>5} OBs (total {len(all_rows)})")
            sys.stdout.flush()

    t1 = time.time()
    print(f"\nDataset total : {len(all_rows)} entries en {t1-t0:.0f}s")
    # Resume actifs vides
    empty = [a for a, n in actifs_n.items() if n == 0]
    if empty:
        print(f"  Actifs vides (skip) : {empty}")
    sys.stdout.flush()

    arr = np.array(all_rows, dtype=np.float32)
    print(f"  Dataset NumPy : {arr.shape}, {arr.nbytes / 1024 / 1024:.1f} MB")
    sys.stdout.flush()

    print("\n>>> Phase 2 : grid search (vectorise + fork)...")
    h1_mom_mins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
    atr_ratios_max = [None, 1.5, 2.0, 2.5, 3.0]
    hours_min = [0, 5, 6, 7, 8, 9]
    hours_max = [17, 19, 20, 21, 22, 24]
    disp_mins = [0.0, 0.3, 0.5, 0.7, 1.0]
    sl_idx_list = list(range(len(SL_BUFS)))
    min_cons_list = MIN_CONS_LIST

    combos = []
    for h1m, atr_m, h_min, h_max, dm, sl_idx, mc in product(
        h1_mom_mins, atr_ratios_max, hours_min, hours_max,
        disp_mins, sl_idx_list, min_cons_list
    ):
        if h_min >= h_max: continue
        combos.append((h1m, atr_m, h_min, h_max, dm, sl_idx, mc))
    n_combos = len(combos)
    print(f"  {n_combos} combos a tester")
    sys.stdout.flush()

    results = []
    t2 = time.time()
    n_done = 0
    n_kept = 0
    LOG_EVERY = max(1, n_combos // 20)
    last_log = time.time()

    with ProcessPoolExecutor(
        max_workers=200, initializer=init_worker,
        initargs=(arr, len(ASSETS))
    ) as ex:
        print(f"  Submitting {n_combos} jobs...")
        sys.stdout.flush()
        futures = [ex.submit(evaluate_combo_fast, c) for c in combos]
        print(f"  Attente resultats...")
        sys.stdout.flush()

        for f in as_completed(futures):
            n_done += 1
            res = f.result()
            if res is not None:
                results.append(res)
                n_kept += 1

            now = time.time()
            if n_done % LOG_EVERY == 0 or (now - last_log) > 30:
                elapsed = now - t2
                pct = n_done / n_combos * 100
                rate = n_done / max(1, elapsed)
                eta = (n_combos - n_done) / max(0.1, rate)
                cur_best_exp = max(r[3] for r in results) if results else 0
                print(f"  [{n_done:>5}/{n_combos}] {pct:5.1f}%  "
                      f"elapsed={elapsed:5.0f}s  rate={rate:6.1f}/s  eta={eta:5.0f}s  "
                      f"kept={n_kept}  best_exp={cur_best_exp:+.3f}R")
                sys.stdout.flush()
                last_log = now
    t3 = time.time()
    print(f"\n  Grid termine en {t3-t2:.0f}s. {len(results)} combos valides")
    sys.stdout.flush()

    # Sort par exp_R desc
    results.sort(key=lambda x: -x[3])

    # TOP 30
    print(f"\n>>> TOP 30 (par exp_R, n>={MIN_TRADES}, profit>={MIN_PROFITABLE_ASSETS}/28 actifs) :")
    print(f"{'rk':<4} {'n':<6} {'WR%':<6} {'exp_R':<8} {'DD':<7} {'+act':<5} "
          f"{'h1mom':<6} {'atr_max':<8} {'h_min':<6} {'h_max':<6} {'disp':<5} {'sl_buf':<7} {'cons'}")
    print("-" * 120)
    for i, (combo, n, wr, exp, total, dd, n_prof) in enumerate(results[:30], 1):
        h1m, atr_max, h_min, h_max, dm, sl_idx, mc = combo
        atr_str = "None" if atr_max is None else f"{atr_max:.1f}"
        sl_buf = SL_BUFS[sl_idx]
        print(f"{i:<4} {n:<6} {wr:<6.1f} {exp:<+8.3f} {dd:<7.1f} {n_prof:<5} "
              f"{h1m:<6.2f} {atr_str:<8} {h_min:<6} {h_max:<6} "
              f"{dm:<5.1f} {sl_buf:<7.2f} {mc}")

    # Sauvegarde top 200
    out = {"top": [
        {"combo": {"h1_mom_min": c[0], "atr_ratio_max": c[1], "hour_min": c[2],
                    "hour_max": c[3], "displacement_min": c[4],
                    "sl_buffer": SL_BUFS[c[5]], "min_consecutive": c[6]},
         "n": n, "wr": wr, "exp_R": exp, "total_R": total, "dd_R": dd,
         "n_profitable_assets": n_prof}
        for (c, n, wr, exp, total, dd, n_prof) in results[:200]
    ]}
    Path("/workspace/v22_grid_v3_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nTop 200 sauves dans /workspace/v22_grid_v3_results.json")
    print(f"\nTemps total : {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
