"""v22_grid_per_actif.py - Trouve la MEILLEURE config par actif (28 grids).

Pour chaque actif on cherche la combinaison de filtres qui maximise exp_R,
avec une contrainte minimum de trades pour pas overfitter sur 50 trades.

Contraintes par actif :
- Min 100 trades sur 3 ans (= ~1 par semaine, raisonnable)
- WR >= 50% (sinon mauvais signal meme avec exp positif)

Sortie : tableau (actif, meilleure config, n, WR, exp_R, DD)
+ stats agregees : edge moyen, edge moyen pondere par trades.
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

ASSETS = [
    "AUDUSD", "BTCUSD", "BVSPX", "CL-OIL", "Cocoa-C", "Coffee-C", "DJ30",
    "ETHUSD", "EURUSD", "FRA40", "GAS-C", "GBPUSD", "GER40", "HK50",
    "NAS100", "NZDUSD", "Nikkei225", "SP500", "Sugar-C", "UK100",
    "USDCAD", "USDCHF", "USDJPY", "USDMXN", "USDZAR", "Wheat-C",
    "XAGUSD", "XAUUSD",
]

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

# CONTRAINTES PAR ACTIF
# 1 trade/jour minimum sur 3 ans :
# - bourse  : ~250 jours x 3 ans = 750 trades
# - crypto/forex 24/7 : ~365 x 3 = 1095 trades
# On prend 800 pour les "marche ouvert ~250j/an" et le crypto/24h passera plus large
MIN_TRADES_PER_ASSET = 800   # >= ~1 trade/jour de bourse
MIN_WR_PER_ASSET = 50.0      # Au moins 50% WR
MIN_EXP_PER_ASSET = 0.10     # Au moins +0.10R par trade


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
    """Pour 1 actif : retourne (asset_name, list de rows)."""
    asset_name, sp = args
    try:
        df_m5 = pd.read_parquet(DATA / f"{asset_name}_M5.parquet")[["open","high","low","close"]]
        df_h1 = pd.read_parquet(DATA / f"{asset_name}_H1.parquet")[["open","high","low","close"]]
        df_d1 = pd.read_parquet(DATA / f"{asset_name}_D1.parquet")[["open","high","low","close"]]
    except Exception:
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
            row = [min_cons, atr_ratio, hour_fr, h1_mom, disp] + pnls
            rows.append(row)
    return asset_name, rows


# Worker globals
_G_arr = None
_G_min_cons = None
_G_atr_ratio = None
_G_hour_fr = None
_G_h1_mom = None
_G_disp = None
_G_pnls = None


def init_worker(arr):
    global _G_arr, _G_min_cons, _G_atr_ratio, _G_hour_fr, _G_h1_mom, _G_disp, _G_pnls
    _G_arr = arr
    _G_min_cons = arr[:, 0].astype(np.int8)
    _G_atr_ratio = arr[:, 1]
    _G_hour_fr = arr[:, 2]
    _G_h1_mom = arr[:, 3]
    _G_disp = arr[:, 4]
    _G_pnls = arr[:, 5:]


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
    valid = ~np.isnan(sub_pnls)
    sub_pnls = sub_pnls[valid]
    n = len(sub_pnls)
    if n < MIN_TRADES_PER_ASSET: return None
    wins = int((sub_pnls > 0).sum())
    wr = wins / n * 100
    if wr < MIN_WR_PER_ASSET: return None
    exp = float(sub_pnls.mean())
    if exp < MIN_EXP_PER_ASSET: return None
    total = float(sub_pnls.sum())
    eq = np.cumsum(sub_pnls); peak = np.maximum.accumulate(eq); dd = float((peak - eq).max())
    return (combo_packed, n, wr, exp, total, dd)


# Combos a tester (memes que v3)
def build_combos():
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
    return combos


# =====================================================================
# Phase 2 : architecture "128 workers globaux"
# =====================================================================
# Les arrays par actif sont stockes en globals (charges 1 fois par worker
# via fork COW sur Linux). Chaque job = (asset_name, chunk de combos).
# Avec 28 actifs et chunks de 1500 combos, on a ~28*48 = 1344 jobs qui
# saturent 128 workers en parallele.

# Globals partages dans chaque worker (fork COW)
_ARRAYS = {}

def _init_pool(arrays_dict):
    global _ARRAYS
    _ARRAYS = arrays_dict


def grid_chunk(args):
    """Worker : (name, chunk_combos) -> (name, partial_results).

    Optimisations :
    - Pre-cache par min_cons : on isole une fois pour toutes les rows ou min_cons==mc
    - Pre-cache du masque valid (non-NaN) par sl_idx
    - Pas de DD complet si exp < MIN_EXP_PER_ASSET (court-circuit)
    """
    name, chunk_combos = args
    arr = _ARRAYS.get(name)
    if arr is None or len(arr) == 0:
        return name, []
    min_cons_col = arr[:, 0].astype(np.int8)
    atr_ratio = arr[:, 1]
    hour_fr = arr[:, 2]
    h1_mom = arr[:, 3]
    disp = arr[:, 4]
    pnls = arr[:, 5:]

    # Pre-cache : pour chaque (mc, sl_idx), on garde les rows valides (mc match + non-nan pnl)
    # cache_mc[mc] = mask des rows ou min_cons == mc
    cache_mc = {mc: (min_cons_col == mc) for mc in MIN_CONS_LIST}

    results = []
    for combo in chunk_combos:
        h1m, atr_max, h_min, h_max, dm, sl_idx, mc = combo
        base_mask = cache_mc[mc]
        # combine
        mask = base_mask & (h1_mom >= h1m) & (hour_fr >= h_min) & (hour_fr < h_max) & (disp >= dm)
        if atr_max is not None:
            mask &= (atr_ratio < atr_max)
        if not mask.any():
            continue
        sub = pnls[mask, sl_idx]
        # filtre NaN
        sub = sub[~np.isnan(sub)]
        n = sub.shape[0]
        if n < MIN_TRADES_PER_ASSET:
            continue
        # Court-circuit : exp = sum/n, calcule sum d'abord (rapide)
        s = sub.sum()
        exp = s / n
        if exp < MIN_EXP_PER_ASSET:
            continue
        wins = int((sub > 0).sum())
        wr = wins / n * 100
        if wr < MIN_WR_PER_ASSET:
            continue
        eq = np.cumsum(sub); peak = np.maximum.accumulate(eq); dd = float((peak - eq).max())
        results.append((combo, n, wr, float(exp), float(s), dd))
    return name, results


def main():
    print("=" * 100)
    print(f"V22 GRID PER-ASSET - 28 actifs - meilleure config par actif")
    print(f"Contraintes : >={MIN_TRADES_PER_ASSET} trades (~1/jour), WR>={MIN_WR_PER_ASSET}%, exp>=+{MIN_EXP_PER_ASSET}R")
    print("=" * 100)

    # Phase 1 : build datasets pour les 28 actifs
    print(f"\n>>> Phase 1 : build datasets pour {len(ASSETS)} actifs (en parallele)...")
    sys.stdout.flush()
    t0 = time.time()
    args_assets = [(name, SPREADS.get(name, 0.001)) for name in ASSETS]
    datasets = {}  # name -> rows
    with ProcessPoolExecutor(max_workers=28) as ex:
        futures = {ex.submit(build_dataset_for_asset, a): a for a in args_assets}
        n_done = 0
        for f in as_completed(futures):
            name, rows = f.result()
            datasets[name] = rows
            n_done += 1
            elapsed = time.time() - t0
            print(f"  [{n_done:>2}/{len(ASSETS)}] +{elapsed:5.0f}s  {name:<12} : {len(rows):>5} rows")
            sys.stdout.flush()
    print(f"\nPhase 1 termine en {time.time()-t0:.0f}s. Total : {sum(len(r) for r in datasets.values())} rows")
    sys.stdout.flush()

    # Phase 2 : 128 workers globaux, jobs = (actif, chunk_combos)
    combos = build_combos()
    N_WORKERS = 128
    CHUNK_SIZE = 1500
    print(f"\n>>> Phase 2 : {len(combos)} combos x 28 actifs sur {N_WORKERS} workers...")
    sys.stdout.flush()

    # Prepare arrays par actif (transferes via fork COW)
    arrays = {}
    for name in ASSETS:
        rows = datasets.get(name, [])
        if rows:
            arrays[name] = np.array(rows, dtype=np.float32)
        else:
            print(f"  SKIP {name:<12} (0 rows)")
    print(f"  {len(arrays)} actifs a traiter")

    # Decoupe les combos en chunks
    chunks = [combos[i:i+CHUNK_SIZE] for i in range(0, len(combos), CHUNK_SIZE)]
    print(f"  {len(chunks)} chunks de {CHUNK_SIZE} combos -> {len(arrays)*len(chunks)} jobs total")
    sys.stdout.flush()

    # Accumulateur : nom -> liste results
    per_asset_results = {name: [] for name in arrays}

    jobs = [(name, chunk) for name in arrays for chunk in chunks]
    t1 = time.time()
    n_done = 0
    last_print = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS,
                               initializer=_init_pool,
                               initargs=(arrays,)) as ex:
        futures = [ex.submit(grid_chunk, j) for j in jobs]
        for f in as_completed(futures):
            name, partial = f.result()
            per_asset_results[name].extend(partial)
            n_done += 1
            elapsed = time.time() - t1
            if n_done - last_print >= 50 or n_done == len(jobs):
                last_print = n_done
                rate = n_done / max(elapsed, 0.01)
                eta = (len(jobs) - n_done) / max(rate, 0.01)
                print(f"  [{n_done:>5}/{len(jobs)}] +{elapsed:5.0f}s  {rate:.1f} jobs/s  ETA {eta:.0f}s")
                sys.stdout.flush()

    # Compile top5 par actif
    per_asset_best = {}
    per_asset_top5 = {}
    print()
    for name in ASSETS:
        results = per_asset_results.get(name, [])
        if not results:
            print(f"  {name:<12} : aucune combo valide")
            continue
        results.sort(key=lambda x: -x[3])
        per_asset_top5[name] = results[:5]
        best = results[0]
        per_asset_best[name] = best
        combo, n, wr, exp, total, dd = best
        h1m, atr_max, h_min, h_max, dm, sl_idx, mc = combo
        atr_str = "None" if atr_max is None else f"{atr_max:.1f}"
        sl_buf = SL_BUFS[sl_idx]
        print(f"  {name:<12} : WR={wr:>5.1f}% exp={exp:>+6.3f}R n={n:>4} DD={dd:>5.1f}R "
              f"| h1={h1m:.2f} atr={atr_str} h={h_min}-{h_max} d={dm:.1f} sl={sl_buf:.2f} c={mc}")
    sys.stdout.flush()

    t2 = time.time()
    print(f"\nPhase 2 termine en {t2-t1:.0f}s")

    # === AGGREGATION ===
    print("\n" + "=" * 100)
    print("RESUME PAR ACTIF (best config)")
    print("=" * 100)
    print(f"{'actif':<12} {'n':<6} {'WR%':<6} {'exp_R':<8} {'DD':<6} {'h1mom':<6} {'atr':<6} {'h_min':<6} {'h_max':<6} {'disp':<5} {'sl_buf':<7} {'cons'}")
    print("-" * 100)
    total_exp_pondere = 0.0; total_n = 0; total_pnl = 0.0
    for name in ASSETS:
        if name not in per_asset_best:
            print(f"{name:<12} pas de config valide")
            continue
        combo, n, wr, exp, total, dd = per_asset_best[name]
        h1m, atr_max, h_min, h_max, dm, sl_idx, mc = combo
        atr_str = "None" if atr_max is None else f"{atr_max:.1f}"
        sl_buf = SL_BUFS[sl_idx]
        print(f"{name:<12} {n:<6} {wr:<6.1f} {exp:<+8.3f} {dd:<6.1f} "
              f"{h1m:<6.2f} {atr_str:<6} {h_min:<6} {h_max:<6} {dm:<5.1f} {sl_buf:<7.2f} {mc}")
        total_exp_pondere += exp * n
        total_n += n
        total_pnl += total

    if total_n > 0:
        avg_exp_pondere = total_exp_pondere / total_n
        # Edge moyen non pondere
        avg_exp_brut = np.mean([per_asset_best[n][3] for n in per_asset_best])
        print(f"\n--- AGGREGE ---")
        print(f"  Actifs avec config valide : {len(per_asset_best)}/28")
        print(f"  Total trades : {total_n} (~{total_n/3.25:.0f}/an = {total_n/3.25/250:.1f}/jour)")
        print(f"  Total PnL    : {total_pnl:+.1f}R")
        print(f"  Edge moyen brut         : {avg_exp_brut:+.3f}R")
        print(f"  Edge moyen pondere (n)  : {avg_exp_pondere:+.3f}R")

    # Sauvegarde JSON
    out = {"per_asset": {}}
    for name, best in per_asset_best.items():
        combo, n, wr, exp, total, dd = best
        h1m, atr_max, h_min, h_max, dm, sl_idx, mc = combo
        out["per_asset"][name] = {
            "n": n, "wr": wr, "exp_R": exp, "total_R": total, "dd_R": dd,
            "config": {
                "h1_mom_min": h1m, "atr_ratio_max": atr_max,
                "hour_min": h_min, "hour_max": h_max,
                "displacement_min": dm,
                "sl_buffer": SL_BUFS[sl_idx],
                "min_consecutive": mc,
            },
            "top5": [
                {"exp": r[3], "n": r[1], "wr": r[2],
                 "config": {"h1m": r[0][0], "atr_max": r[0][1], "h_min": r[0][2],
                             "h_max": r[0][3], "disp": r[0][4],
                             "sl_buf": SL_BUFS[r[0][5]], "cons": r[0][6]}}
                for r in per_asset_top5[name]
            ],
        }
    out["aggregated"] = {
        "valid_assets": len(per_asset_best),
        "total_trades": total_n,
        "total_pnl_R": total_pnl,
        "avg_exp_brut": avg_exp_brut if total_n > 0 else 0,
        "avg_exp_pondere": avg_exp_pondere if total_n > 0 else 0,
    }
    Path("/workspace/v22_per_asset_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nDetails sauves dans /workspace/v22_per_asset_results.json")
    print(f"\nTemps total : {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
