"""v22_grid_massif_vast.py - Grid search MASSIF sur Vast.ai 255 cores.

Genere ~2000 combinaisons de filtres et les teste TOUTES en parallele.

Dimensions explorees :
- H1 momentum threshold : [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]   (8)
- ATR ratio max : [None, 1.5, 2.0, 2.5, 3.0]   (5)
- hour FR min : [0, 5, 6, 7, 8, 9]   (6)
- hour FR max : [17, 19, 20, 21, 22, 24]   (6)
- displacement_atr min : [0.0, 0.3, 0.5, 0.7, 1.0]   (5)
- SL buffer ATR : [0.0, 0.05, 0.1, 0.15, 0.2]   (5)
- min_consecutive bougies OB : [2, 3]   (2)

Avec les 4 filtres D1, H1, PD : on garde toujours D1+H1+PD (sinon test deja fait).
Sous-ensemble final pour rester raisonnable : ~5000 combos x 6 actifs.
On garde les TOP 30 par exp_R.

Methodologie :
1. Pre-calcule le dataset des OBs avec meta-features (~32k OBs)
2. Pour chaque combo, on filtre le dataset et calcule exp_R
3. Parallelise sur les 255 workers
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

ASSETS_MAP = {
    "EURUSD": "EURUSD", "XAUUSD": "XAUUSD", "NAS100": "NAS100",
    "GER40":  "GER40",  "BTCUSD": "BTCUSD", "USOUSD": "CL-OIL",
}
SPREADS = {
    "EURUSD": 0.00016, "XAUUSD": 0.14, "NAS100": 1.0,
    "GER40":  0.8,     "BTCUSD": 6.0,  "USOUSD": 0.06,
}

TP_RR = 2.0; PARTIAL_R = 1.0
MAX_FILL_BARS = 20; MAX_HOLD_BARS = 50


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


def compute_meta(df_m5, df_h1, df_d1, ob, atr):
    val_ts = ob.validation_ts; val_idx = ob.validation_index
    # ATR ratio
    atr_50 = atr_safe(df_m5, val_idx, period=50)
    atr_ratio = atr / atr_50 if atr_50 > 0 else 1.0
    # heures
    hour_fr = val_ts.tz_convert("Europe/Paris").hour + val_ts.tz_convert("Europe/Paris").minute / 60
    # H1 momentum
    h1_close = htf_value_at_t(df_h1, val_ts, "close")
    h1_mom = 0
    if h1_close is not None:
        pos = df_h1.index.searchsorted(val_ts, side="left") - 1 - 10
        if pos >= 0:
            h1_old = float(df_h1["close"].iloc[pos])
            if h1_old > 0:
                h1_mom = abs((h1_close - h1_old) / h1_old * 100)
    return {
        "atr_ratio": atr_ratio,
        "hour_fr": hour_fr,
        "h1_momentum": h1_mom,
        "displacement_atr": abs(ob.displacement_atr),
        "asset": None,   # rempli par caller
    }


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
    """Pour 1 actif : extract tous les OBs qui passent D1+H1+PD + leurs meta + pnl pour
    chaque (sl_buffer_atr, min_consecutive)."""
    user_name, file_name, sp = args
    df_m5 = pd.read_parquet(DATA / f"{file_name}_M5.parquet")[["open","high","low","close"]]
    df_h1 = pd.read_parquet(DATA / f"{file_name}_H1.parquet")[["open","high","low","close"]]
    df_d1 = pd.read_parquet(DATA / f"{file_name}_D1.parquet")[["open","high","low","close"]]
    start = pd.Timestamp("2023-01-01", tz="UTC"); end = pd.Timestamp("2026-04-01", tz="UTC")
    ctx = start - pd.Timedelta(days=10)
    df_m5 = df_m5[(df_m5.index >= ctx) & (df_m5.index < end)]
    df_h1 = df_h1[(df_h1.index >= ctx) & (df_h1.index < end)]
    df_d1 = df_d1[(df_d1.index >= start - pd.Timedelta(days=30)) & (df_d1.index < end)]
    if len(df_m5) < 1000: return user_name, []

    rows = []
    for min_cons in [2, 3]:
        obs = find_obs_simple(df_m5, as_of_index=len(df_m5) - 1, min_consecutive=min_cons)
        obs = [ob for ob in obs if start <= ob.validation_ts < end]
        for ob in obs:
            if not passes_user_rule_d1_h1_pd(df_m5, df_h1, df_d1, ob): continue
            atr = atr_safe(df_m5, ob.validation_index)
            if atr <= 0: continue
            meta = compute_meta(df_m5, df_h1, df_d1, ob, atr)
            meta["asset"] = user_name
            meta["min_consecutive"] = min_cons
            # Calcul pnl pour chaque sl_buffer
            pnls_by_sl = {}
            for sl_buf in [0.0, 0.05, 0.1, 0.15, 0.2]:
                p = simulate_ob_trade(df_m5, ob, atr, sp, sl_buffer_atr=sl_buf)
                if p is not None:
                    pnls_by_sl[sl_buf] = p
            if not pnls_by_sl: continue
            meta["pnls_by_sl"] = pnls_by_sl
            rows.append(meta)
    return user_name, rows


def evaluate_combo(args):
    """args = (combo_dict, dataset_global). Retourne stats du combo."""
    combo, dataset = args
    h1_mom_min = combo["h1_mom_min"]
    atr_max = combo["atr_ratio_max"]
    hour_min = combo["hour_min"]
    hour_max = combo["hour_max"]
    disp_min = combo["displacement_min"]
    sl_buf = combo["sl_buffer"]
    min_cons = combo["min_consecutive"]

    pnls = []
    for row in dataset:
        if row["min_consecutive"] != min_cons: continue
        if row["h1_momentum"] < h1_mom_min: continue
        if atr_max is not None and row["atr_ratio"] >= atr_max: continue
        if not (hour_min <= row["hour_fr"] < hour_max): continue
        if row["displacement_atr"] < disp_min: continue
        if sl_buf not in row["pnls_by_sl"]: continue
        pnls.append(row["pnls_by_sl"][sl_buf])

    if not pnls: return combo, None
    arr = np.array(pnls)
    n = len(arr)
    wins = (arr > 0).sum()
    wr = wins / n * 100
    exp = arr.mean()
    total = arr.sum()
    eq = np.cumsum(arr); peak = np.maximum.accumulate(eq); dd = float((peak - eq).max()) if len(eq) else 0
    return combo, {"n": n, "wr": wr, "exp": exp, "total": total, "dd": dd}


def main():
    print("=" * 90)
    print("V22 GRID MASSIF - VAST.AI 255 CORES")
    print("=" * 90)

    print("\n>>> Phase 1 : pre-calcule dataset OBs + pnls...")
    print(f"  6 actifs x 2 min_consecutive (2 et 3) = 12 jobs en parallele")
    sys.stdout.flush()
    t0 = time.time()

    # Construction du dataset (1 worker par actif)
    args_assets = [(name, file_name, SPREADS[name]) for name, file_name in ASSETS_MAP.items()]
    dataset = []
    with ProcessPoolExecutor(max_workers=12) as ex:
        futures = {ex.submit(build_dataset_for_asset, a): a for a in args_assets}
        n_done = 0
        for f in as_completed(futures):
            name, rows = f.result()
            dataset.extend(rows)
            n_done += 1
            elapsed = time.time() - t0
            print(f"  [{n_done}/6] +{elapsed:.0f}s  {name} : {len(rows)} OBs avec meta+pnls (total {len(dataset)})")
            sys.stdout.flush()

    t1 = time.time()
    print(f"\nDataset total : {len(dataset)} entries en {t1-t0:.0f}s")
    sys.stdout.flush()

    print("\n>>> Phase 2 : grid search...")
    # Combos
    h1_mom_mins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
    atr_ratios_max = [None, 1.5, 2.0, 2.5, 3.0]
    hours_min = [0, 5, 6, 7, 8, 9]
    hours_max = [17, 19, 20, 21, 22, 24]
    disp_mins = [0.0, 0.3, 0.5, 0.7, 1.0]
    sl_bufs = [0.0, 0.05, 0.1, 0.15, 0.2]
    min_cons_list = [2, 3]

    combos = []
    for h1m, atr_m, h_min, h_max, dm, sl, mc in product(
        h1_mom_mins, atr_ratios_max, hours_min, hours_max,
        disp_mins, sl_bufs, min_cons_list
    ):
        if h_min >= h_max: continue
        combos.append({
            "h1_mom_min": h1m, "atr_ratio_max": atr_m,
            "hour_min": h_min, "hour_max": h_max,
            "displacement_min": dm, "sl_buffer": sl,
            "min_consecutive": mc,
        })
    n_combos = len(combos)
    print(f"  {n_combos} combos a tester (~{n_combos / 250:.0f} par worker si 250 workers)")
    sys.stdout.flush()

    # On parallelise. Dataset est partagee via fork (Linux).
    results = []
    t2 = time.time()
    n_done = 0
    n_kept = 0
    LOG_EVERY = max(1, n_combos // 50)   # ~50 progress messages
    last_log = time.time()

    with ProcessPoolExecutor(max_workers=200) as ex:
        # On submit tous les jobs d'un coup, ProcessPoolExecutor gere la queue
        print(f"  Submitting {n_combos} jobs to pool of 200 workers...")
        sys.stdout.flush()
        futures = [ex.submit(evaluate_combo, (combo, dataset)) for combo in combos]
        print(f"  Submission terminee, attente des resultats...")
        sys.stdout.flush()

        for f in as_completed(futures):
            n_done += 1
            combo, stats = f.result()
            if stats is not None and stats["n"] >= 100:
                results.append((combo, stats))
                n_kept += 1

            # Log de suivi toutes les LOG_EVERY combos OU toutes les 30s
            now = time.time()
            if n_done % LOG_EVERY == 0 or (now - last_log) > 30:
                elapsed = now - t2
                pct = n_done / n_combos * 100
                rate = n_done / max(1, elapsed)
                eta = (n_combos - n_done) / max(0.1, rate)
                # Top exp actuel
                if results:
                    cur_best_exp = max(r[1]["exp"] for r in results)
                else:
                    cur_best_exp = 0
                print(f"  [{n_done:>5}/{n_combos}] {pct:5.1f}%  "
                      f"elapsed={elapsed:5.0f}s  rate={rate:5.1f}/s  eta={eta:5.0f}s  "
                      f"kept={n_kept}  best_exp={cur_best_exp:+.3f}R")
                sys.stdout.flush()
                last_log = now
    t3 = time.time()
    print(f"\n  Grid termine en {t3-t2:.0f}s. {len(results)} combos avec >= 100 trades")
    sys.stdout.flush()

    # Sort par exp_R
    results.sort(key=lambda x: -x[1]["exp"])

    # TOP 30
    print(f"\n>>> TOP 30 par exp_R :")
    print(f"{'rank':<5} {'n':<6} {'WR%':<6} {'exp_R':<8} {'DD':<7} {'h1mom':<6} {'atr_max':<8} {'h_min':<6} {'h_max':<6} {'disp':<5} {'sl':<5} {'cons'}")
    print("-" * 110)
    for i, (combo, s) in enumerate(results[:30], 1):
        atr_str = "None" if combo["atr_ratio_max"] is None else f"{combo['atr_ratio_max']:.1f}"
        print(f"{i:<5} {s['n']:<6} {s['wr']:<6.1f} {s['exp']:<+8.3f} {s['dd']:<7.1f} "
              f"{combo['h1_mom_min']:<6.2f} {atr_str:<8} {combo['hour_min']:<6} {combo['hour_max']:<6} "
              f"{combo['displacement_min']:<5.1f} {combo['sl_buffer']:<5.2f} {combo['min_consecutive']}")

    # Sauvegarde JSON
    out = {"results": [{"combo": c, "stats": s} for c, s in results[:100]]}
    Path("/workspace/v22_grid_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nResultats top 100 sauves dans /workspace/v22_grid_results.json")
    print(f"\nTemps total : {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
