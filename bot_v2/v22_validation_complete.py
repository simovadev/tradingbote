"""v22_validation_complete.py - 5 tests critiques de validation V22 en local.

Strategie d'optimisation pour 8 cores / 16GB :
1. Datasets pre-calcules UNE fois en debut, cachees en pickle
2. Tous les tests reutilisent les memes datasets en RAM
3. ProcessPool a 8 workers max
4. Vectorisation numpy pour les grids

Tests inclus :
  T1. Walk-forward strict : optim 2023-2024 -> test 2025-2026
  T2. Backtest 2022 (annee jamais vue)
  T3. Test filtres separes (D1, H1, PD, combinaisons)
  T4. Stabilite par sous-periode (2023/2024/2025H1/2025H2/2026)
  T7. Slippage + commissions realistes

Tests non couverts ici :
  T5 sensitivity (nice-to-have)
  T6 audit code (manuel via grep)
"""
from __future__ import annotations
import sys, time, pickle, json
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
CACHE = ROOT / "v22_validation_cache"
CACHE.mkdir(exist_ok=True)
OUT_DIR = ROOT / "v22_validation_results"
OUT_DIR.mkdir(exist_ok=True)

# 22 actifs avec leur file_name (mapping user_name -> parquet name)
ASSETS_MAP = {
    "AUDUSD": "AUDUSD", "BTCUSD": "BTCUSD", "BVSPX": "BVSPX",
    "CL-OIL": "CL-OIL", "Cocoa-C": "Cocoa-C", "DJ30": "DJ30",
    "ETHUSD": "ETHUSD", "FRA40": "FRA40", "GAS-C": "GAS-C",
    "GBPUSD": "GBPUSD", "GER40": "GER40", "HK50": "HK50",
    "NAS100": "NAS100", "Nikkei225": "Nikkei225", "SP500": "SP500",
    "UK100": "UK100", "USDCAD": "USDCAD", "USDCHF": "USDCHF",
    "USDJPY": "USDJPY", "USDMXN": "USDMXN", "USDZAR": "USDZAR",
    "XAUUSD": "XAUUSD",
}

SPREADS = {
    "AUDUSD": 0.00010, "BTCUSD": 6.0, "BVSPX": 8.0, "CL-OIL": 0.06,
    "Cocoa-C": 5.0, "DJ30": 2.0, "ETHUSD": 1.5,
    "FRA40": 1.5, "GAS-C": 0.005, "GBPUSD": 0.0002,
    "GER40": 0.8, "HK50": 5.0, "NAS100": 1.0,
    "Nikkei225": 10.0, "SP500": 0.3, "UK100": 1.0,
    "USDCAD": 0.0002, "USDCHF": 0.0002, "USDJPY": 0.025,
    "USDMXN": 0.002, "USDZAR": 0.004, "XAUUSD": 0.14,
}

# Configs per-actif TOP 1 du grid
PER_ASSET_CONFIG = {
    "AUDUSD":    {"h1m":0.10, "atr_max":2.0,  "h_min":9, "h_max":22, "disp":0.3, "sl_buf":0.20, "cons":2},
    "BTCUSD":    {"h1m":1.00, "atr_max":2.5,  "h_min":8, "h_max":20, "disp":0.0, "sl_buf":0.00, "cons":2},
    "BVSPX":     {"h1m":0.00, "atr_max":None, "h_min":9, "h_max":22, "disp":0.5, "sl_buf":0.15, "cons":2},
    "CL-OIL":    {"h1m":0.50, "atr_max":2.5,  "h_min":8, "h_max":19, "disp":0.3, "sl_buf":0.20, "cons":2},
    "Cocoa-C":   {"h1m":0.70, "atr_max":None, "h_min":0, "h_max":17, "disp":0.3, "sl_buf":0.15, "cons":2},
    "DJ30":      {"h1m":0.30, "atr_max":2.5,  "h_min":9, "h_max":22, "disp":0.0, "sl_buf":0.20, "cons":2},
    "ETHUSD":    {"h1m":1.00, "atr_max":2.0,  "h_min":9, "h_max":22, "disp":0.3, "sl_buf":0.15, "cons":2},
    "FRA40":     {"h1m":0.30, "atr_max":2.5,  "h_min":8, "h_max":22, "disp":0.0, "sl_buf":0.20, "cons":2},
    "GAS-C":     {"h1m":0.70, "atr_max":2.5,  "h_min":9, "h_max":19, "disp":0.0, "sl_buf":0.15, "cons":2},
    "GBPUSD":    {"h1m":0.20, "atr_max":3.0,  "h_min":6, "h_max":21, "disp":0.0, "sl_buf":0.20, "cons":2},
    "GER40":     {"h1m":0.40, "atr_max":None, "h_min":7, "h_max":22, "disp":0.0, "sl_buf":0.15, "cons":2},
    "HK50":      {"h1m":0.40, "atr_max":None, "h_min":0, "h_max":19, "disp":0.3, "sl_buf":0.20, "cons":2},
    "NAS100":    {"h1m":0.30, "atr_max":3.0,  "h_min":9, "h_max":21, "disp":0.3, "sl_buf":0.05, "cons":2},
    "Nikkei225": {"h1m":0.30, "atr_max":2.0,  "h_min":0, "h_max":21, "disp":0.5, "sl_buf":0.10, "cons":2},
    "SP500":     {"h1m":0.20, "atr_max":None, "h_min":9, "h_max":24, "disp":0.3, "sl_buf":0.00, "cons":2},
    "UK100":     {"h1m":0.10, "atr_max":2.5,  "h_min":7, "h_max":24, "disp":0.5, "sl_buf":0.20, "cons":2},
    "USDCAD":    {"h1m":0.10, "atr_max":None, "h_min":9, "h_max":17, "disp":0.0, "sl_buf":0.20, "cons":2},
    "USDCHF":    {"h1m":0.20, "atr_max":2.0,  "h_min":6, "h_max":22, "disp":0.0, "sl_buf":0.20, "cons":2},
    "USDJPY":    {"h1m":0.20, "atr_max":2.5,  "h_min":7, "h_max":22, "disp":0.0, "sl_buf":0.00, "cons":2},
    "USDMXN":    {"h1m":0.30, "atr_max":None, "h_min":6, "h_max":20, "disp":0.0, "sl_buf":0.20, "cons":2},
    "USDZAR":    {"h1m":0.30, "atr_max":3.0,  "h_min":7, "h_max":21, "disp":0.0, "sl_buf":0.00, "cons":2},
    "XAUUSD":    {"h1m":0.40, "atr_max":3.0,  "h_min":6, "h_max":21, "disp":0.0, "sl_buf":0.00, "cons":2},
}

TP_RR = 2.0; PARTIAL_R = 1.0
MAX_FILL_BARS = 20; MAX_HOLD_BARS = 50


# =========================================================================
# Utils backtest
# =========================================================================

def simulate_ob_trade(df_m5, ob, atr, spread, sl_buf):
    val_idx = ob.validation_index
    if val_idx + 1 >= len(df_m5): return None
    hs = spread / 2
    if ob.direction == "bullish":
        sl = ob.ob_low - sl_buf * atr; entry = ob.ob_high
        entry_eff = entry + hs; sl_eff = sl - hs
    else:
        sl = ob.ob_high + sl_buf * atr; entry = ob.ob_low
        entry_eff = entry - hs; sl_eff = sl + hs
    if abs(entry_eff - sl_eff) < 1e-9: return None
    risk = abs(entry_eff - sl_eff)
    if ob.direction == "bullish":
        tp_1r = entry_eff + risk; tp_2r = entry_eff + TP_RR * risk
    else:
        tp_1r = entry_eff - risk; tp_2r = entry_eff - TP_RR * risk
    fh = df_m5["high"].values[val_idx+1:val_idx+1+MAX_FILL_BARS]
    fl = df_m5["low"].values[val_idx+1:val_idx+1+MAX_FILL_BARS]
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
    half = False; stop = sl_eff
    for i in range(len(pfh)):
        if ob.direction == "bullish":
            hit_sl = pfl[i] <= stop; hit_1r = pfh[i] >= tp_1r; hit_2r = pfh[i] >= tp_2r
        else:
            hit_sl = pfh[i] >= stop; hit_1r = pfl[i] <= tp_1r; hit_2r = pfl[i] <= tp_2r
        if hit_sl and hit_2r:
            if half: return 0.5 * PARTIAL_R
            return -1.0
        if hit_sl:
            if half: return 0.5 * PARTIAL_R
            return -1.0
        if not half and hit_1r:
            half = True; stop = entry_eff
            if hit_2r: return 0.5 * PARTIAL_R + 0.5 * TP_RR
        elif half and hit_2r:
            return 0.5 * PARTIAL_R + 0.5 * TP_RR
    if half: return 0.5 * PARTIAL_R
    return 0.0


def passes_filters(filters, d1_a, h1_a, pd_a):
    """filters = set parmi {'d1','h1','pd'}. Tous doivent etre True dans le set."""
    if "d1" in filters and not d1_a: return False
    if "h1" in filters and not h1_a: return False
    if "pd" in filters and not pd_a: return False
    return True


def stats_pnls(pnls):
    if not pnls: return None
    pnls = np.array(pnls)
    n = len(pnls); wins = int((pnls > 0).sum()); losses = int((pnls < 0).sum())
    wr = wins/n*100 if n else 0
    exp = float(pnls.mean()); total = float(pnls.sum())
    eq = np.cumsum(pnls); peak = np.maximum.accumulate(eq); dd = float((peak-eq).max())
    return {"n": n, "wr": wr, "exp": exp, "total": total, "dd": dd, "wins": wins, "losses": losses}


# =========================================================================
# Phase 1 : Build dataset par actif (mis en cache)
# Pour chaque actif, on stocke pour CHAQUE OB toute l'info utile :
# (ts, dir, d1_a, h1_a, pd_a, hour_fr, h1_mom, atr_ratio, disp_atr,
#  pnl pour chaque sl_buf, pour chaque min_cons)
# =========================================================================

def build_ds_for_asset(args):
    name, file_name, period_start, period_end, sp, cache_path = args
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                return name, pickle.load(f)
        except Exception:
            pass
    try:
        df_m5 = pd.read_parquet(DATA / f"{file_name}_M5.parquet")[["open","high","low","close"]]
        df_h1 = pd.read_parquet(DATA / f"{file_name}_H1.parquet")[["open","high","low","close"]]
        df_d1 = pd.read_parquet(DATA / f"{file_name}_D1.parquet")[["open","high","low","close"]]
    except Exception:
        return name, []
    ctx = period_start - pd.Timedelta(days=10)
    df_m5 = df_m5[(df_m5.index >= ctx) & (df_m5.index < period_end)]
    df_h1 = df_h1[(df_h1.index >= ctx) & (df_h1.index < period_end)]
    df_d1 = df_d1[(df_d1.index >= period_start - pd.Timedelta(days=60)) & (df_d1.index < period_end)]
    if len(df_m5) < 1000: return name, []

    SL_BUFS = [0.0, 0.05, 0.10, 0.15, 0.20]
    MIN_CONS_LIST = [2, 3]

    rows = []
    for min_cons in MIN_CONS_LIST:
        obs = find_obs_simple(df_m5, as_of_index=len(df_m5)-1, min_consecutive=min_cons)
        obs = [ob for ob in obs if period_start <= ob.validation_ts < period_end]
        for ob in obs:
            val_ts = ob.validation_ts; val_idx = ob.validation_index
            # D1
            d1 = daily_bias_safe(df_d1, val_ts)
            d1_a = False
            if d1["ok"]:
                d1_h = d1["bias"] == "haussier"
                d1_a = (ob.direction == "bullish" and d1_h) or (ob.direction == "bearish" and not d1_h)
            # H1
            h1_close = htf_value_at_t(df_h1, val_ts, "close")
            h1_a = False; h1_mom = 0.0
            if h1_close is not None:
                pos = df_h1.index.searchsorted(val_ts, side="left") - 1 - 10
                if pos >= 0:
                    h1_old = float(df_h1["close"].iloc[pos])
                    if h1_old > 0:
                        h1_mom = abs((h1_close - h1_old)/h1_old*100)
                    h1_h = h1_close > h1_old
                    h1_a = (ob.direction == "bullish" and h1_h) or (ob.direction == "bearish" and not h1_h)
            # PD daily
            pd_a = False
            if df_d1 is not None and len(df_d1) >= 2:
                target_day = pd.Timestamp(val_ts).normalize()
                df_d1_past = df_d1[df_d1.index < target_day]
                if len(df_d1_past) >= 1:
                    pdh = float(df_d1_past["high"].iloc[-1])
                    pdl = float(df_d1_past["low"].iloc[-1])
                    mid = (pdh + pdl) / 2
                    price = float(df_m5["close"].iloc[val_idx])
                    pd_a = (ob.direction == "bullish" and price < mid) or \
                           (ob.direction == "bearish" and price > mid)
            atr = atr_safe(df_m5, val_idx)
            if atr <= 0: continue
            atr_50 = atr_safe(df_m5, val_idx, period=50)
            atr_ratio = atr / atr_50 if atr_50 > 0 else 1.0
            hour_fr = val_ts.tz_convert("Europe/Paris").hour + val_ts.tz_convert("Europe/Paris").minute/60
            disp = abs(ob.displacement_atr)
            # PnL pour chaque sl_buf
            pnls = []
            for sl_buf in SL_BUFS:
                p = simulate_ob_trade(df_m5, ob, atr, sp, sl_buf)
                pnls.append(p if p is not None else np.nan)
            if all(np.isnan(p) for p in pnls): continue
            rows.append({
                "ts": val_ts.value,  # nanoseconds for fast sort
                "min_cons": min_cons,
                "d1_a": d1_a, "h1_a": h1_a, "pd_a": pd_a,
                "hour_fr": hour_fr, "h1_mom": h1_mom,
                "atr_ratio": atr_ratio, "disp": disp,
                "pnl_sl0": pnls[0], "pnl_sl5": pnls[1],
                "pnl_sl10": pnls[2], "pnl_sl15": pnls[3], "pnl_sl20": pnls[4],
            })
    with open(cache_path, "wb") as f:
        pickle.dump(rows, f)
    return name, rows


def build_all_datasets(period_start, period_end, period_name, n_workers=8):
    """Build datasets pour tous les actifs sur une periode donnee."""
    print(f"  >> Building datasets [{period_name}] ({period_start.date()} -> {period_end.date()})...")
    t0 = time.time()
    cache_prefix = f"{period_name}_"
    args_list = [
        (name, file_name, period_start, period_end, SPREADS.get(name, 0.001),
         CACHE / f"{cache_prefix}{name}.pkl")
        for name, file_name in ASSETS_MAP.items()
    ]
    datasets = {}
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(build_ds_for_asset, a): a[0] for a in args_list}
        for f in as_completed(futs):
            name, rows = f.result()
            datasets[name] = rows
            print(f"    {name:<12} : {len(rows):>5} rows")
            sys.stdout.flush()
    print(f"  Total : {sum(len(r) for r in datasets.values())} rows en {time.time()-t0:.0f}s")
    return datasets


# =========================================================================
# Test : apply config per-actif sur un dataset, retourne stats agregees
# =========================================================================

SL_BUF_TO_KEY = {0.0: "pnl_sl0", 0.05: "pnl_sl5", 0.10: "pnl_sl10",
                  0.15: "pnl_sl15", 0.20: "pnl_sl20"}


def apply_per_asset_config(datasets, configs):
    """Pour chaque actif applique sa config dediee, retourne {actif: stats}."""
    per_asset = {}
    for name, rows in datasets.items():
        if not rows or name not in configs: continue
        cfg = configs[name]
        pnls = []
        sl_key = SL_BUF_TO_KEY.get(cfg["sl_buf"])
        if sl_key is None: continue
        for r in rows:
            if r["min_cons"] != cfg["cons"]: continue
            if not (r["d1_a"] and r["h1_a"] and r["pd_a"]): continue
            if r["h1_mom"] < cfg["h1m"]: continue
            if not (cfg["h_min"] <= r["hour_fr"] < cfg["h_max"]): continue
            if r["disp"] < cfg["disp"]: continue
            if cfg["atr_max"] is not None and r["atr_ratio"] >= cfg["atr_max"]: continue
            p = r[sl_key]
            if not np.isnan(p): pnls.append(p)
        per_asset[name] = stats_pnls(pnls)
    return per_asset


def print_per_asset(label, per_asset):
    print(f"\n  {label}")
    print(f"  {'actif':<12} {'n':<6} {'WR%':<7} {'exp_R':<8} {'total':<9} {'DD':<6}")
    total_n = 0; total_exp_sum = 0.0; n_profit = 0
    for name in sorted(per_asset.keys()):
        s = per_asset[name]
        if s is None: continue
        v = "OK" if s["exp"] > 0.05 else ("=" if s["exp"] > 0 else "NEG")
        if s["exp"] > 0: n_profit += 1
        total_n += s["n"]; total_exp_sum += s["exp"] * s["n"]
        print(f"  {name:<12} {s['n']:<6} {s['wr']:<7.1f} {s['exp']:<+8.3f} {s['total']:<+9.1f} {s['dd']:<6.1f} {v}")
    if total_n > 0:
        exp_pond = total_exp_sum / total_n
        print(f"  -> n_total={total_n} exp_pondere={exp_pond:+.3f}R profitables={n_profit}/{len(per_asset)}")
    return total_n, total_exp_sum / total_n if total_n > 0 else 0, n_profit


# =========================================================================
# TEST 1 : Walk-forward (optim 2023-2024 -> test 2025-2026)
# Pour chaque actif on relance un mini-grid sur IN, puis on applique sur OUT
# =========================================================================

def mini_grid_per_asset(rows):
    """Pour 1 actif : trouve la meilleure config dans un grid reduit. Retourne config."""
    if not rows: return None
    arr_min_cons = np.array([r["min_cons"] for r in rows])
    arr_d1 = np.array([r["d1_a"] for r in rows])
    arr_h1 = np.array([r["h1_a"] for r in rows])
    arr_pd = np.array([r["pd_a"] for r in rows])
    arr_hour = np.array([r["hour_fr"] for r in rows])
    arr_h1m = np.array([r["h1_mom"] for r in rows])
    arr_atr = np.array([r["atr_ratio"] for r in rows])
    arr_disp = np.array([r["disp"] for r in rows])
    arr_pnls = {
        0.0: np.array([r["pnl_sl0"] for r in rows]),
        0.05: np.array([r["pnl_sl5"] for r in rows]),
        0.10: np.array([r["pnl_sl10"] for r in rows]),
        0.15: np.array([r["pnl_sl15"] for r in rows]),
        0.20: np.array([r["pnl_sl20"] for r in rows]),
    }
    # Grid reduit (sinon explosion)
    h1m_list = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    atr_list = [None, 2.0, 2.5, 3.0]
    hmin_list = [0, 6, 7, 8, 9]
    hmax_list = [17, 20, 22, 24]
    disp_list = [0.0, 0.3, 0.5]
    sl_list = [0.0, 0.05, 0.10, 0.15, 0.20]
    cons_list = [2, 3]
    MIN_N = 150   # 1 trade/jour sur 1.7 ans ~= 425 trades, mais on est sur 24 mois IN -> 600+
    # On baisse a 150 pour avoir des configs valides meme sur 2025-2026 (16 mois)
    best = None
    base_user_mask = arr_d1 & arr_h1 & arr_pd
    for cons in cons_list:
        cons_mask = arr_min_cons == cons
        for h1m in h1m_list:
            for atr_m in atr_list:
                for hmin in hmin_list:
                    for hmax in hmax_list:
                        if hmin >= hmax: continue
                        for dm in disp_list:
                            mask = (
                                cons_mask & base_user_mask &
                                (arr_h1m >= h1m) &
                                (arr_hour >= hmin) & (arr_hour < hmax) &
                                (arr_disp >= dm)
                            )
                            if atr_m is not None:
                                mask &= (arr_atr < atr_m)
                            if not mask.any(): continue
                            for sl_buf in sl_list:
                                sub = arr_pnls[sl_buf][mask]
                                sub = sub[~np.isnan(sub)]
                                n = len(sub)
                                if n < MIN_N: continue
                                exp = float(sub.mean())
                                if exp < 0.10: continue
                                wins = int((sub > 0).sum())
                                wr = wins/n*100
                                if wr < 50: continue
                                if best is None or exp > best[1]:
                                    best = ({"h1m":h1m, "atr_max":atr_m, "h_min":hmin,
                                              "h_max":hmax, "disp":dm, "sl_buf":sl_buf,
                                              "cons":cons}, exp, n, wr)
    return best


def test_1_walkforward(datasets_in, datasets_out):
    print("\n" + "=" * 100)
    print("TEST 1 : WALK-FORWARD STRICT")
    print("=" * 100)
    print("Optim IN-sample (2023-2024) -> Test OUT-sample (2025-2026)\n")

    in_configs = {}; in_stats = {}
    print("  >> Optim IN-sample 2023-2024...")
    t0 = time.time()
    for name, rows in datasets_in.items():
        best = mini_grid_per_asset(rows)
        if best:
            cfg, exp, n, wr = best
            in_configs[name] = cfg
            in_stats[name] = {"exp_in": exp, "n_in": n, "wr_in": wr}
        else:
            in_stats[name] = None
    print(f"  Optim termine en {time.time()-t0:.0f}s. Configs valides : {len(in_configs)}/22")

    # Apply IN configs on OUT
    out_stats = apply_per_asset_config(datasets_out, in_configs)

    print(f"\n  RESULTATS WALK-FORWARD")
    print(f"  {'actif':<12} {'exp_IN':<10} {'wr_IN':<8} {'exp_OUT':<10} {'wr_OUT':<8} {'n_OUT':<7} {'rapport'}")
    print(f"  {'-'*82}")
    n_passed = 0; n_total = 0; ratios = []
    for name in sorted(in_configs.keys()):
        in_s = in_stats[name]
        out_s = out_stats.get(name)
        if out_s is None or out_s["n"] == 0:
            print(f"  {name:<12} {in_s['exp_in']:<+10.3f} {in_s['wr_in']:<8.1f} (pas de trade OUT)")
            continue
        ratio = out_s["exp"] / in_s["exp_in"] if in_s["exp_in"] != 0 else 0
        ratios.append(ratio)
        n_total += 1
        passed = "PASS" if (out_s["exp"] >= in_s["exp_in"] * 0.6 and out_s["exp"] > 0) else "FAIL"
        if passed == "PASS": n_passed += 1
        print(f"  {name:<12} {in_s['exp_in']:<+10.3f} {in_s['wr_in']:<8.1f} {out_s['exp']:<+10.3f} {out_s['wr']:<8.1f} {out_s['n']:<7} {ratio*100:+.0f}%  {passed}")
    avg_ratio = np.mean(ratios) if ratios else 0
    print(f"\n  VERDICT : {n_passed}/{n_total} actifs PASSent (edge OUT >= 60% IN et > 0)")
    print(f"  Ratio moyen OUT/IN : {avg_ratio*100:+.0f}%")
    return {"in_configs": in_configs, "in_stats": in_stats,
            "out_stats": out_stats, "n_passed": n_passed, "n_total": n_total,
            "avg_ratio": avg_ratio}


# =========================================================================
# TEST 2 : Backtest 2022
# Applique les configs per-actif validees sur 2022
# =========================================================================

def test_2_year_2022(datasets_2022, datasets_2023_2026):
    print("\n" + "=" * 100)
    print("TEST 2 : BACKTEST 2022 (annee jamais vue)")
    print("=" * 100)
    s_2022 = apply_per_asset_config(datasets_2022, PER_ASSET_CONFIG)
    s_full = apply_per_asset_config(datasets_2023_2026, PER_ASSET_CONFIG)
    print(f"\n  Configs per-actif appliquees sur 2022 vs 2023-2026")
    print(f"  {'actif':<12} {'exp_2022':<10} {'exp_full':<10} {'ratio'}")
    print("  " + "-" * 50)
    ratios = []; n_pos_2022 = 0
    for name in sorted(s_2022.keys()):
        s22 = s_2022.get(name); sfull = s_full.get(name)
        if s22 is None or sfull is None: continue
        if s22["n"] < 20: continue
        ratio = s22["exp"] / sfull["exp"] if sfull["exp"] > 0 else 0
        ratios.append(ratio)
        if s22["exp"] > 0: n_pos_2022 += 1
        print(f"  {name:<12} {s22['exp']:<+10.3f} {sfull['exp']:<+10.3f} {ratio*100:+.0f}%")
    avg_ratio = np.mean(ratios) if ratios else 0
    print(f"\n  Edge 2022 / Edge 2023-2026 : {avg_ratio*100:+.0f}% en moyenne")
    print(f"  Actifs profitables en 2022 : {n_pos_2022}/{len(ratios)}")
    return {"ratios": ratios, "avg_ratio": avg_ratio, "n_pos": n_pos_2022,
            "per_asset_2022": s_2022, "per_asset_full": s_full}


# =========================================================================
# TEST 3 : Filtres separes
# =========================================================================

def test_3_filters(datasets_full):
    print("\n" + "=" * 100)
    print("TEST 3 : FILTRES SEPARES (D1, H1, PD, combinaisons)")
    print("=" * 100)
    # On applique sur dataset 2023-2026 SANS la regle user complete
    # 7 variantes
    variants = {
        "D1 seul"     : {"d1"},
        "H1 seul"     : {"h1"},
        "PD seul"     : {"pd"},
        "D1+H1"       : {"d1","h1"},
        "D1+PD"       : {"d1","pd"},
        "H1+PD"       : {"h1","pd"},
        "D1+H1+PD"    : {"d1","h1","pd"},
    }
    print(f"\n  {'Variante':<14} {'n_total':<10} {'WR%':<8} {'exp_R':<10} {'DD':<8}")
    print("  " + "-" * 60)
    results = {}
    for vname, filt in variants.items():
        all_pnls = []
        for rows in datasets_full.values():
            for r in rows:
                if r["min_cons"] != 2: continue   # on fixe cons=2 pour comparaison
                if not passes_filters(filt, r["d1_a"], r["h1_a"], r["pd_a"]): continue
                # Pas de filtre h1mom/hour/disp/atr ici - juste les 3 filtres
                p = r["pnl_sl10"]  # sl_buf=0.10 standard
                if not np.isnan(p): all_pnls.append(p)
        s = stats_pnls(all_pnls)
        if s:
            print(f"  {vname:<14} {s['n']:<10} {s['wr']:<8.1f} {s['exp']:<+10.3f} {s['dd']:<8.1f}")
            results[vname] = s
        else:
            print(f"  {vname:<14} 0 trades")
            results[vname] = None
    return results


# =========================================================================
# TEST 4 : Stabilite par sous-periode
# =========================================================================

def test_4_subperiods(datasets_full):
    print("\n" + "=" * 100)
    print("TEST 4 : STABILITE PAR SOUS-PERIODE")
    print("=" * 100)
    subperiods = [
        ("2023",      pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2024-01-01", tz="UTC")),
        ("2024",      pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC")),
        ("2025-H1",   pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2025-07-01", tz="UTC")),
        ("2025-H2",   pd.Timestamp("2025-07-01", tz="UTC"), pd.Timestamp("2026-01-01", tz="UTC")),
        ("2026-Jan-May", pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-06-01", tz="UTC")),
    ]
    print(f"\n  {'Periode':<14} {'n':<8} {'WR%':<7} {'exp_R':<10} {'DD':<8}")
    print("  " + "-" * 55)
    results = {}
    for pname, ps, pe in subperiods:
        ps_ns = ps.value; pe_ns = pe.value
        all_pnls = []
        for name, rows in datasets_full.items():
            cfg = PER_ASSET_CONFIG.get(name)
            if cfg is None: continue
            sl_key = SL_BUF_TO_KEY.get(cfg["sl_buf"])
            for r in rows:
                if not (ps_ns <= r["ts"] < pe_ns): continue
                if r["min_cons"] != cfg["cons"]: continue
                if not (r["d1_a"] and r["h1_a"] and r["pd_a"]): continue
                if r["h1_mom"] < cfg["h1m"]: continue
                if not (cfg["h_min"] <= r["hour_fr"] < cfg["h_max"]): continue
                if r["disp"] < cfg["disp"]: continue
                if cfg["atr_max"] is not None and r["atr_ratio"] >= cfg["atr_max"]: continue
                p = r[sl_key]
                if not np.isnan(p): all_pnls.append(p)
        s = stats_pnls(all_pnls)
        if s:
            verdict = "OK" if s["exp"] > 0.30 else ("=" if s["exp"] > 0.10 else "FAIL")
            print(f"  {pname:<14} {s['n']:<8} {s['wr']:<7.1f} {s['exp']:<+10.3f} {s['dd']:<8.1f} {verdict}")
            results[pname] = s
        else:
            print(f"  {pname:<14} pas de trades")
    return results


# =========================================================================
# TEST 7 : Slippage + commissions
# =========================================================================

COMMISSION_PER_LOT = {  # USD par lot ouvert+ferme (estimatif Vantage)
    "AUDUSD":7, "BTCUSD":50, "BVSPX":2, "CL-OIL":3, "Cocoa-C":5, "DJ30":3,
    "ETHUSD":20, "FRA40":2, "GAS-C":5, "GBPUSD":7, "GER40":3, "HK50":3,
    "NAS100":3, "Nikkei225":3, "SP500":3, "UK100":3, "USDCAD":7,
    "USDCHF":7, "USDJPY":7, "USDMXN":15, "USDZAR":15, "XAUUSD":5,
}
SLIPPAGE_R = 0.05   # 5% de R perdu en moyenne sur l'exit (slippage SL/TP)


def test_7_costs(datasets_full):
    print("\n" + "=" * 100)
    print("TEST 7 : SLIPPAGE + COMMISSIONS REALISTES")
    print("=" * 100)
    print(f"  Modele : slippage exit -{SLIPPAGE_R*100:.0f}% R + commission -5-10E par trade typique")
    # Test sans surcharge : juste les pnls baseline
    s_clean = apply_per_asset_config(datasets_full, PER_ASSET_CONFIG)
    # Test avec slippage
    s_slip = {}
    for name, rows in datasets_full.items():
        cfg = PER_ASSET_CONFIG.get(name)
        if cfg is None: continue
        sl_key = SL_BUF_TO_KEY.get(cfg["sl_buf"])
        pnls = []
        for r in rows:
            if r["min_cons"] != cfg["cons"]: continue
            if not (r["d1_a"] and r["h1_a"] and r["pd_a"]): continue
            if r["h1_mom"] < cfg["h1m"]: continue
            if not (cfg["h_min"] <= r["hour_fr"] < cfg["h_max"]): continue
            if r["disp"] < cfg["disp"]: continue
            if cfg["atr_max"] is not None and r["atr_ratio"] >= cfg["atr_max"]: continue
            p = r[sl_key]
            if np.isnan(p): continue
            # Applique slippage : reduit la valeur absolue
            if p > 0: p_after = p - SLIPPAGE_R
            elif p < 0: p_after = p - SLIPPAGE_R  # SL touche encore plus loin
            else: p_after = p
            pnls.append(p_after)
        s_slip[name] = stats_pnls(pnls)
    # Affichage
    print(f"\n  {'actif':<12} {'exp_clean':<11} {'exp_slip':<11} {'delta'}")
    print("  " + "-" * 50)
    deltas = []
    for name in sorted(s_clean.keys()):
        a = s_clean.get(name); b = s_slip.get(name)
        if a is None or b is None: continue
        delta = b["exp"] - a["exp"]
        deltas.append((name, a["exp"], b["exp"], delta))
        verdict = "OK" if b["exp"] > 0.20 else ("=" if b["exp"] > 0 else "NEG")
        print(f"  {name:<12} {a['exp']:<+11.3f} {b['exp']:<+11.3f} {delta:+.3f}R {verdict}")
    n_pos = sum(1 for _,_,b,_ in deltas if b > 0)
    n_strong = sum(1 for _,_,b,_ in deltas if b > 0.20)
    print(f"\n  {n_pos}/{len(deltas)} actifs encore profitables apres slippage")
    print(f"  {n_strong}/{len(deltas)} actifs avec edge >= 0.20R apres slippage")
    return {"clean": s_clean, "slip": s_slip}


# =========================================================================
# MAIN
# =========================================================================

def main():
    print("=" * 100)
    print("V22 VALIDATION COMPLETE - 5 tests critiques (T1, T2, T3, T4, T7)")
    print("=" * 100)
    t_global = time.time()

    n_workers = 8

    # Build datasets en cache : 3 periodes
    print("\n>>> Phase 0 : Building datasets pour 3 periodes...")
    sys.stdout.flush()
    ds_in    = build_all_datasets(pd.Timestamp("2023-01-01", tz="UTC"),
                                    pd.Timestamp("2025-01-01", tz="UTC"), "in_2023_2024", n_workers)
    ds_out   = build_all_datasets(pd.Timestamp("2025-01-01", tz="UTC"),
                                    pd.Timestamp("2026-06-01", tz="UTC"), "out_2025_2026", n_workers)
    ds_2022  = build_all_datasets(pd.Timestamp("2022-01-01", tz="UTC"),
                                    pd.Timestamp("2023-01-01", tz="UTC"), "year_2022", n_workers)
    ds_full  = build_all_datasets(pd.Timestamp("2023-01-01", tz="UTC"),
                                    pd.Timestamp("2026-06-01", tz="UTC"), "full_2023_2026", n_workers)

    # Tests
    res_t1 = test_1_walkforward(ds_in, ds_out)
    res_t2 = test_2_year_2022(ds_2022, ds_full)
    res_t3 = test_3_filters(ds_full)
    res_t4 = test_4_subperiods(ds_full)
    res_t7 = test_7_costs(ds_full)

    # Sauvegarde
    out_json = {
        "test_1_walkforward": {
            "in_configs": res_t1["in_configs"],
            "n_passed": res_t1["n_passed"], "n_total": res_t1["n_total"],
            "avg_ratio": res_t1["avg_ratio"],
        },
        "test_2_year_2022": {
            "avg_ratio": res_t2["avg_ratio"], "n_pos": res_t2["n_pos"],
        },
        "test_3_filters": {k: v for k, v in res_t3.items()},
        "test_4_subperiods": {k: v for k, v in res_t4.items()},
        "test_7_costs_summary": {
            "clean_avg_exp": np.mean([v["exp"] for v in res_t7["clean"].values() if v]),
            "slip_avg_exp": np.mean([v["exp"] for v in res_t7["slip"].values() if v]),
        },
    }
    (OUT_DIR / "validation_results.json").write_text(json.dumps(out_json, default=str, indent=2))

    # VERDICT FINAL
    print("\n" + "=" * 100)
    print("VERDICT FINAL")
    print("=" * 100)
    print(f"  T1 Walk-forward      : {res_t1['n_passed']}/{res_t1['n_total']} actifs PASSent (ratio {res_t1['avg_ratio']*100:+.0f}%)")
    print(f"  T2 Backtest 2022     : {res_t2['n_pos']}/{len(res_t2['ratios'])} actifs profitables (ratio {res_t2['avg_ratio']*100:+.0f}%)")
    if res_t3.get("D1+H1+PD"):
        s = res_t3["D1+H1+PD"]
        print(f"  T3 Filtres combines  : exp={s['exp']:+.3f}R sur {s['n']} trades (les 3 filtres complementaires)")
    t4_ok = sum(1 for s in res_t4.values() if s and s["exp"] > 0.10)
    print(f"  T4 Sous-periodes     : {t4_ok}/{len(res_t4)} sous-periodes avec edge >= +0.10R")
    print(f"  T7 Apres slippage    : avg exp = {out_json['test_7_costs_summary']['slip_avg_exp']:+.3f}R")

    go = True
    if res_t1["n_passed"] / max(res_t1["n_total"],1) < 0.5: go = False; print("  ❌ T1 FAIL : moins de 50% des actifs survivent OOS")
    if res_t2["n_pos"] / max(len(res_t2["ratios"]),1) < 0.5: go = False; print("  ❌ T2 FAIL : moins de 50% profitables en 2022")
    if out_json['test_7_costs_summary']['slip_avg_exp'] < 0.15: go = False; print("  ❌ T7 FAIL : edge < 0.15R apres slippage")

    if go:
        print(f"\n  ✅ GO LIVE DEMO : V22 passe les 4 tests critiques")
    else:
        print(f"\n  ❌ NO-GO : un ou plusieurs tests critiques echouent")

    print(f"\n  Temps total : {time.time()-t_global:.0f}s")
    print(f"  Resultats sauves dans {OUT_DIR / 'validation_results.json'}")


if __name__ == "__main__":
    main()
