"""v22_backtest_brut.py - Backtest ICT BRUT propre (sans ML).

But : voir si l'algorithme ICT corrige (concepts/safe.py) a un edge BRUT,
sans aucun ML, juste avec les filtres ICT classiques.

On teste plusieurs configurations en parallele :
- baseline : tous les OBs detectes
- + killzone (NY AM, London uniquement)
- + macro (dans une macro window uniquement)
- + daily_bias aligne (D1)
- + H1 trend aligne
- + tous les filtres combines

Et pour chaque config, on teste differents plans trade :
- SL 1 ATR / TP 2R
- SL 1 ATR / TP 3R
- SL 0.5 ATR / TP 2R

Verdict : on cherche une combinaison qui donne WR + RR > breakeven
sur plusieurs actifs simultanement.
"""
from __future__ import annotations
import sys
from pathlib import Path
from itertools import product

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

from bot_v2.concepts.safe import (
    find_obs_safe, daily_bias_safe, htf_value_at_t, atr_safe,
)
from bot_screener.time_utils import killzone_name, current_macro

DATA = ROOT / "data_vantage"

# Periode
START_DATE = "2023-01-01"
END_DATE = "2026-04-01"

# Parametres OB
DISPLACEMENT_MIN = 1.5
SWING_LOOKBACK = 20
MAX_HOLD_BARS = 60   # 5h M5


def simulate_trade(df_m5, ob, atr, sl_mult, tp_rr):
    """Simule trade avec params donnes. Retourne 1 si TP touche AVANT SL, 0 sinon."""
    val_idx = ob.validation_index
    if val_idx + 1 + MAX_HOLD_BARS >= len(df_m5):
        return None

    entry = float(df_m5["close"].iloc[val_idx])
    if ob.direction == "bullish":
        sl = entry - sl_mult * atr
        tp = entry + tp_rr * sl_mult * atr
    else:
        sl = entry + sl_mult * atr
        tp = entry - tp_rr * sl_mult * atr

    if abs(entry - sl) < 1e-9:
        return None

    fh = df_m5["high"].values[val_idx + 1:val_idx + 1 + MAX_HOLD_BARS]
    fl = df_m5["low"].values[val_idx + 1:val_idx + 1 + MAX_HOLD_BARS]
    fo = df_m5["open"].values[val_idx + 1:val_idx + 1 + MAX_HOLD_BARS]

    for i in range(len(fh)):
        if ob.direction == "bullish":
            hit_sl = fl[i] <= sl; hit_tp = fh[i] >= tp
        else:
            hit_sl = fh[i] >= sl; hit_tp = fl[i] <= tp
        if hit_sl and hit_tp:
            # ambiguite : conservateur
            if abs(fo[i] - sl) < abs(fo[i] - tp): return 0
            return 1
        if hit_sl: return 0
        if hit_tp: return 1
    return 0


def get_filters(df_m5, df_h1, df_d1, ob):
    """Retourne un dict de booleans : quel filtre passe pour cet OB."""
    val_ts = ob.validation_ts
    val_idx = ob.validation_index

    filters = {}

    # Killzone
    kz = killzone_name(val_ts)
    filters["kz_ny"] = kz == "NY AM"
    filters["kz_london"] = kz == "London"
    filters["kz_main"] = kz in ("NY AM", "London", "NY PM", "Silver Bullet AM", "Silver Bullet London")

    # Macro
    macro = current_macro(val_ts)
    filters["macro"] = bool(macro)

    # Daily bias aligne
    d1 = daily_bias_safe(df_d1, val_ts)
    if d1["ok"]:
        d1_haussier = d1["bias"] == "haussier"
        filters["d1_aligned"] = (
            (ob.direction == "bullish" and d1_haussier) or
            (ob.direction == "bearish" and not d1_haussier)
        )
    else:
        filters["d1_aligned"] = False

    # H1 trend (close vs 10h plus tot)
    h1_close = htf_value_at_t(df_h1, val_ts, "close")
    h1_close_10ago = None
    if h1_close is not None:
        pos = df_h1.index.searchsorted(val_ts, side="left") - 1 - 10
        if pos >= 0:
            h1_close_10ago = float(df_h1["close"].iloc[pos])
    if h1_close and h1_close_10ago:
        h1_haussier = h1_close > h1_close_10ago
        filters["h1_aligned"] = (
            (ob.direction == "bullish" and h1_haussier) or
            (ob.direction == "bearish" and not h1_haussier)
        )
    else:
        filters["h1_aligned"] = False

    return filters


def run_asset(asset, sl_mults, tp_rrs):
    print(f"\n=== {asset} ===")
    try:
        df_m5 = pd.read_parquet(DATA / f"{asset}_M5.parquet")[["open", "high", "low", "close"]]
        df_h1 = pd.read_parquet(DATA / f"{asset}_H1.parquet")[["open", "high", "low", "close"]]
        df_d1 = pd.read_parquet(DATA / f"{asset}_D1.parquet")[["open", "high", "low", "close"]]
    except Exception as e:
        print(f"  ERREUR : {e}")
        return None

    start = pd.Timestamp(START_DATE, tz="UTC")
    end = pd.Timestamp(END_DATE, tz="UTC")
    context_start = start - pd.Timedelta(days=10)
    df_m5 = df_m5[(df_m5.index >= context_start) & (df_m5.index < end)]
    df_h1 = df_h1[(df_h1.index >= context_start) & (df_h1.index < end)]
    df_d1 = df_d1[(df_d1.index >= start - pd.Timedelta(days=30)) & (df_d1.index < end)]

    if len(df_m5) < 1000:
        print(f"  Pas assez de data M5 ({len(df_m5)})")
        return None

    # Detect tous les OBs sur la periode
    obs = find_obs_safe(
        df_m5, as_of_index=len(df_m5) - 1,
        sweep_lookback=SWING_LOOKBACK,
        displacement_min_atr=DISPLACEMENT_MIN,
    )
    obs = [ob for ob in obs if start <= ob.validation_ts < end]
    print(f"  OBs : {len(obs)}")

    # Pour chaque OB, on stocke ses filtres + son outcome a differents plans trade
    samples = []
    for ob in obs:
        atr = atr_safe(df_m5, ob.validation_index)
        if atr <= 0: continue
        filt = get_filters(df_m5, df_h1, df_d1, ob)
        outcomes = {}
        for sl_mult in sl_mults:
            for tp_rr in tp_rrs:
                outcomes[(sl_mult, tp_rr)] = simulate_trade(df_m5, ob, atr, sl_mult, tp_rr)
        samples.append({"filters": filt, "outcomes": outcomes})

    return samples


def evaluate_config(samples, filter_keys, sl_mult, tp_rr):
    """Pour une config de filtres + plan trade, retourne (n, WR, esperance R, etc.)."""
    selected = [s for s in samples if all(s["filters"].get(k, False) for k in filter_keys)]
    if len(selected) == 0:
        return None
    outcomes = [s["outcomes"][(sl_mult, tp_rr)] for s in selected if s["outcomes"][(sl_mult, tp_rr)] is not None]
    if len(outcomes) == 0:
        return None
    wins = sum(o == 1 for o in outcomes)
    losses = sum(o == 0 for o in outcomes)
    n = wins + losses
    if n == 0: return None
    wr = wins / n * 100
    # Esperance R : WIN = +tp_rr*R, LOSS = -1*R
    avg_r = (wins * tp_rr + losses * (-1)) / n
    return {"n": n, "wr": wr, "exp_r": avg_r, "tp_rr": tp_rr, "sl_mult": sl_mult}


def main():
    print(f"\n=== V22 BACKTEST BRUT (ICT corrige, sans ML) ===")
    print(f"Periode : {START_DATE} -> {END_DATE}")
    print(f"OB strict : displacement {DISPLACEMENT_MIN} ATR, sweep_lookback {SWING_LOOKBACK}, max_hold {MAX_HOLD_BARS} bougies")

    # Configurations a tester
    SL_MULTS = [0.5, 1.0, 1.5]
    TP_RRS = [2.0, 3.0, 5.0]

    # Combinaisons de filtres a tester
    FILTER_CONFIGS = [
        ("baseline", []),
        ("kz_main", ["kz_main"]),
        ("kz_ny", ["kz_ny"]),
        ("macro", ["macro"]),
        ("d1_aligned", ["d1_aligned"]),
        ("h1_aligned", ["h1_aligned"]),
        ("kz_main + d1", ["kz_main", "d1_aligned"]),
        ("kz_main + d1 + h1", ["kz_main", "d1_aligned", "h1_aligned"]),
        ("kz_ny + d1 + h1", ["kz_ny", "d1_aligned", "h1_aligned"]),
        ("macro + d1", ["macro", "d1_aligned"]),
        ("all combined", ["kz_main", "d1_aligned", "h1_aligned"]),
    ]

    ASSETS = ["XAUUSD", "NAS100", "SP500", "GER40", "EURUSD", "BTCUSD"]

    # Run par actif
    all_samples = {}
    for asset in ASSETS:
        samples = run_asset(asset, SL_MULTS, TP_RRS)
        if samples is not None:
            all_samples[asset] = samples

    # Synthese
    print("\n" + "=" * 110)
    print("=== SYNTHESE GLOBAL : config x plan trade ===")
    print("=" * 110)

    # Pour chaque combo (filter, sl, tp), on agrege sur tous les actifs
    best_configs = []
    for cfg_name, filter_keys in FILTER_CONFIGS:
        for sl_mult in SL_MULTS:
            for tp_rr in TP_RRS:
                total_n = 0; total_wins = 0; total_losses = 0
                per_asset = {}
                for asset, samples in all_samples.items():
                    r = evaluate_config(samples, filter_keys, sl_mult, tp_rr)
                    if r:
                        per_asset[asset] = r
                        total_n += r["n"]
                        wins_a = int(r["n"] * r["wr"] / 100)
                        total_wins += wins_a
                        total_losses += r["n"] - wins_a
                if total_n < 30:
                    continue
                wr = total_wins / total_n * 100 if total_n > 0 else 0
                exp_r = (total_wins * tp_rr + total_losses * (-1)) / total_n
                breakeven_wr = 100 / (tp_rr + 1)
                # n actifs profitables
                pos_assets = sum(1 for r in per_asset.values() if r["exp_r"] > 0)
                best_configs.append({
                    "config": cfg_name, "sl_mult": sl_mult, "tp_rr": tp_rr,
                    "n": total_n, "wr": wr, "exp_r": exp_r, "breakeven_wr": breakeven_wr,
                    "n_assets": len(per_asset), "pos_assets": pos_assets,
                    "per_asset": per_asset,
                })

    # Sort par exp_r descendant
    best_configs.sort(key=lambda x: x["exp_r"], reverse=True)
    print(f"\n{'config':<25} {'sl':<5} {'tp':<5} {'n':<6} {'wr':<6} {'BE_wr':<7} {'exp_R':<8} {'+actifs'}")
    print("-" * 90)
    for c in best_configs[:25]:
        edge = "EDGE" if c["exp_r"] > 0 else ""
        print(f"{c['config']:<25} {c['sl_mult']:<5} {c['tp_rr']:<5} {c['n']:<6} {c['wr']:<6.1f} {c['breakeven_wr']:<7.1f} {c['exp_r']:<+8.3f} {c['pos_assets']}/{c['n_assets']} {edge}")

    print("\n\n=== TOP 3 CONFIGS DETAIL PAR ACTIF ===")
    for c in best_configs[:3]:
        print(f"\n--- {c['config']} | sl={c['sl_mult']}ATR tp={c['tp_rr']}R | exp_R={c['exp_r']:+.3f}R ---")
        for asset, r in c["per_asset"].items():
            edge = "+" if r["exp_r"] > 0 else "-"
            print(f"  {asset:<10} n={r['n']:<5} WR={r['wr']:<5.1f}% exp_R={r['exp_r']:+.3f}R {edge}")


if __name__ == "__main__":
    main()
