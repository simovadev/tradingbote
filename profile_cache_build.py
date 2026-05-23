"""Profile les 6 fonctions du cache_build pour identifier le coupable des 5.2s.

Reproduit EXACTEMENT le bloc cache_build de compute_asset (live_runner_v2.py:429-443),
sur des bougies reelles parquet, avec les memes parametres que le live (sws=2 par defaut).

Mesure chaque fonction individuellement + le total. Tourne sur 14 actifs.
"""
from __future__ import annotations
import os
import sys
import time
import statistics as st

os.environ.setdefault("BUILD_DATA_DIR", "data_vantage")
sys.path.insert(0, "c:/Users/Shadow/TradingBot")

import pandas as pd

from bot_v2.concepts.order_block import detect_order_blocks
from bot_v2.concepts.liquidity import find_swings
from bot_v2.concepts.fvg import detect_fvg
from bot_v2.concepts.breaker import detect_breakers
from bot_v2.concepts.structure import detect_structure_breaks, detect_trend
from bot_v2.config import get_param

LIVE_ASSETS = [
    "XAUUSD", "NAS100", "GER40", "BTCUSD",
    "EURUSD", "GBPUSD", "AUDUSD", "USDJPY",
    "SP500", "DJ30", "UK100", "FRA40",
    "USDCAD", "USDCHF",
]

DATA_DIR = "c:/Users/Shadow/TradingBot/data_vantage"
N_BARS_M1 = 88000
N_BARS_M15 = 11000
N_BARS_H1 = 2800


def load_bars(asset: str):
    df_m1 = pd.read_parquet(f"{DATA_DIR}/{asset}_M1.parquet").tail(N_BARS_M1)
    df_m15 = pd.read_parquet(f"{DATA_DIR}/{asset}_M15.parquet").tail(N_BARS_M15)
    df_h1 = pd.read_parquet(f"{DATA_DIR}/{asset}_H1.parquet").tail(N_BARS_H1)
    return df_m1, df_m15, df_h1


def profile_one(asset: str) -> dict:
    df_m1, df_m15, df_h1 = load_bars(asset)
    sws = get_param(asset, "swing_strength_m1", 2)

    timings = {}

    t = time.perf_counter()
    obs = detect_order_blocks(df_m1, swing_strength=sws)
    timings["detect_order_blocks_M1"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    swings = find_swings(df_m1, strength=sws)
    timings["find_swings_M1"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    fvgs = detect_fvg(df_m1)
    timings["detect_fvg_M1"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    breakers = detect_breakers(df_m1)
    timings["detect_breakers_M1"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    obs_htf = detect_order_blocks(df_m15)
    timings["detect_order_blocks_M15"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    sb = detect_structure_breaks(df_m1, swings=swings, fvgs=fvgs)
    timings["detect_structure_breaks_M1"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    trend = detect_trend(swings, lookback=6)
    timings["detect_trend"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    obs_htf2 = detect_order_blocks(df_h1)
    timings["detect_order_blocks_H1"] = (time.perf_counter() - t) * 1000

    timings["_TOTAL_cache_build"] = sum(v for k, v in timings.items() if not k.startswith("_"))
    timings["_n_obs"] = len(obs)
    timings["_n_swings"] = len(swings)
    timings["_n_fvgs"] = len(fvgs)
    timings["_n_breakers"] = len(breakers)
    timings["_n_sb"] = len(sb)
    timings["_n_m1"] = len(df_m1)
    return timings


def main():
    print(f"{'ASSET':<8} {'TOTAL':>7}  {'OB_M1':>7} {'OB_M15':>7} {'OB_H1':>7}  {'SWNG':>6} {'FVG':>6} {'BRK':>6} {'STR':>6} {'TRND':>5}   counts")
    print("-" * 130)

    rows = []
    for a in LIVE_ASSETS:
        try:
            r = profile_one(a)
            rows.append((a, r))
            print(f"{a:<8} {r['_TOTAL_cache_build']:>7.0f}ms"
                  f"  {r['detect_order_blocks_M1']:>5.0f}ms"
                  f" {r['detect_order_blocks_M15']:>5.0f}ms"
                  f" {r['detect_order_blocks_H1']:>5.0f}ms"
                  f"  {r['find_swings_M1']:>4.0f}ms"
                  f" {r['detect_fvg_M1']:>4.0f}ms"
                  f" {r['detect_breakers_M1']:>4.0f}ms"
                  f" {r['detect_structure_breaks_M1']:>4.0f}ms"
                  f" {r['detect_trend']:>3.1f}ms"
                  f"   OB={r['_n_obs']} SW={r['_n_swings']} FVG={r['_n_fvgs']} BR={r['_n_breakers']} SB={r['_n_sb']} m1={r['_n_m1']}")
        except Exception as e:
            print(f"{a:<8} ERR {e}")

    if rows:
        print("\n=== MEDIANES SUR 14 ACTIFS ===")
        keys = ["detect_order_blocks_M1", "detect_order_blocks_M15", "detect_order_blocks_H1",
                "find_swings_M1", "detect_fvg_M1", "detect_breakers_M1",
                "detect_structure_breaks_M1", "detect_trend", "_TOTAL_cache_build"]
        for k in keys:
            vals = [r[1][k] for r in rows]
            label = k.replace("_M1", " M1").replace("_M15", " M15").replace("_H1", " H1")
            print(f"  {label:<32} median={st.median(vals):>7.0f}ms  mean={st.mean(vals):>7.0f}ms  max={max(vals):>7.0f}ms")

        total_median = st.median([r[1]["_TOTAL_cache_build"] for r in rows])
        print(f"\n=== TOP COUPABLES (% du TOTAL median = {total_median:.0f}ms) ===")
        funcs = [k for k in keys if k != "_TOTAL_cache_build"]
        pcts = [(k, st.median([r[1][k] for r in rows])) for k in funcs]
        pcts.sort(key=lambda x: -x[1])
        for k, v in pcts:
            pct = v / total_median * 100
            print(f"  {k:<32} {v:>6.0f}ms  ({pct:>5.1f}%)")


if __name__ == "__main__":
    main()
