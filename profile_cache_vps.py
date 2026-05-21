"""Profile EXACT du cache_build de scan_asset() — a lancer sur le VPS.

Reproduit ligne par ligne les 6 operations lourdes de cache_build
(live_runner_v2.py lignes 294-310) + detect_mss_setups, en lisant
data_vantage/. Decompose chaque appel pour voir lequel coute le plus.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from bot_v2.concepts.liquidity import find_swings
from bot_v2.concepts.fvg import detect_fvg
from bot_v2.concepts.breaker import detect_breakers
from bot_v2.concepts.order_block import detect_order_blocks
from bot_v2.concepts.structure import detect_structure_breaks, detect_trend
from bot_v2.concepts.mss_setup import detect_mss_setups

DATA = Path(__file__).parent / "data_vantage"

N_BARS_M1 = 88000
N_BARS_M15 = 11000
N_BARS_H1 = 2800


def resample(df_m1, rule):
    return df_m1.resample(rule, label="left", closed="left").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna()


def profile(asset):
    pq = DATA / f"{asset}_M1.parquet"
    if not pq.exists():
        print(f"  {asset}: pas de parquet")
        return
    df_full = pd.read_parquet(pq)
    if df_full.index.tz is None:
        df_full.index = df_full.index.tz_localize("UTC")

    df_m1 = df_full.tail(N_BARS_M1).iloc[:-1]
    df_m15 = resample(df_full, "15min").tail(N_BARS_M15)
    df_h1 = resample(df_full, "1h").tail(N_BARS_H1)
    print(f"\n=== {asset} (M1={len(df_m1)}, M15={len(df_m15)}, H1={len(df_h1)}) ===")

    t = {}
    g0 = time.time()

    _t = time.time()
    obs = detect_order_blocks(df_m1, swing_strength=2, max_group_size=2)
    t["1.detect_order_blocks(M1)"] = time.time() - _t

    _t = time.time()
    swings_ltf = find_swings(df_m1, strength=2)
    t["2.find_swings(M1)"] = time.time() - _t

    _t = time.time()
    fvgs_ltf = detect_fvg(df_m1)
    t["3.detect_fvg(M1)"] = time.time() - _t

    _t = time.time()
    breakers_ltf = detect_breakers(df_m1)
    t["4.detect_breakers(M1)"] = time.time() - _t

    _t = time.time()
    obs_htf = detect_order_blocks(df_m15)
    t["5.detect_order_blocks(M15)"] = time.time() - _t

    _t = time.time()
    structure_breaks = detect_structure_breaks(df_m1, swings=swings_ltf, fvgs=fvgs_ltf)
    t["6.detect_structure_breaks(M1)"] = time.time() - _t

    _t = time.time()
    obs_htf2 = detect_order_blocks(df_h1)
    t["7.detect_order_blocks(H1)"] = time.time() - _t

    cache_build_total = time.time() - g0

    _t = time.time()
    mss = detect_mss_setups(df_m1, structure_breaks=structure_breaks,
                            swings=swings_ltf, fvgs=fvgs_ltf)
    t["8.detect_mss_setups(M1)"] = time.time() - _t

    for k, v in sorted(t.items()):
        flag = "  <<< GOULOT" if v > 2.0 else ""
        print(f"  {k:<35} {v:6.2f}s{flag}")
    print(f"  {'--> cache_build TOTAL (1-7)':<35} {cache_build_total:6.2f}s")
    print(f"  {'--> + mss_setups':<35} {cache_build_total + t['8.detect_mss_setups(M1)']:6.2f}s")


if __name__ == "__main__":
    assets = sys.argv[1:] or ["XAUUSD", "NAS100"]
    print(f"Profile cache_build EXACT sur {len(assets)} actif(s)")
    for a in assets:
        profile(a)
