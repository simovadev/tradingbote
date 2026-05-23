"""Test d'equivalence STRICTE : cache via daemon == cache calcule directement.

Logique :
1. Lance le daemon une fois en mode --once sur 5 actifs -> ecrit data_cache/*.pkl
2. Pour chaque actif :
   a. Lit le pickle (= ce que verra le live)
   b. Calcule directement le cache (= comportement actuel)
   c. Compare strictement : obs, swings_ltf, fvgs_ltf, breakers_ltf, obs_htf,
      structure_breaks, htf_trend, obs_htf2 doivent etre identiques.

Si TOUT est identique -> daemon SAFE, on peut deployer.
Si quoi que ce soit differe -> bug, on ne deploie pas.
"""
from __future__ import annotations
import os
import sys
import pickle
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("BUILD_DATA_DIR", "data_vantage")
sys.path.insert(0, "c:/Users/Shadow/TradingBot")

import pandas as pd

from bot_v2.concepts.order_block import detect_order_blocks
from bot_v2.concepts.liquidity import find_swings
from bot_v2.concepts.fvg import detect_fvg
from bot_v2.concepts.breaker import detect_breakers
from bot_v2.concepts.structure import detect_structure_breaks, detect_trend
from bot_v2.config import get_param
from bot_v2.mt5_executor import MT5Executor

N_BARS_M1 = 88000
N_BARS_M15 = 11000
N_BARS_H1 = 2800
CACHE_DIR = Path("c:/Users/Shadow/TradingBot/data_cache")


def make_signature(item):
    """Cle hashable d'un dataclass (asdict + tri)."""
    d = asdict(item)
    return tuple(sorted((k, (v if not isinstance(v, dict) else tuple(sorted(v.items())))) for k, v in d.items()))


def compare_list(name, a, b):
    if len(a) != len(b):
        return False, f"len differ: {len(a)} vs {len(b)}"
    # Compare element par element en ordre
    for i, (x, y) in enumerate(zip(a, b)):
        try:
            if make_signature(x) != make_signature(y):
                return False, f"item {i} differ:\n  daemon: {x}\n  direct: {y}"
        except TypeError:
            # Fallback : compare via repr (si dataclass non hashable)
            if repr(x) != repr(y):
                return False, f"item {i} differ (repr):\n  daemon: {x}\n  direct: {y}"
    return True, "OK"


def calc_cache_direct(asset: str, mt5_exec) -> dict:
    df_m1 = mt5_exec.get_bars(asset, "M1", N_BARS_M1, force_sync=True).iloc[:-1]
    df_m15 = mt5_exec.get_bars(asset, "M15", N_BARS_M15).iloc[:-1]
    df_h1 = mt5_exec.get_bars(asset, "H1", N_BARS_H1).iloc[:-1]
    sws = get_param(asset, "swing_strength_m1", 2)
    return {
        "obs": detect_order_blocks(df_m1, swing_strength=sws),
        "swings_ltf": find_swings(df_m1, strength=sws),
        "fvgs_ltf": detect_fvg(df_m1),
        "breakers_ltf": detect_breakers(df_m1),
        "obs_htf": detect_order_blocks(df_m15),
        "structure_breaks": detect_structure_breaks(
            df_m1, swings=find_swings(df_m1, strength=sws), fvgs=detect_fvg(df_m1)
        ),
        "htf_trend": detect_trend(find_swings(df_m1, strength=sws), lookback=6),
        "obs_htf2": detect_order_blocks(df_h1),
        "df_m1_last_ts": df_m1.index[-1],
    }


def main():
    asset = sys.argv[1] if len(sys.argv) > 1 else "XAUUSD"
    print(f"=== Test equivalence daemon vs direct : {asset} ===\n")

    pkl_path = CACHE_DIR / f"{asset}.pkl"
    if not pkl_path.exists():
        print(f"FAIL: {pkl_path} introuvable. Lance d'abord :")
        print(f"   python cache_daemon.py --assets {asset} --once")
        return

    print(f"1. Lecture du pickle daemon : {pkl_path}")
    with open(pkl_path, "rb") as f:
        from_daemon = pickle.load(f)
    print(f"   daemon last_ts = {from_daemon['df_m1_last_ts']}")

    print(f"\n2. Calcul direct (= comportement live actuel)")
    mt5_exec = MT5Executor()
    if not mt5_exec.initialize():
        print("MT5 init FAIL"); return
    from_direct = calc_cache_direct(asset, mt5_exec)
    mt5_exec.shutdown()
    print(f"   direct last_ts = {from_direct['df_m1_last_ts']}")

    if from_daemon["df_m1_last_ts"] != from_direct["df_m1_last_ts"]:
        print(f"\nATTENTION : last_ts different (daemon plus vieux), une nouvelle bougie est arrivee depuis.")
        print("Le test peut quand meme passer si le daemon a fini avant l'arrivee de la nouvelle bougie.")
        print("Re-lance le daemon puis ce test rapidement pour avoir le meme last_ts.")
        return

    print("\n3. Comparaison element par element :")
    keys = ["obs", "swings_ltf", "fvgs_ltf", "breakers_ltf", "obs_htf",
            "structure_breaks", "obs_htf2"]
    all_ok = True
    for k in keys:
        ok, msg = compare_list(k, from_daemon[k], from_direct[k])
        if ok:
            print(f"   {k:<20} OK ({len(from_daemon[k])} items)")
        else:
            print(f"   {k:<20} FAIL : {msg[:200]}")
            all_ok = False

    # htf_trend est un scalaire (Direction|None)
    if from_daemon["htf_trend"] == from_direct["htf_trend"]:
        print(f"   htf_trend            OK ({from_daemon['htf_trend']})")
    else:
        print(f"   htf_trend            FAIL : daemon={from_daemon['htf_trend']} direct={from_direct['htf_trend']}")
        all_ok = False

    print()
    if all_ok:
        print("=== TOUT IDENTIQUE -> daemon SAFE ===")
    else:
        print("=== DIVERGENCE DETECTEE -> NE PAS DEPLOYER ===")


if __name__ == "__main__":
    main()
