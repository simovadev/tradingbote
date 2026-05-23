"""Test d'equivalence STRICTE du cache_filter.

Pour chaque actif et plusieurs cut_iloc :
   filter(cache_full, cut)  ==  compute_direct(df[:cut])

Si TOUT passe -> on peut utiliser filter_cache_for_cut() dans le backtest.
Sinon -> identifie quelle fonction casse.
"""
from __future__ import annotations
import os
import sys
from dataclasses import asdict

os.environ.setdefault("BUILD_DATA_DIR", "data_vantage")
sys.path.insert(0, "c:/Users/Shadow/TradingBot")

import pandas as pd

from bot_v2.concepts.order_block import detect_order_blocks
from bot_v2.concepts.liquidity import find_swings
from bot_v2.concepts.fvg import detect_fvg
from bot_v2.concepts.breaker import detect_breakers
from bot_v2.concepts.structure import detect_structure_breaks
from bot_v2.cache_filter import filter_cache_for_cut

DATA_DIR = "c:/Users/Shadow/TradingBot/data_vantage"
N_FULL = 88000


def sig_obs(obs):
    return set((o.direction, o.validation_index, round(o.ob_high, 6), round(o.ob_low, 6),
                o.group_start_index, o.group_end_index) for o in obs)


def sig_swings(sw):
    return set((s.kind, s.index, round(s.price, 6), s.strength, s.is_live) for s in sw)


def sig_fvgs(fv):
    return set((f.direction, f.center_index, round(f.top, 6), round(f.bottom, 6),
                f.rebalanced, f.rebalance_index, f.inversed, f.inverse_index) for f in fv)


def sig_breakers(br):
    return set((b.direction, b.inverse_index, round(b.top, 6), round(b.bottom, 6),
                b.has_quality_ob_origin, b.has_displacement, b.associated_pdr_count) for b in br)


def sig_sb(sb):
    return set((s.kind, s.direction, s.break_index, round(s.break_close, 6),
                s.has_displacement_fvg) for s in sb)


def main():
    asset = sys.argv[1] if len(sys.argv) > 1 else "XAUUSD"
    print(f"=== Test equivalence cache_filter : {asset} ===\n")

    df_full = pd.read_parquet(f"{DATA_DIR}/{asset}_M1.parquet").tail(N_FULL + 1000)
    n_total = len(df_full)
    print(f"   {n_total} bougies M1 chargees")

    # Cache complet : calcule sur toutes les bougies
    print("\n1. Calcul du cache complet (88k + futur)")
    cache_full = {
        "obs": detect_order_blocks(df_full, swing_strength=2),
        "swings_ltf": find_swings(df_full, strength=2),
        "fvgs_ltf": detect_fvg(df_full),
        "breakers_ltf": detect_breakers(df_full),
    }
    cache_full["structure_breaks"] = detect_structure_breaks(
        df_full, swings=cache_full["swings_ltf"], fvgs=cache_full["fvgs_ltf"]
    )
    cache_full["obs_htf"] = []  # pas teste ici (M15)
    cache_full["obs_htf2"] = []  # pas teste ici (H1)
    cache_full["htf_trend"] = None
    print(f"   cache : obs={len(cache_full['obs'])} swings={len(cache_full['swings_ltf'])}"
          f" fvgs={len(cache_full['fvgs_ltf'])} breakers={len(cache_full['breakers_ltf'])}"
          f" sb={len(cache_full['structure_breaks'])}")

    # 5 cuts a tester (a 88k, 88k-100, 88k-500, 88k-1000)
    cuts = [n_total - 1000, n_total - 500, n_total - 200, n_total - 50, n_total - 10]
    cuts = [c for c in cuts if c > 200]

    print(f"\n2. Test sur {len(cuts)} valeurs de cut_iloc :\n")
    all_ok = True
    for cut in cuts:
        df_cut = df_full.iloc[:cut]
        # Recalcul direct sur df[:cut]
        direct = {
            "obs": detect_order_blocks(df_cut, swing_strength=2),
            "swings_ltf": find_swings(df_cut, strength=2),
            "fvgs_ltf": detect_fvg(df_cut),
            "breakers_ltf": detect_breakers(df_cut),
        }
        direct["structure_breaks"] = detect_structure_breaks(
            df_cut, swings=direct["swings_ltf"], fvgs=direct["fvgs_ltf"]
        )

        # Filtre du cache full
        filtered = filter_cache_for_cut(cache_full, cut_iloc_m1=cut, df_m1_cut=df_cut)

        print(f"--- cut={cut} (df_cut={len(df_cut)}) ---")
        cut_ok = True
        # OB
        sf = sig_obs(filtered["obs"]); sd = sig_obs(direct["obs"])
        if sf == sd:
            print(f"   obs              OK ({len(direct['obs'])})")
        else:
            print(f"   obs              FAIL  direct={len(direct['obs'])} filtered={len(filtered['obs'])}"
                  f"  only_direct={len(sd-sf)} only_filtered={len(sf-sd)}")
            for x in list(sd-sf)[:2]: print(f"      only_direct: {x}")
            for x in list(sf-sd)[:2]: print(f"      only_filtered: {x}")
            cut_ok = False

        # Swings
        sf = sig_swings(filtered["swings_ltf"]); sd = sig_swings(direct["swings_ltf"])
        if sf == sd:
            print(f"   swings_ltf       OK ({len(direct['swings_ltf'])})")
        else:
            print(f"   swings_ltf       FAIL  direct={len(direct['swings_ltf'])} filtered={len(filtered['swings_ltf'])}"
                  f"  only_direct={len(sd-sf)} only_filtered={len(sf-sd)}")
            for x in list(sd-sf)[:2]: print(f"      only_direct: {x}")
            for x in list(sf-sd)[:2]: print(f"      only_filtered: {x}")
            cut_ok = False

        # FVG
        sf = sig_fvgs(filtered["fvgs_ltf"]); sd = sig_fvgs(direct["fvgs_ltf"])
        if sf == sd:
            print(f"   fvgs_ltf         OK ({len(direct['fvgs_ltf'])})")
        else:
            print(f"   fvgs_ltf         FAIL  direct={len(direct['fvgs_ltf'])} filtered={len(filtered['fvgs_ltf'])}"
                  f"  only_direct={len(sd-sf)} only_filtered={len(sf-sd)}")
            for x in list(sd-sf)[:2]: print(f"      only_direct: {x}")
            for x in list(sf-sd)[:2]: print(f"      only_filtered: {x}")
            cut_ok = False

        # Breakers
        sf = sig_breakers(filtered["breakers_ltf"]); sd = sig_breakers(direct["breakers_ltf"])
        if sf == sd:
            print(f"   breakers_ltf     OK ({len(direct['breakers_ltf'])})")
        else:
            print(f"   breakers_ltf     FAIL  direct={len(direct['breakers_ltf'])} filtered={len(filtered['breakers_ltf'])}"
                  f"  only_direct={len(sd-sf)} only_filtered={len(sf-sd)}")
            for x in list(sd-sf)[:2]: print(f"      only_direct: {x}")
            for x in list(sf-sd)[:2]: print(f"      only_filtered: {x}")
            cut_ok = False

        # StructureBreak
        sf = sig_sb(filtered["structure_breaks"]); sd = sig_sb(direct["structure_breaks"])
        if sf == sd:
            print(f"   structure_breaks OK ({len(direct['structure_breaks'])})")
        else:
            print(f"   structure_breaks FAIL  direct={len(direct['structure_breaks'])} filtered={len(filtered['structure_breaks'])}"
                  f"  only_direct={len(sd-sf)} only_filtered={len(sf-sd)}")
            cut_ok = False

        if not cut_ok:
            all_ok = False

    print()
    if all_ok:
        print("=== CACHE FILTER SAFE -> deployable ===")
    else:
        print("=== DIVERGENCE -> ne pas utiliser tel quel ===")


if __name__ == "__main__":
    main()
