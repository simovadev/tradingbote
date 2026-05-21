"""Compare _extract_features (ml_dataset, training) vs _features_from_result
(ml_filter, live) sur EXACTEMENT le meme OB/result.

Si les 2 divergent -> le ML V8 a appris sur des features differentes de
celles qu'il recoit en live -> probas ecrasees. C'est LE bug a traquer.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bot_v2.config import SMT_PAIRS, get_param
from bot_v2.concepts.htf_swings import collect_htf_swings
from bot_v2.concepts.liquidity import find_swings
from bot_v2.concepts.fvg import detect_fvg
from bot_v2.concepts.breaker import detect_breakers
from bot_v2.concepts.order_block import detect_order_blocks
from bot_v2.concepts.structure import detect_structure_breaks, detect_trend
from bot_v2.concepts.mss_setup import detect_mss_setups
from bot_v2.pipeline import evaluate_ob
from bot_v2 import ml_filter, ml_dataset
from bot_v2.live_runner_v2 import N_BARS_M1, N_BARS_M15, N_BARS_H1, N_BARS_D1


def main():
    asset = "XAUUSD"
    pq_dir = ROOT / "data_vantage"
    df = pd.read_parquet(pq_dir / f"{asset}_M1.parquet")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df_m1 = df.tail(N_BARS_M1).copy()

    def rs(rule, n):
        out = df_m1.resample(rule, label="left", closed="left").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum"}).dropna()
        return out.tail(n)

    df_m15 = rs("15min", N_BARS_M15)
    df_h1 = rs("1h", N_BARS_H1)
    df_h4 = rs("4h", 500)
    df_d1 = rs("1D", N_BARS_D1)

    htf_swings = collect_htf_swings({"H1": df_h1, "H4": df_h4, "D1": df_d1}, swing_strength=3)
    correlated_dfs = {}
    for cn, ct in SMT_PAIRS.get(asset, []):
        try:
            dc = pd.read_parquet(pq_dir / f"{cn}_M1.parquet")
            if dc.index.tz is None:
                dc.index = dc.index.tz_localize("UTC")
            correlated_dfs[cn] = (dc.tail(N_BARS_M1).copy(), ct)
        except Exception:
            pass

    sws = get_param(asset, "swing_strength_m1", 2)
    obs = detect_order_blocks(df_m1, swing_strength=sws)
    cache = {
        "swings_ltf": find_swings(df_m1, strength=sws),
        "fvgs_ltf": detect_fvg(df_m1),
        "breakers_ltf": detect_breakers(df_m1),
        "obs_htf": detect_order_blocks(df_m15),
    }
    cache["structure_breaks"] = detect_structure_breaks(
        df_m1, swings=cache["swings_ltf"], fvgs=cache["fvgs_ltf"])
    cache["htf_trend"] = detect_trend(cache["swings_ltf"], lookback=6)
    cache["obs_htf2"] = detect_order_blocks(df_h1)
    mss = detect_mss_setups(df_m1, structure_breaks=cache["structure_breaks"],
                            swings=cache["swings_ltf"], fvgs=cache["fvgs_ltf"])

    # Prend les 3 derniers OB qui donnent un TRADE
    n_tested = 0
    for ob in reversed(obs[-50:]):
        try:
            r = evaluate_ob(ob, df_m1, df_m15, df_d1, asset,
                ltf_name="M1", htf_name="M15", df_htf2=df_h1, htf2_name="H1",
                correlated_dfs=correlated_dfs, htf_swings=htf_swings,
                df_h1=df_h1, min_score=0, min_quality=0, cache=cache)
        except Exception:
            continue
        if r.verdict != "TRADE" or r.trade_setup is None:
            continue

        # Feature func LIVE
        f_live = ml_filter._features_from_result(
            r, ob, asset, df_ltf=df_m1, df_d1=df_d1, mss_setups=mss)
        # Feature func TRAINING
        f_train = ml_dataset._extract_features(
            r, ob, asset, df_ltf=df_m1, df_d1=df_d1, df_htf=df_m15, mss_setups=mss)

        ts = df_m1.index[ob.validation_index]
        print(f"\n{'='*70}")
        print(f"OB {ts} | {ob.direction} | score={r.score}")
        print(f"{'='*70}")
        all_keys = sorted(set(f_live.keys()) | set(f_train.keys()))
        diffs = 0
        for k in all_keys:
            vl = f_live.get(k, "<ABSENT>")
            vt = f_train.get(k, "<ABSENT>")
            same = False
            try:
                same = abs(float(vl) - float(vt)) < 1e-6
            except (ValueError, TypeError):
                same = (vl == vt)
            if not same:
                diffs += 1
                print(f"  DIFF {k:<28} live={vl!s:<22} train={vt!s}")
        if diffs == 0:
            print("  -> AUCUNE difference, les 2 fonctions sont identiques")
        else:
            print(f"  -> {diffs} FEATURES DIVERGENTES")

        n_tested += 1
        if n_tested >= 3:
            break


if __name__ == "__main__":
    main()
