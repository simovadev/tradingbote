"""Confirme le bug data leakage PDH/PDL.

Compare 3 calculs pour le meme setup :
1. PDH/PDL via le bug actuel (df_d1.index < ts) -> inclut D1 du jour
2. PDH/PDL fix (df_d1.index < ts.normalize()) -> exclut D1 du jour
3. PDH/PDL en live (df_d1 resample sans la journee future)

Et calcule la proba V7 dans les 3 cas.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import numpy as np

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
from bot_v2.concepts.daily_bias import build_d1_from_h1
from bot_v2.pipeline import evaluate_ob
from bot_v2 import ml_filter
from bot_v2.live_runner_v2 import load_model, N_BARS_M1, N_BARS_M15, N_BARS_H1, N_BARS_D1


def main():
    asset = "DJ30"
    parquet_dir = ROOT / "data_vantage"

    print(f"=== CONFIRM PDH/PDL DATA LEAKAGE BUG ({asset}) ===\n")

    df = pd.read_parquet(parquet_dir / f"{asset}_M1.parquet")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df_m1 = df.tail(N_BARS_M1).copy()

    def resample(rule, n):
        out = df_m1.resample(rule, label="left", closed="left").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna()
        return out.tail(n)

    df_m15 = resample("15min", N_BARS_M15)
    df_h1 = resample("1h", N_BARS_H1)
    df_h4 = resample("4h", 500)
    df_d1_full = resample("1D", N_BARS_D1)

    htf_swings = collect_htf_swings({"H1": df_h1, "H4": df_h4, "D1": df_d1_full}, swing_strength=3)
    correlated_dfs = {}
    for corr_name, corr_type in SMT_PAIRS.get(asset, []):
        try:
            df_c = pd.read_parquet(parquet_dir / f"{corr_name}_M1.parquet")
            if df_c.index.tz is None:
                df_c.index = df_c.index.tz_localize("UTC")
            correlated_dfs[corr_name] = (df_c.tail(N_BARS_M1).copy(), corr_type)
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
        df_m1, swings=cache["swings_ltf"], fvgs=cache["fvgs_ltf"]
    )
    cache["htf_trend"] = detect_trend(cache["swings_ltf"], lookback=6)
    cache["obs_htf2"] = detect_order_blocks(df_h1)
    mss_setups = detect_mss_setups(
        df_m1, structure_breaks=cache["structure_breaks"], swings=cache["swings_ltf"],
        fvgs=cache["fvgs_ltf"],
    )

    # Cherche le setup DJ30 12:34 (score 144)
    target_ts = pd.Timestamp("2026-05-21 12:34:00", tz="UTC")
    target_obs = [ob for ob in obs if df_m1.index[ob.validation_index] == target_ts]
    if not target_obs:
        # Fallback : prend le plus proche
        idx_target = df_m1.index.get_indexer([target_ts], method="nearest")[0]
        nearest_ts = df_m1.index[idx_target]
        target_obs = [ob for ob in obs if df_m1.index[ob.validation_index] == nearest_ts]
    if not target_obs:
        print(f"Aucun OB trouve a {target_ts}")
        return

    ob = target_obs[0]
    ts_val = df_m1.index[ob.validation_index]
    print(f"Setup cible : {ts_val} | {ob.direction} | obs trouves : {len(target_obs)}")
    print()

    loaded = load_model(asset)
    model, features = loaded

    # ================ TEST 1 : Bug actuel (D1 includes today bar) ================
    df_d1_buggy = df_d1_full.copy()  # contient le D1 d'aujourd'hui partiellement formee

    # ================ TEST 2 : Fix (exclure D1 du jour de validation) ================
    today_start = ts_val.normalize()
    df_d1_fixed = df_d1_full[df_d1_full.index < today_start].copy()

    # ================ TEST 3 : Live = D1 resample comme buffer (jour partiel) ================
    # En live, df_d1 contient les D1 PASSES complets + la D1 du jour PARTIELLE.
    # On simule ca : on remplace le D1 du jour par sa version "jusqu'a ts_val"
    df_d1_live = df_d1_full[df_d1_full.index < today_start].copy()
    # Resample M1 jusqu'a ts_val pour avoir le D1 du jour partiel
    df_m1_until_now = df_m1[df_m1.index <= ts_val]
    if len(df_m1_until_now) > 0:
        today_bars = df_m1_until_now[df_m1_until_now.index >= today_start]
        if len(today_bars) > 0:
            today_row = pd.DataFrame({
                "open": [today_bars.iloc[0]["open"]],
                "high": [today_bars["high"].max()],
                "low":  [today_bars["low"].min()],
                "close": [today_bars.iloc[-1]["close"]],
                "volume": [today_bars["volume"].sum()],
            }, index=[today_start])
            df_d1_live = pd.concat([df_d1_live, today_row])

    print(f"D1 STATE :")
    print(f"  Bug actuel       last D1 = {df_d1_buggy.index[-1]} (high={df_d1_buggy.iloc[-1]['high']:.2f}, low={df_d1_buggy.iloc[-1]['low']:.2f})")
    print(f"  Fix exclude today last D1 = {df_d1_fixed.index[-1]} (high={df_d1_fixed.iloc[-1]['high']:.2f}, low={df_d1_fixed.iloc[-1]['low']:.2f})")
    print(f"  Live partiel     last D1 = {df_d1_live.index[-1]} (high={df_d1_live.iloc[-1]['high']:.2f}, low={df_d1_live.iloc[-1]['low']:.2f})")
    print()

    for label, df_d1_used in [("BUG (D1 today complete)", df_d1_buggy),
                                ("FIX (no D1 today)", df_d1_fixed),
                                ("LIVE (D1 today partial)", df_d1_live)]:
        try:
            r = evaluate_ob(
                ob, df_m1, df_m15, df_d1_used, asset,
                ltf_name="M1", htf_name="M15",
                df_htf2=df_h1, htf2_name="H1",
                correlated_dfs=correlated_dfs,
                htf_swings=htf_swings,
                df_h1=df_h1, min_score=0, min_quality=0,
                cache=cache,
            )
        except Exception as e:
            print(f"  {label:<32} ERROR : {e}")
            continue
        if r.verdict != "TRADE" or r.trade_setup is None:
            print(f"  {label:<32} REJET : {r.rejection_reason}")
            continue
        feat = ml_filter._features_from_result(r, ob, asset, df_ltf=df_m1, df_d1=df_d1_used, mss_setups=mss_setups)
        X = pd.DataFrame([[feat.get(f, 0) for f in features]], columns=features)
        proba = float(model.predict_proba(X)[0, 1])
        pdh = feat.get("dist_to_pdh_pct")
        pdl = feat.get("dist_to_pdl_pct")
        d1o = feat.get("dist_to_d1_open_pct")
        print(f"  {label:<32} proba={proba:.3f}  pdh={pdh:.3f}  pdl={pdl:.3f}  d1_open={d1o:.3f}")


if __name__ == "__main__":
    main()
