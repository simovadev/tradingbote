"""Diagnostic V7 features live vs backtest sur le MEME instant.

Reproduit le pipeline de live_runner_v2.compute_asset pour le dernier
setup detecte sur XAUUSD/NAS100/SP500, dump les 45 features + proba ML.

But : etre lance sur le PC local ET sur le VPS, comparer les sorties.
Si les features sont identiques -> le pipeline est OK, le ML est calibre.
Si elles divergent -> on a localise le bug.

Usage :
    python tools/diag_v7_features.py XAUUSD
    python tools/diag_v7_features.py NAS100
"""
from __future__ import annotations

import argparse
import json
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
from bot_v2.concepts.daily_bias import build_d1_from_h1
from bot_v2.pipeline import evaluate_ob
from bot_v2 import ml_filter
from bot_v2.live_runner_v2 import (
    load_model, predict_proba, LIVE_ASSETS,
    N_BARS_M1, N_BARS_M15, N_BARS_H1, N_BARS_D1,
)


def load_asset_dfs(asset, parquet_dir):
    """Charge M1 + resample M15/H1/H4/D1 depuis parquet_dir/{asset}_M1.parquet."""
    pq = parquet_dir / f"{asset}_M1.parquet"
    if not pq.exists():
        return None
    df = pd.read_parquet(pq)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    if len(df) < 1000:
        return None

    df_m1 = df.tail(N_BARS_M1).copy()
    # MIMIC LIVE : le live runner fait df_m1 = df_m1.iloc[:-1] pour virer la bougie en cours.
    # On reproduit le meme comportement pour comparer apples-to-apples.
    df_m1 = df_m1.iloc[:-1]

    def resample(rule, n):
        out = df_m1.resample(rule, label="left", closed="left").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna()
        # MIMIC LIVE : vire la bougie HTF en cours
        out = out.iloc[:-1]
        return out.tail(n)

    return {
        "df_m1": df_m1,
        "df_m15": resample("15min", N_BARS_M15),
        "df_h1": resample("1h", N_BARS_H1),
        "df_h4": resample("4h", 500),
        "df_d1": resample("1D", N_BARS_D1),
    }


def diag_asset(asset: str, parquet_dir: Path, n_setups: int = 5):
    """Reproduit le pipeline complet et dump les features des N derniers setups."""
    print(f"\n{'='*70}")
    print(f"=== DIAG V7 {asset} (parquet : {parquet_dir}) ===")
    print(f"{'='*70}")

    dfs = load_asset_dfs(asset, parquet_dir)
    if dfs is None:
        print(f"  KO : parquet absent ou trop petit pour {asset}")
        return

    df_m1 = dfs["df_m1"]
    df_m15 = dfs["df_m15"]
    df_h1 = dfs["df_h1"]
    df_h4 = dfs["df_h4"]
    df_d1 = dfs["df_d1"]

    print(f"  M1 : {len(df_m1):>6} bougies, derniere = {df_m1.index[-1]}")
    print(f"  M15: {len(df_m15):>6} bougies, derniere = {df_m15.index[-1]}")
    print(f"  H1 : {len(df_h1):>6} bougies, derniere = {df_h1.index[-1]}")
    print(f"  D1 : {len(df_d1):>6} bougies, derniere = {df_d1.index[-1]}")
    print(f"  Derniere bougie M1 OHLCV : {df_m1.iloc[-1].to_dict()}")

    if df_d1 is None or len(df_d1) < 10:
        df_d1 = build_d1_from_h1(df_h1)

    htf_dfs = {"H1": df_h1, "D1": df_d1}
    if df_h4 is not None and len(df_h4) > 0:
        htf_dfs["H4"] = df_h4
    htf_swings = collect_htf_swings(htf_dfs, swing_strength=3)

    # SMT
    correlated_dfs = {}
    for corr_name, corr_type in SMT_PAIRS.get(asset, []):
        cd = load_asset_dfs(corr_name, parquet_dir)
        if cd is not None:
            correlated_dfs[corr_name] = (cd["df_m1"], corr_type)

    # Detection (meme que compute_asset)
    sws = get_param(asset, "swing_strength_m1", 2)
    obs = detect_order_blocks(df_m1, swing_strength=sws, max_group_size=2)
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
        df_m1,
        structure_breaks=cache["structure_breaks"],
        swings=cache["swings_ltf"],
        fvgs=cache["fvgs_ltf"],
    )

    print(f"\n  Detected {len(obs)} OBs, {len(mss_setups)} MSS setups")
    cutoff = df_m1.index[-1] - pd.Timedelta(hours=4)
    obs_recent = [ob for ob in obs if df_m1.index[ob.validation_index] >= cutoff]
    print(f"  OBs derniere 4h : {len(obs_recent)}")

    loaded = load_model(asset)
    if loaded is None:
        print(f"  !! Pas de modele pour {asset}")
        return
    model, features = loaded
    threshold = ml_filter.get_dynamic_threshold(asset, balance=200)
    print(f"  Modele : {len(features)} features, threshold {threshold}")

    # Evalue chaque OB recent + dump les features
    results = []
    for i, ob in enumerate(obs_recent[-n_setups:]):
        ts_val = df_m1.index[ob.validation_index]
        try:
            r = evaluate_ob(
                ob, df_m1, df_m15, df_d1, asset,
                ltf_name="M1", htf_name="M15",
                df_htf2=df_h1, htf2_name="H1",
                correlated_dfs=correlated_dfs,
                htf_swings=htf_swings,
                df_h1=df_h1, min_score=0, min_quality=0,
                cache=cache,
            )
        except Exception as e:
            print(f"  [{ts_val}] evaluate_ob exception : {e}")
            continue

        if r.verdict != "TRADE" or r.trade_setup is None:
            print(f"  [{ts_val}] {ob.direction:>7s} REJET pipeline : {r.rejection_reason}")
            continue

        # Calcul features
        feat_dict = ml_filter._features_from_result(
            r, ob, asset, df_ltf=df_m1, df_d1=df_d1, mss_setups=mss_setups,
        )
        X = pd.DataFrame([[feat_dict.get(f, 0) for f in features]], columns=features)
        proba = float(model.predict_proba(X)[0, 1])

        print(f"\n  --- Setup {i+1} | {ts_val} | {ob.direction} | score={r.score} | ML={proba:.3f} ---")
        for k, v in sorted(feat_dict.items()):
            print(f"    {k:<30s} = {v}")
        results.append({
            "ts": str(ts_val),
            "direction": ob.direction,
            "score": int(r.score),
            "proba": proba,
            "features": {k: float(v) if isinstance(v, (int, float, bool)) else v
                         for k, v in feat_dict.items()},
        })

    # Dump en JSON pour comparaison
    out_path = ROOT / f"diag_v7_features_{asset}.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  >>> Dump : {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("asset", help="XAUUSD, NAS100, ...")
    p.add_argument("--data-dir", default="data_vantage",
                   help="Repertoire parquet (defaut: data_vantage)")
    p.add_argument("--n", type=int, default=5, help="N derniers setups a dumper")
    args = p.parse_args()

    parquet_dir = ROOT / args.data_dir
    if not parquet_dir.exists():
        print(f"!! repertoire absent : {parquet_dir}")
        sys.exit(1)

    diag_asset(args.asset.upper(), parquet_dir, n_setups=args.n)


if __name__ == "__main__":
    main()
