"""Backtest V5 SIMPLE : reprend la logique de ml_dataset.py (rapide, fait UNE fois les calculs).

Au lieu de recalculer pour chaque OB, on :
1. Charge les data UNE FOIS
2. Detecte les OBs UNE FOIS
3. Calcule les caches UNE FOIS
4. Pour chaque OB : appelle evaluate_ob() avec le cache partage

Comme ca le pipeline V5 voit toute l'historique (comme le training).

Usage:
    python backtest_v5_simple.py XAUUSD --days 1
"""
import os
import sys
import time
import json
import pickle
import argparse
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")  # 1 thread par worker (32 workers x 1 = 32 cores)
os.environ.setdefault("LIGHTGBM_NUM_THREADS", "1")

import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from bot_v2.config import SMT_PAIRS, get_param
from bot_v2.concepts.order_block import detect_order_blocks
from bot_v2.concepts.liquidity import find_swings
from bot_v2.concepts.structure import detect_structure_breaks, detect_trend
from bot_v2.concepts.fvg import detect_fvg
from bot_v2.concepts.breaker import detect_breakers
from bot_v2.concepts.mss_setup import detect_mss_setups
from bot_v2.concepts.htf_swings import collect_htf_swings
from bot_v2.concepts.daily_bias import build_d1_from_h1
from bot_v2.pipeline import evaluate_ob
from bot_v2.backtest import simulate_trade
from bot_v2.trade_setup import compute_position_size, TradeSetup
from bot_v2 import ml_filter


VANTAGE_DIR = Path(ROOT) / "data_vantage"
ML_THRESHOLD = 0.75
BALANCE = 154.96
RISK_PCT = 0.01


def load_vantage(asset, tf):
    path = VANTAGE_DIR / f"{asset}_{tf}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def load_model_v5(asset):
    pkl = os.path.join(ROOT, "bot_v2", f"ml_model_{asset}_admiral_v5.pkl")
    feat = os.path.join(ROOT, "bot_v2", f"ml_features_{asset}_admiral_v5.json")
    with open(pkl, "rb") as f:
        model = pickle.load(f)
    features = json.loads(open(feat).read())["features"]
    return model, features


def backtest(asset, last_days):
    print(f"\n{'='*70}", flush=True)
    print(f"BACKTEST V5 SIMPLE : {asset}  [{last_days}j]", flush=True)
    print(f"{'='*70}", flush=True)

    # 1. Charge TOUT une fois
    t0 = time.time()
    df_m1 = load_vantage(asset, "M1")
    df_m15 = load_vantage(asset, "M15")
    df_h1 = load_vantage(asset, "H1")
    df_h4 = load_vantage(asset, "H4")
    df_d1 = load_vantage(asset, "D1")
    if df_d1 is None:
        df_d1 = build_d1_from_h1(df_h1)
    print(f"  Data chargee : M1={len(df_m1):,} M15={len(df_m15):,} H1={len(df_h1):,} ({time.time()-t0:.1f}s)", flush=True)

    cutoff_ts = df_m1.index[-1] - pd.Timedelta(days=last_days)
    print(f"  Periode test : {cutoff_ts} -> {df_m1.index[-1]}", flush=True)

    # 2. Modele
    model, features = load_model_v5(asset)
    print(f"  Modele V5 charge ({len(features)} features)", flush=True)

    # 3. Detecte OBs UNE fois sur tout
    sws = get_param(asset, "swing_strength_m1", 2)
    t0 = time.time()
    obs_all = detect_order_blocks(df_m1, swing_strength=sws, max_group_size=2)
    print(f"  OBs total : {len(obs_all):,} ({time.time()-t0:.1f}s)", flush=True)

    # 4. Filtre OBs sur la periode test
    obs_test = [ob for ob in obs_all if df_m1.index[ob.validation_index] >= cutoff_ts]
    print(f"  OBs periode : {len(obs_test):,}", flush=True)

    # 5. Cache UNE fois (sur tout l'historique)
    t0 = time.time()
    cache = {
        "swings_ltf": find_swings(df_m1, strength=sws),
        "fvgs_ltf": detect_fvg(df_m1),
        "breakers_ltf": detect_breakers(df_m1),
        "obs_htf": detect_order_blocks(df_m15),
        "obs_htf2": detect_order_blocks(df_h1),
    }
    cache["structure_breaks"] = detect_structure_breaks(df_m1, swings=cache["swings_ltf"])
    cache["htf_trend"] = detect_trend(cache["swings_ltf"], lookback=6)
    print(f"  Cache pre-calcule ({time.time()-t0:.1f}s)", flush=True)

    # MSS setups
    t0 = time.time()
    mss_setups = detect_mss_setups(df_m1, structure_breaks=cache["structure_breaks"], swings=cache["swings_ltf"])
    print(f"  MSS detectes : {len(mss_setups):,} ({time.time()-t0:.1f}s)", flush=True)

    # HTF swings
    htf_dfs = {"H1": df_h1, "D1": df_d1}
    if df_h4 is not None:
        htf_dfs["H4"] = df_h4
    htf_swings = collect_htf_swings(htf_dfs, swing_strength=3)

    # SMT correles
    correlated_dfs = {}
    for corr_name, corr_type in SMT_PAIRS.get(asset, []):
        df_c = load_vantage(corr_name, "M1")
        if df_c is not None:
            correlated_dfs[corr_name] = (df_c, corr_type)

    # 6. Pour chaque OB : evaluate (avec cache partage)
    trades = []
    rejets = {}
    all_probas = []
    t0 = time.time()
    for i, ob in enumerate(obs_test):
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
            rejets["exception"] = rejets.get("exception", 0) + 1
            continue

        if r.verdict != "TRADE" or r.trade_setup is None:
            reason = (r.rejection_reason or "no_trade")[:50]
            rejets[reason] = rejets.get(reason, 0) + 1
            continue

        feats = ml_filter._features_from_result(r, ob, asset, df_ltf=df_m1, df_d1=df_d1, mss_setups=mss_setups)
        X = pd.DataFrame([[feats.get(f, 0) for f in features]], columns=features)
        proba = float(model.predict_proba(X)[0, 1])
        all_probas.append(proba)

        if proba < ML_THRESHOLD:
            rejets[f"ml<{ML_THRESHOLD}"] = rejets.get(f"ml<{ML_THRESHOLD}", 0) + 1
            continue

        # Trade
        setup = r.trade_setup
        lots, risk_usd = compute_position_size(setup.entry_price, setup.stop_loss, asset, balance=BALANCE, risk_pct=RISK_PCT)
        if lots <= 0:
            continue
        sim_setup = TradeSetup(
            instrument=asset, direction=setup.direction,
            entry_price=setup.entry_price, stop_loss=setup.stop_loss,
            take_profit=setup.take_profit, rr=setup.rr,
            risk_points=setup.risk_points, reward_points=setup.reward_points,
            risk_usd=risk_usd, reward_usd=risk_usd * setup.rr,
            position_size_lots=lots, ob_validation_ts=setup.ob_validation_ts,
            tp_source=setup.tp_source,
        )
        tr = simulate_trade(sim_setup, df_m1, ob.validation_index + 1)
        trades.append({
            "ts": df_m1.index[ob.validation_index],
            "direction": ob.direction, "proba": proba,
            "outcome": tr.outcome, "pnl_usd": tr.pnl_usd,
        })

    print(f"  Pipeline+ML : {time.time()-t0:.1f}s pour {len(obs_test)} OBs", flush=True)

    # 7. Stats
    df_trades = pd.DataFrame(trades)
    if len(df_trades) > 0:
        closed = df_trades[df_trades["outcome"].isin(["WIN", "LOSS"])]
        wr = (closed["outcome"] == "WIN").mean() * 100 if len(closed) > 0 else 0
        print(f"\n  RECAP : {len(df_trades)} trades, WR={wr:.1f}% sur {len(closed)} fermes", flush=True)
        print(df_trades.to_string(), flush=True)
    else:
        print(f"\n  AUCUN trade au seuil {ML_THRESHOLD}", flush=True)

    # Distribution probas
    if all_probas:
        import numpy as np
        p = np.array(all_probas)
        print(f"\n  Probas ({len(p)} OBs ML) : min={p.min():.3f} max={p.max():.3f} mean={p.mean():.3f} median={np.median(p):.3f}", flush=True)
        for thr in [0.3, 0.5, 0.65, 0.70, 0.75]:
            n = (p >= thr).sum()
            print(f"    >= {thr:.2f} : {n} ({n/len(p)*100:.1f}%)", flush=True)

    # Top rejets
    print(f"\n  Top rejets :", flush=True)
    for reason, count in sorted(rejets.items(), key=lambda x: -x[1])[:5]:
        print(f"    {count:>4}x : {reason}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("asset")
    p.add_argument("--days", type=int, default=1)
    args = p.parse_args()
    backtest(args.asset, args.days)


if __name__ == "__main__":
    main()
