"""Simulation bot live sur 7 jours consecutifs.

Reproduit EXACTEMENT le comportement live :
- Pour chaque OB du jour J, applique la fenetre 3 mois en arriere
- Calcule pipeline + features comme le bot live
- Applique le modele V5 8 ans

Comparaison avec check_week_all_assets.py :
- check_week = lit features depuis dataset training (deja calculees sur chunks 3 mois)
- simulate_live = recalcule features sur fenetre 3 mois (= ce que fait le live)
Si les deux donnent les memes chiffres -> le code marche, bug live est ailleurs (MT5)
Si simulate_live donne des chiffres differents -> le bug est dans simulate_live (alignement)

Usage :
    python simulate_live_week.py XAUUSD
    python simulate_live_week.py --all
"""
import os
import sys
import time
import json
import pickle
import argparse
from pathlib import Path

import pandas as pd

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
from bot_v2.data_loader import load
from bot_v2.pipeline import evaluate_ob
from bot_v2.backtest import simulate_trade
from bot_v2.trade_setup import compute_position_size, TradeSetup
from bot_v2 import ml_filter


ML_THRESHOLD = 0.75
BALANCE = 154.96
RISK_PCT = 0.01
# Fenetre = identique au training V5 (chunks 14j + 30j buffer = 44j)
LOOKBACK_DAYS = int(os.environ.get("LOOKBACK_DAYS", "44"))

ALL_ASSETS = [
    "XAUUSD", "NAS100", "GER40", "BTCUSD",
    "EURUSD", "GBPUSD", "AUDUSD", "USDJPY",
    "SP500", "DJ30", "UK100", "FRA40",
    "USDCAD", "USDCHF",
]

END_TS = pd.Timestamp("2026-05-19", tz="UTC")
START_TS = END_TS - pd.Timedelta(days=7)


def load_model_v5(asset):
    pkl = os.path.join(ROOT, "bot_v2", f"ml_model_{asset}_admiral_v5.pkl")
    feat = os.path.join(ROOT, "bot_v2", f"ml_features_{asset}_admiral_v5.json")
    if not os.path.exists(pkl):
        return None, None
    with open(pkl, "rb") as f:
        model = pickle.load(f)
    features = json.loads(open(feat).read())["features"]
    return model, features


def simulate_asset(asset):
    print(f"\n{'='*70}", flush=True)
    print(f"SIMULATE LIVE WEEK : {asset} ({START_TS.date()} -> {END_TS.date()})", flush=True)
    print(f"{'='*70}", flush=True)

    model, features = load_model_v5(asset)
    if model is None:
        print(f"  KO: modele V5 manquant", flush=True)
        return None

    t0 = time.time()
    df_m1_full = load(asset, "M1")
    df_m15_full = load(asset, "M15")
    df_h1_full = load(asset, "H1")
    try:
        df_h4_full = load(asset, "H4")
    except Exception:
        df_h4_full = None
    try:
        df_d1_full = load(asset, "D1")
        if len(df_d1_full) < 10:
            raise ValueError
    except Exception:
        df_d1_full = build_d1_from_h1(df_h1_full)

    # Slice : on prend 3 mois avant START + jusqu'a END
    lookback_start = START_TS - pd.Timedelta(days=LOOKBACK_DAYS)
    df_m1 = df_m1_full[(df_m1_full.index >= lookback_start) & (df_m1_full.index <= END_TS)].copy()
    df_m15 = df_m15_full[(df_m15_full.index >= lookback_start) & (df_m15_full.index <= END_TS)].copy()
    df_h1 = df_h1_full[(df_h1_full.index >= lookback_start) & (df_h1_full.index <= END_TS)].copy()
    df_h4 = df_h4_full[(df_h4_full.index >= lookback_start) & (df_h4_full.index <= END_TS)].copy() if df_h4_full is not None else None
    df_d1 = df_d1_full[(df_d1_full.index >= lookback_start) & (df_d1_full.index <= END_TS)].copy()
    print(f"  Data : M1={len(df_m1):,} M15={len(df_m15):,} H1={len(df_h1):,} ({time.time()-t0:.1f}s)", flush=True)

    # SMT correles
    correlated_dfs = {}
    for corr_name, corr_type in SMT_PAIRS.get(asset, []):
        try:
            df_c_full = load(corr_name, "M1")
            df_c = df_c_full[(df_c_full.index >= lookback_start) & (df_c_full.index <= END_TS)].copy()
            if len(df_c) > 0:
                correlated_dfs[corr_name] = (df_c, corr_type)
        except Exception:
            continue

    # Detection OBs sur toute la periode
    sws = get_param(asset, "swing_strength_m1", 2)
    t0 = time.time()
    obs_all = detect_order_blocks(df_m1, swing_strength=sws, max_group_size=2)
    obs_week = [ob for ob in obs_all if START_TS <= df_m1.index[ob.validation_index] <= END_TS]
    print(f"  OBs total : {len(obs_all):,} | semaine : {len(obs_week)} ({time.time()-t0:.1f}s)", flush=True)

    # Cache partage
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
    mss_setups = detect_mss_setups(df_m1, structure_breaks=cache["structure_breaks"], swings=cache["swings_ltf"])
    print(f"  Cache + MSS ({len(mss_setups):,}) calcules en {time.time()-t0:.1f}s", flush=True)

    htf_dfs = {"H1": df_h1, "D1": df_d1}
    if df_h4 is not None:
        htf_dfs["H4"] = df_h4
    htf_swings = collect_htf_swings(htf_dfs, swing_strength=3)

    # Eval OBs
    trades = []
    rejets = {}
    all_probas = []
    t0 = time.time()
    for ob in obs_week:
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
        except Exception:
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
            "proba": proba,
            "outcome": tr.outcome,
            "pnl_usd": tr.pnl_usd,
        })
    print(f"  Eval {len(obs_week)} OBs en {time.time()-t0:.1f}s", flush=True)

    # Stats
    import numpy as np
    p = np.array(all_probas) if all_probas else np.array([0])
    df_trades = pd.DataFrame(trades)
    closed = df_trades[df_trades["outcome"].isin(["WIN", "LOSS"])] if len(df_trades) > 0 else pd.DataFrame()
    n_win = (closed["outcome"] == "WIN").sum() if len(closed) > 0 else 0
    n_loss = (closed["outcome"] == "LOSS").sum() if len(closed) > 0 else 0
    wr = n_win / (n_win + n_loss) * 100 if (n_win + n_loss) > 0 else 0
    pnl = df_trades["pnl_usd"].sum() if len(df_trades) > 0 else 0

    print(f"\n  RECAP {asset} (7j simulate live) :", flush=True)
    print(f"    OBs week : {len(obs_week)}", flush=True)
    print(f"    OBs ML evalues : {len(all_probas)}", flush=True)
    print(f"    Max proba : {p.max():.3f} | Mean : {p.mean():.3f}", flush=True)
    print(f"    Trades pris (ML>={ML_THRESHOLD}) : {len(trades)}", flush=True)
    print(f"    WIN/LOSS : {n_win}/{n_loss} | WR : {wr:.1f}%", flush=True)
    print(f"    PnL : {pnl:+.2f}$", flush=True)

    return {
        "asset": asset,
        "obs_week": len(obs_week),
        "obs_ml": len(all_probas),
        "max_proba": p.max(),
        "trades": len(trades),
        "win": n_win,
        "loss": n_loss,
        "wr": wr,
        "pnl": pnl,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("asset", nargs="?")
    p.add_argument("--all", action="store_true")
    args = p.parse_args()

    assets = ALL_ASSETS if (args.all or args.asset == "--all") else [args.asset]
    if not assets[0]:
        p.print_help()
        return

    results = []
    t0 = time.time()
    for a in assets:
        if a not in ALL_ASSETS:
            continue
        try:
            r = simulate_asset(a)
            if r:
                results.append(r)
        except Exception as e:
            print(f"!! FAIL {a}: {e}", flush=True)
            import traceback; traceback.print_exc()

    if len(results) > 1:
        print(f"\n\n{'='*90}", flush=True)
        print(f"RECAP GLOBAL SIMULATE LIVE 7J ({time.time()-t0:.0f}s)", flush=True)
        print(f"{'='*90}", flush=True)
        print(f"{'ASSET':<10} {'OBs':>6} {'ML':>6} {'MaxP':>6} {'TRADES':>7} {'WIN':>5} {'LOSS':>5} {'WR':>6} {'PnL$':>10}", flush=True)
        print("-"*90, flush=True)
        tot = {"obs": 0, "ml": 0, "trades": 0, "win": 0, "loss": 0, "pnl": 0}
        for r in results:
            print(f"{r['asset']:<10} {r['obs_week']:>6} {r['obs_ml']:>6} {r['max_proba']:>5.3f} {r['trades']:>7} {r['win']:>5} {r['loss']:>5} {r['wr']:>5.1f}% {r['pnl']:>+10.2f}", flush=True)
            tot["obs"] += r['obs_week']; tot["ml"] += r['obs_ml']; tot["trades"] += r['trades']
            tot["win"] += r['win']; tot["loss"] += r['loss']; tot["pnl"] += r['pnl']
        wr = tot["win"]/(tot["win"]+tot["loss"])*100 if (tot["win"]+tot["loss"]) > 0 else 0
        print("-"*90, flush=True)
        print(f"{'TOTAL':<10} {tot['obs']:>6} {tot['ml']:>6} {'':>6} {tot['trades']:>7} {tot['win']:>5} {tot['loss']:>5} {wr:>5.1f}% {tot['pnl']:>+10.2f}", flush=True)
        print(f"\nTrades/jour total : {tot['trades']/7:.1f}", flush=True)


if __name__ == "__main__":
    main()
