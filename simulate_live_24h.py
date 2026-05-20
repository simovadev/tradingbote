"""Simule le bot live sur 24h precises, avec EXACTEMENT le comportement live.

Le bot voit :
- 3 mois de M1 en arriere depuis le moment present (sliding window)
- Le modele V5 8 ans deja entraine (ml_model_*_admiral_v5.pkl)

On simule une journee J et on compte les trades pris/rejetes/WR.

Usage:
    python simulate_live_24h.py XAUUSD --day 2026-02-15
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
BALANCE = 60.0    # IDENTIQUE training V5
RISK_PCT = 0.10   # IDENTIQUE training V5 (10%)

# Fenetre live = 3 mois en arriere (comme le bot live aujourd'hui)
LOOKBACK_DAYS = 90


def load_model_v5(asset):
    pkl = os.path.join(ROOT, "bot_v2", f"ml_model_{asset}_admiral_v5.pkl")
    feat = os.path.join(ROOT, "bot_v2", f"ml_features_{asset}_admiral_v5.json")
    with open(pkl, "rb") as f:
        model = pickle.load(f)
    features = json.loads(open(feat).read())["features"]
    return model, features


def main():
    p = argparse.ArgumentParser()
    p.add_argument("asset")
    p.add_argument("--day", required=True, help="Jour a simuler (YYYY-MM-DD)")
    args = p.parse_args()

    asset = args.asset
    day_ts = pd.Timestamp(args.day, tz="UTC")
    day_end = day_ts + pd.Timedelta(days=1)
    # Le bot voit 3 mois en arriere depuis le jour J
    lookback_start = day_ts - pd.Timedelta(days=LOOKBACK_DAYS)

    print(f"\n{'='*70}", flush=True)
    print(f"SIMULATION LIVE 24H : {asset}", flush=True)
    print(f"  Jour simule : {day_ts.date()} -> {day_end.date()}", flush=True)
    print(f"  Fenetre vue par bot : {lookback_start.date()} -> {day_ts.date()} (3 mois)", flush=True)
    print(f"  Modele : V5 8 ans (ml_model_{asset}_admiral_v5.pkl)", flush=True)
    print(f"{'='*70}", flush=True)

    # 1. Charge data Admiral
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
    print(f"  Data chargee : M1={len(df_m1_full):,} ({time.time()-t0:.1f}s)", flush=True)

    # 2. Slice : 3 mois jusqu'a la fin du jour J (le bot voit toute la journee progressivement)
    # Pour simplifier : on coupe a day_end, et le bot voit lookback_start -> day_end
    # Mais l'OB doit etre dans [day_ts, day_end] pour compter dans le jour J
    df_m1 = df_m1_full[(df_m1_full.index >= lookback_start) & (df_m1_full.index <= day_end)].copy()
    df_m15 = df_m15_full[(df_m15_full.index >= lookback_start) & (df_m15_full.index <= day_end)].copy()
    df_h1 = df_h1_full[(df_h1_full.index >= lookback_start) & (df_h1_full.index <= day_end)].copy()
    df_h4 = df_h4_full[(df_h4_full.index >= lookback_start) & (df_h4_full.index <= day_end)].copy() if df_h4_full is not None else None
    df_d1 = df_d1_full[(df_d1_full.index >= lookback_start) & (df_d1_full.index <= day_end)].copy()
    print(f"  Slice 3 mois : M1={len(df_m1):,} M15={len(df_m15):,} H1={len(df_h1):,}", flush=True)

    # 3. Modele V5 8 ans
    model, features = load_model_v5(asset)
    print(f"  Modele V5 charge ({len(features)} features)", flush=True)

    # 4. SMT correles (3 mois)
    correlated_dfs = {}
    for corr_name, corr_type in SMT_PAIRS.get(asset, []):
        try:
            df_c_full = load(corr_name, "M1")
            df_c = df_c_full[(df_c_full.index >= lookback_start) & (df_c_full.index <= day_end)].copy()
            if len(df_c) > 0:
                correlated_dfs[corr_name] = (df_c, corr_type)
        except Exception:
            continue

    # 5. Detection OBs sur les 3 mois (le bot voit toute la fenetre)
    sws = get_param(asset, "swing_strength_m1", 2)
    t0 = time.time()
    obs_all = detect_order_blocks(df_m1, swing_strength=sws, max_group_size=2)
    print(f"  OBs detectes (3 mois) : {len(obs_all):,} ({time.time()-t0:.1f}s)", flush=True)

    # 6. Filtre : OBs valides dans la journee J seulement
    obs_day = [ob for ob in obs_all if day_ts <= df_m1.index[ob.validation_index] <= day_end]
    print(f"  OBs jour J : {len(obs_day)}", flush=True)

    # 7. Cache partage (calcule sur 3 mois, comme le ferait le live)
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

    mss_setups = detect_mss_setups(df_m1, structure_breaks=cache["structure_breaks"], swings=cache["swings_ltf"])
    print(f"  MSS : {len(mss_setups):,}", flush=True)

    # HTF swings
    htf_dfs = {"H1": df_h1, "D1": df_d1}
    if df_h4 is not None:
        htf_dfs["H4"] = df_h4
    htf_swings = collect_htf_swings(htf_dfs, swing_strength=3)

    # 8. Evaluation des OBs du jour J
    print(f"\n  >>> Evaluation des {len(obs_day)} OBs du jour", flush=True)
    trades = []
    rejets = {}
    all_probas = []
    t0 = time.time()

    for i, ob in enumerate(obs_day):
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

        # Simule trade
        setup = r.trade_setup
        lots, risk_usd = compute_position_size(setup.entry_price, setup.stop_loss, asset, balance=BALANCE, risk_pct=RISK_PCT)
        if lots <= 0:
            rejets[f"lots<=0_proba{proba:.2f}"] = rejets.get(f"lots<=0_proba{proba:.2f}", 0) + 1
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
            "direction": ob.direction,
            "score": r.score,
            "proba": proba,
            "outcome": tr.outcome,
            "pnl_usd": tr.pnl_usd,
            "entry": setup.entry_price,
            "sl": setup.stop_loss,
            "tp": setup.take_profit,
        })

    print(f"  Eval termine en {time.time()-t0:.1f}s", flush=True)

    # 9. RECAP
    print(f"\n{'='*70}", flush=True)
    print(f"RECAP JOUR {day_ts.date()} - {asset}", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  OBs evalues : {len(obs_day)}", flush=True)
    print(f"  OBs passe pipeline (atteint ML) : {len(all_probas)}", flush=True)
    print(f"  Trades pris (ML>={ML_THRESHOLD}) : {len(trades)}", flush=True)

    if trades:
        df_t = pd.DataFrame(trades)
        closed = df_t[df_t["outcome"].isin(["WIN", "LOSS"])]
        if len(closed) > 0:
            wr = (closed["outcome"] == "WIN").mean() * 100
            print(f"  WIN  : {(closed['outcome'] == 'WIN').sum()}", flush=True)
            print(f"  LOSS : {(closed['outcome'] == 'LOSS').sum()}", flush=True)
            print(f"  WR   : {wr:.1f}%", flush=True)
        pnl = df_t["pnl_usd"].sum() if "pnl_usd" in df_t else 0
        print(f"  PnL  : {pnl:+.2f} USD", flush=True)
        print(f"\n  Trades :", flush=True)
        for t in trades:
            print(f"    {t['ts']} | {t['direction']:8s} | proba={t['proba']:.3f} | {t['outcome']} | PnL={t['pnl_usd']:+.2f}", flush=True)

    # Distribution probas
    if all_probas:
        import numpy as np
        p = np.array(all_probas)
        print(f"\n  Probas ({len(p)} OBs ML evalues) :", flush=True)
        print(f"    min={p.min():.3f} max={p.max():.3f} mean={p.mean():.3f} median={np.median(p):.3f}", flush=True)
        for thr in [0.3, 0.5, 0.65, 0.70, 0.75, 0.80]:
            n = (p >= thr).sum()
            print(f"    >= {thr:.2f} : {n} ({n/len(p)*100:.1f}%)", flush=True)

    # Top rejets
    print(f"\n  Top rejets :", flush=True)
    for reason, count in sorted(rejets.items(), key=lambda x: -x[1])[:8]:
        print(f"    {count:>4}x : {reason}", flush=True)


if __name__ == "__main__":
    main()
