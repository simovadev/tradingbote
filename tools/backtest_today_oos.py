"""OOS rigoureux sur la session 17h-23h30 d'aujourd'hui.

Recupere les bougies M1 fraiches via MT5 (puisque les parquets sur disque
s'arretent a 13h51 UTC), lance le pipeline EXACT du bot (max_group_size=5,
1 OB = 1 evaluation a validation, V8 models) et compte les trades qui
auraient ete pris au seuil 0.70.

Compare avec ce que le live a fait reellement.
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

import pandas as pd
import MetaTrader5 as mt5

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bot_v2.config import SMT_PAIRS, get_param
from bot_v2.live_runner_v2 import LIVE_ASSETS
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
from bot_v2.live_runner_v2 import load_model
from bot_v2.mt5_executor import to_broker_symbol


N_BARS_M1 = 88000  # ~3 mois


def get_m1_from_mt5(symbol):
    """Recupere 88k bougies M1 fraiches via MT5."""
    bs = to_broker_symbol(symbol)
    rates = mt5.copy_rates_from_pos(bs, mt5.TIMEFRAME_M1, 0, N_BARS_M1)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    # Compense l'offset broker UTC+3
    df["time"] = pd.to_datetime(df["time"] - 3*3600, unit="s", utc=True)
    df = df.set_index("time").sort_index()
    df = df.rename(columns={"tick_volume": "volume"})
    return df[["open","high","low","close","volume"]]


def analyse_asset(asset, start_ts, end_ts, threshold=0.70):
    """OOS sur [start_ts, end_ts] pour un actif. Retourne liste de trades."""
    df_m1 = get_m1_from_mt5(asset)
    if df_m1 is None or len(df_m1) < 1000:
        return []

    # Resamples HTF
    def resample(rule, n):
        out = df_m1.resample(rule, label="left", closed="left").agg({
            "open":"first","high":"max","low":"min","close":"last","volume":"sum"
        }).dropna()
        return out.tail(n)

    df_m15 = resample("15min", 11000)
    df_h1 = resample("1h", 2800)
    df_h4 = resample("4h", 500)
    df_d1 = resample("1D", 120)
    if len(df_d1) < 10:
        df_d1 = build_d1_from_h1(df_h1)

    htf_swings = collect_htf_swings({"H1":df_h1,"H4":df_h4,"D1":df_d1}, swing_strength=3)

    # SMT
    correlated_dfs = {}
    for cn, ct in SMT_PAIRS.get(asset, []):
        dc = get_m1_from_mt5(cn)
        if dc is not None:
            correlated_dfs[cn] = (dc, ct)

    sws = get_param(asset, "swing_strength_m1", 2)
    # max_group_size DEFAUT = 5, ALIGNE OOS
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
        df_m1, structure_breaks=cache["structure_breaks"],
        swings=cache["swings_ltf"], fvgs=cache["fvgs_ltf"]
    )

    # OB dans la fenetre 17h-23h30
    obs_window = [
        ob for ob in obs
        if start_ts <= df_m1.index[ob.validation_index] <= end_ts
    ]

    loaded = load_model(asset)
    if loaded is None:
        return []
    model, features = loaded

    trades = []
    for ob in obs_window:
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
        except Exception:
            continue
        if r.verdict != "TRADE" or r.trade_setup is None:
            continue

        feat = ml_filter._features_from_result(
            r, ob, asset, df_ltf=df_m1, df_d1=df_d1, mss_setups=mss_setups
        )
        X = pd.DataFrame([[feat.get(f, 0) for f in features]], columns=features)
        proba = float(model.predict_proba(X)[0, 1])

        trades.append({
            "ts": ts_val, "direction": ob.direction,
            "proba": proba, "score": int(r.score),
            "entry": float(r.trade_setup.entry_price),
            "passed": proba >= threshold,
        })

    return trades


def main():
    # Fenetre : 17h-23h30 Paris d'aujourd'hui = 15h-21h30 UTC
    today = datetime.now(timezone.utc).date()
    start_ts = pd.Timestamp(f"{today} 15:00:00", tz="UTC")
    end_ts = pd.Timestamp(f"{today} 21:30:00", tz="UTC")

    print(f"OOS WINDOW : {start_ts} -> {end_ts} UTC")
    print(f"            ({start_ts.tz_convert('Europe/Paris')} -> {end_ts.tz_convert('Europe/Paris')} Paris)")
    print()

    if not mt5.initialize():
        print(f"MT5 init KO: {mt5.last_error()}")
        return

    all_trades = []
    by_asset_summary = []
    for asset in LIVE_ASSETS:
        try:
            trades = analyse_asset(asset, start_ts, end_ts, threshold=0.70)
        except Exception as e:
            print(f"  {asset:8s} ERROR : {e}")
            continue
        n_total = len(trades)
        n_passed = sum(1 for t in trades if t["passed"])
        max_p = max((t["proba"] for t in trades), default=0)
        by_asset_summary.append((asset, n_total, n_passed, max_p))
        all_trades.extend(trades)
        print(f"  {asset:8s} : {n_total:>4} candidats | {n_passed:>2} >= 0.70 | max proba {max_p:.3f}")

    print()
    print("=" * 60)
    n_total = len(all_trades)
    n_passed_total = sum(1 for t in all_trades if t["passed"])
    print(f"TOTAL : {n_total} candidats sur 14 actifs, {n_passed_total} >= 0.70")
    print()

    if n_passed_total > 0:
        print(f"=== Trades qui auraient ete pris (proba >= 0.70) ===")
        for t in sorted(all_trades, key=lambda x: x["ts"]):
            if t["passed"]:
                print(f"  {t['ts']} | {t['direction']:8s} | proba={t['proba']:.3f} score={t['score']:>3} entry={t['entry']:.5f}")
    else:
        print("Aucun trade >= 0.70 sur la fenetre.")

    print()
    # Distribution probas
    probas = [t["proba"] for t in all_trades]
    if probas:
        from collections import Counter
        bins = [(0.0,0.3),(0.3,0.5),(0.5,0.6),(0.6,0.65),(0.65,0.70),(0.70,0.75),(0.75,0.80),(0.80,1.01)]
        print(f"Distribution probas :")
        for lo,hi in bins:
            n = sum(1 for p in probas if lo<=p<hi)
            print(f"  [{lo:.2f}-{hi:.2f}) : {n}")

    mt5.shutdown()


if __name__ == "__main__":
    main()
