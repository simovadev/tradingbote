"""Reproduit simulate_last_24h() du bot en utilisant data_vantage/ (PAS MT5).

But : verifier sur ton PC local quels trades le bot AURAIT pris sur la
derniere journee disponible dans data_vantage/. Si 0 trade aussi en
backtest -> c'est la periode (heures creuses). Si plusieurs trades
ratoeient passes -> bug entre live et backtest.

Usage : ./venv/Scripts/python.exe backtest_last_24h_local.py
"""
import sys
from pathlib import Path
from collections import defaultdict, Counter

import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from bot_v2.config import SMT_PAIRS, get_param
from bot_v2.concepts.killzones import killzone_at
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

DATA = ROOT / "data_vantage"


def load_asset_dfs(asset, end_ts):
    """Charge M1 + resample M15/H1/H4/D1 depuis data_vantage/, finissant a end_ts."""
    pq = DATA / f"{asset}_M1.parquet"
    if not pq.exists():
        return None
    df = pd.read_parquet(pq)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df = df[df.index <= end_ts]
    if len(df) < 1000:
        return None

    df_m1 = df.tail(N_BARS_M1)

    def resample(rule, n):
        out = df_m1.resample(rule, label="left", closed="left").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna()
        return out.tail(n)

    return {
        "df_m1": df_m1,
        "df_m15": resample("15min", N_BARS_M15),
        "df_h1": resample("1h", N_BARS_H1),
        "df_h4": resample("4h", 500),
        "df_d1": resample("1D", N_BARS_D1),
    }


def simulate_asset(asset, end_ts, cutoff_24h, balance=154.96):
    """Reproduit le pipeline pour un actif. Retourne (setups_24h, trades_passes, rejets, latest_ts)."""
    dfs = load_asset_dfs(asset, end_ts)
    if dfs is None:
        return 0, [], {}, None

    df_m1 = dfs["df_m1"]
    df_m15 = dfs["df_m15"]
    df_h1 = dfs["df_h1"]
    df_h4 = dfs["df_h4"]
    df_d1 = dfs["df_d1"]
    if df_d1 is None or len(df_d1) < 10:
        df_d1 = build_d1_from_h1(df_h1)

    htf_dfs = {"H1": df_h1, "D1": df_d1}
    if df_h4 is not None and len(df_h4) > 0:
        htf_dfs["H4"] = df_h4
    htf_swings = collect_htf_swings(htf_dfs, swing_strength=3)

    # SMT
    correlated_dfs = {}
    for corr_name, corr_type in SMT_PAIRS.get(asset, []):
        cd = load_asset_dfs(corr_name, end_ts)
        if cd is not None:
            correlated_dfs[corr_name] = (cd["df_m1"], corr_type)

    # Detection
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

    obs_24h = [ob for ob in obs if df_m1.index[ob.validation_index] >= cutoff_24h]

    loaded = load_model(asset)
    if loaded is None:
        return len(obs_24h), [], {"no_model": 1}, df_m1.index[-1]
    model, features = loaded
    threshold = ml_filter.get_dynamic_threshold(asset, balance)

    rejets = {}
    trades_passes = []

    for ob in obs_24h:
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
            rejets["evaluate_exception"] = rejets.get("evaluate_exception", 0) + 1
            continue

        if r.verdict != "TRADE" or r.trade_setup is None:
            reason = r.rejection_reason or "no_trade"
            if "displacement" in reason.lower():
                reason = "displacement_faible"
            elif "killzone" in reason.lower():
                reason = "killzone_hors"
            elif "accumulation" in reason.lower():
                reason = "phase_accumulation"
            elif "Forex bloque" in reason:
                reason = "forex_overnight_us"
            rejets[reason] = rejets.get(reason, 0) + 1
            continue

        proba = predict_proba(
            model, features, r, ob, asset,
            df_ltf=df_m1, df_d1=df_d1, mss_setups=mss_setups,
        )
        if proba < threshold:
            rejets[f"ml_below_{threshold:.2f}"] = rejets.get(f"ml_below_{threshold:.2f}", 0) + 1
            # Stocker les probas elevees rejetees (proche du seuil)
            if proba > 0.5:
                rejets.setdefault("_high_proba_rejected", []).append(
                    (df_m1.index[ob.validation_index], proba, ob.direction)
                )
            continue

        kz = killzone_at(df_m1.index[ob.validation_index]) or "?"
        trades_passes.append({
            "asset": asset,
            "ts": df_m1.index[ob.validation_index],
            "direction": ob.direction,
            "proba": float(proba),
            "killzone": kz,
            "entry": float(r.trade_setup.entry_price),
            "sl": float(r.trade_setup.stop_loss),
            "tp": float(r.trade_setup.take_profit),
            "rr": float(r.trade_setup.rr),
            "score": int(r.score),
        })

    return len(obs_24h), trades_passes, rejets, df_m1.index[-1]


def main():
    # Find the latest timestamp available across all assets
    latest = None
    for asset in LIVE_ASSETS:
        pq = DATA / f"{asset}_M1.parquet"
        if pq.exists():
            df = pd.read_parquet(pq)
            ts = df.index[-1]
            if ts.tz is None:
                ts = ts.tz_localize("UTC")
            if latest is None or ts < latest:
                latest = ts

    if latest is None:
        print("Aucun parquet trouve dans data_vantage/")
        return

    end_ts = latest
    cutoff_24h = end_ts - pd.Timedelta(hours=24)
    print(f"=== Backtest 24h LOCAL (data_vantage/) ===")
    print(f"Periode : {cutoff_24h} -> {end_ts} (UTC)")
    print(f"        = {cutoff_24h.tz_convert('Europe/Paris')} -> {end_ts.tz_convert('Europe/Paris')} (Paris)")
    print()

    total_setups = 0
    all_trades = []
    all_high_probas = []

    for asset in LIVE_ASSETS:
        n_setups, trades, rejets, last_ts = simulate_asset(asset, end_ts, cutoff_24h)
        total_setups += n_setups
        all_trades.extend(trades)
        # Collecte les "high probas rejetees" pour analyse
        if "_high_proba_rejected" in rejets:
            for ts, p, d in rejets.pop("_high_proba_rejected"):
                all_high_probas.append((asset, ts, p, d))
        passes = len(trades)
        # Tronque les rejets affiches
        top_rejets = dict(sorted(rejets.items(), key=lambda x: -x[1])[:3])
        print(f"  {asset:10s} setups={n_setups:>4} passes={passes:>2}  rejets={top_rejets}")

    print()
    print(f"TOTAL : {total_setups} setups detectes en 24h -> {len(all_trades)} auraient ete trades (ML >= 0.75)")
    print()

    if all_trades:
        all_trades.sort(key=lambda t: t["ts"])
        print("=== Liste des trades qui auraient ete pris ===")
        for t in all_trades:
            print(
                f"  {t['ts']} | {t['asset']:8s} | {t['direction']:8s} | kz={t['killzone']:15s} | "
                f"ML={t['proba']:.3f} | RR={t['rr']:.2f} | entry={t['entry']:.5f}"
            )
    else:
        print("Aucun trade qualifie (ML >= 0.75) sur cette periode.")

    if all_high_probas:
        all_high_probas.sort(key=lambda x: -x[2])
        print()
        print(f"=== {len(all_high_probas)} setups REJETES avec proba > 0.5 (proches du seuil) ===")
        for asset, ts, p, d in all_high_probas[:20]:
            print(f"  {ts} | {asset:8s} | {d:8s} | ML={p:.3f}")


if __name__ == "__main__":
    main()
