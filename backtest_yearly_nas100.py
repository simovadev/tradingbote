"""Backtest exhaustif NAS100 : meme logique que XAUUSD mais avec :
- ml_model_NAS100.pkl
- features NAS100 (sans SMT obligatoire)
- M1 seul (pas M5 pour l'instant - pas de modele NAS100 M5)
- Seuil ML par defaut 0.55 (a tester aussi 0.60 / 0.65)
"""
import sys
sys.path.insert(0, 'c:/Users/Shadow/TradingBot')

import time
import json
import pickle
import pandas as pd

from bot_v2.config import SMT_PAIRS, get_param
from bot_v2.concepts.breaker import detect_breakers
from bot_v2.concepts.daily_bias import build_d1_from_h1
from bot_v2.concepts.fvg import detect_fvg
from bot_v2.concepts.htf_swings import collect_htf_swings
from bot_v2.concepts.liquidity import find_swings
from bot_v2.concepts.order_block import detect_order_blocks
from bot_v2.concepts.structure import detect_structure_breaks, detect_trend
from bot_v2.data_loader import load
from bot_v2.pipeline import evaluate_ob
from bot_v2.backtest import simulate_trade
from bot_v2.trade_setup import compute_position_size, TradeSetup
from bot_v2 import ml_filter


INITIAL_BALANCE = 100.0
RISK_PCT = 0.20
COOLDOWN_SEC = 15 * 60


def load_nas100_model():
    """Charge le model NAS100 specifique."""
    with open('c:/Users/Shadow/TradingBot/bot_v2/ml_model_NAS100.pkl', 'rb') as f:
        model = pickle.load(f)
    features = json.load(open('c:/Users/Shadow/TradingBot/bot_v2/ml_features_NAS100.json'))['features']
    return model, features


def predict_nas100(model, features, r, ob, instrument):
    feats = ml_filter._features_from_result(r, ob, instrument)
    X = pd.DataFrame([[feats.get(f, 0) for f in features]], columns=features)
    return float(model.predict_proba(X)[0, 1])


def backtest_month_nas100(start, end, df_ltf_full, df_htf, df_htf2, df_d1, htf_swings,
                          model, features, balance_start, threshold):
    mask = (df_ltf_full.index >= start) & (df_ltf_full.index <= end)
    df_ltf = df_ltf_full[mask]
    if len(df_ltf) < 100:
        return None

    instrument = "NAS100"
    correlated_dfs = {}
    for corr_name, corr_type in SMT_PAIRS.get(instrument, []):
        try:
            df_c = load(corr_name, "M1")
            mask_c = (df_c.index >= start) & (df_c.index <= end)
            correlated_dfs[corr_name] = (df_c[mask_c], corr_type)
        except Exception:
            continue

    sws = get_param(instrument, "swing_strength_m1", 2)
    obs = detect_order_blocks(df_ltf, swing_strength=sws, max_group_size=2)
    cache = {
        "swings_ltf": find_swings(df_ltf, strength=sws),
        "fvgs_ltf": detect_fvg(df_ltf),
        "breakers_ltf": detect_breakers(df_ltf),
        "obs_htf": detect_order_blocks(df_htf),
    }
    cache["structure_breaks"] = detect_structure_breaks(df_ltf, swings=cache["swings_ltf"])
    cache["htf_trend"] = detect_trend(cache["swings_ltf"], lookback=6)
    if df_htf2 is not None:
        cache["obs_htf2"] = detect_order_blocks(df_htf2)

    # Filtre OB+MSS confirmes
    from bot_v2.concepts.mss_setup import detect_mss_setups, confirm_ob_with_mss
    mss_setups = detect_mss_setups(df_ltf, structure_breaks=cache["structure_breaks"], swings=cache["swings_ltf"])
    obs_confirmed = confirm_ob_with_mss(obs, mss_setups, window_bars=10)

    setups = []
    for ob in obs_confirmed:
        try:
            r = evaluate_ob(
                ob, df_ltf, df_htf, df_d1, instrument,
                ltf_name="M1", htf_name="M15",
                df_htf2=df_htf2, htf2_name="H1",
                correlated_dfs=correlated_dfs,
                htf_swings=htf_swings,
                df_h1=df_htf2, min_score=0, min_quality=0,
                cache=cache,
            )
        except Exception:
            continue
        if r.verdict != "TRADE" or r.trade_setup is None:
            continue
        proba = predict_nas100(model, features, r, ob, instrument)
        if proba < threshold:
            continue
        setups.append((ob.validation_ts, r, ob, proba))

    setups.sort(key=lambda s: s[0])

    balance = balance_start
    last_trade = pd.Timestamp(0, tz="UTC")
    trades = []
    n_skipped = 0

    for val_ts, r, ob, proba in setups:
        if (val_ts - last_trade).total_seconds() < COOLDOWN_SEC:
            n_skipped += 1
            continue
        last_trade = val_ts
        setup = r.trade_setup
        try:
            lots, risk_usd = compute_position_size(
                setup.entry_price, setup.stop_loss, instrument,
                balance=60.0, risk_pct=0.10,
            )
            if lots <= 0:
                continue
            sim_setup = TradeSetup(
                instrument=instrument, direction=setup.direction,
                entry_price=setup.entry_price, stop_loss=setup.stop_loss,
                take_profit=setup.take_profit, rr=setup.rr,
                risk_points=setup.risk_points, reward_points=setup.reward_points,
                risk_usd=risk_usd, reward_usd=risk_usd * setup.rr,
                position_size_lots=lots,
                ob_validation_ts=setup.ob_validation_ts,
                tp_source=setup.tp_source,
            )
            sim = simulate_trade(sim_setup, df_ltf, ob.validation_index + 1)
            realized_rr = float(sim.pnl_usd) / max(risk_usd, 1e-9)

            risk_amount = balance * RISK_PCT
            if sim.outcome == "WIN":
                pnl_eur = risk_amount * realized_rr
                balance += pnl_eur
            elif sim.outcome == "LOSS":
                pnl_eur = -risk_amount * abs(realized_rr) if realized_rr != 0 else -risk_amount
                balance += pnl_eur
            else:
                pnl_eur = 0.0
            trades.append({"ts": val_ts, "outcome": sim.outcome, "pnl_eur": pnl_eur, "balance": balance})
        except Exception:
            continue

    wins = sum(1 for t in trades if t["outcome"] == "WIN")
    losses = sum(1 for t in trades if t["outcome"] == "LOSS")
    no_fill = sum(1 for t in trades if t["outcome"] == "NO_FILL")
    closed = wins + losses
    wr = wins / closed * 100 if closed > 0 else 0

    return {
        "start": start, "end": end,
        "n_trades": len(trades), "n_skipped": n_skipped,
        "wins": wins, "losses": losses, "no_fill": no_fill,
        "wr": wr, "balance_start": balance_start, "balance_end": balance,
    }


def main(threshold=0.55):
    instrument = "NAS100"
    print(f"=== BACKTEST EXHAUSTIF {instrument} (seuil {threshold}) ===\n", flush=True)

    t0 = time.time()
    df_ltf_full = load(instrument, "M1")
    df_htf = load(instrument, "M15")
    df_htf2 = load(instrument, "H1")
    df_d1 = build_d1_from_h1(load(instrument, "H1"))
    htf_dfs_swings = {}
    for tf in ["H1", "H4", "D1"]:
        try:
            htf_dfs_swings[tf] = load(instrument, tf)
        except Exception:
            pass
    htf_swings = collect_htf_swings(htf_dfs_swings, swing_strength=3)
    print(f"Data charge en {time.time()-t0:.1f}s\n", flush=True)

    model, features = load_nas100_model()

    oos_start = pd.Timestamp("2025-06-11", tz="UTC")
    oos_end = pd.Timestamp("2026-05-15", tz="UTC")
    months = []
    cur = oos_start
    while cur + pd.Timedelta(days=30) <= oos_end:
        months.append((cur, cur + pd.Timedelta(days=30)))
        cur = cur + pd.Timedelta(days=30)

    balance = INITIAL_BALANCE
    all_stats = []
    cumul_trades = cumul_wins = cumul_losses = 0
    max_balance = balance
    max_drawdown = 0

    print(f"{'#':<4} {'Periode':<25} {'Trades':<8} {'WR':<8} {'Balance':<14} {'PnL':<10}", flush=True)
    print("-" * 80)

    for idx, (start, end) in enumerate(months, 1):
        stats = backtest_month_nas100(start, end, df_ltf_full, df_htf, df_htf2, df_d1, htf_swings,
                                       model, features, balance, threshold)
        if stats is None:
            continue
        balance = stats["balance_end"]
        max_balance = max(max_balance, balance)
        dd = (max_balance - balance) / max_balance * 100 if max_balance > 0 else 0
        max_drawdown = max(max_drawdown, dd)
        cumul_trades += stats["n_trades"]
        cumul_wins += stats["wins"]
        cumul_losses += stats["losses"]

        pnl = stats["balance_end"] - stats["balance_start"]
        period_str = f"{start.date()} -> {end.date()}"
        print(f"{idx:<4} {period_str:<25} {stats['n_trades']:<8} {stats['wr']:<7.1f}% {stats['balance_end']:<13.2f}€ {pnl:>+10.2f}€", flush=True)
        all_stats.append(stats)

    print("-" * 80)
    closed = cumul_wins + cumul_losses
    wr_global = cumul_wins / closed * 100 if closed > 0 else 0
    perf = (balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100
    pos = sum(1 for s in all_stats if s["balance_end"] > s["balance_start"])
    neg = sum(1 for s in all_stats if s["balance_end"] < s["balance_start"])

    print(f"\n=== BILAN GLOBAL NAS100 (seuil {threshold}) ===")
    print(f"Trades totaux       : {cumul_trades}")
    print(f"Wins / Losses       : {cumul_wins} / {cumul_losses}")
    print(f"WR global           : {wr_global:.1f}%")
    print(f"Balance finale      : {balance:.2f}€")
    print(f"Performance         : {perf:+.1f}%")
    print(f"Mois positifs       : {pos}/{len(all_stats)}")
    print(f"Mois negatifs       : {neg}/{len(all_stats)}")
    print(f"Drawdown max        : -{max_drawdown:.1f}%")


if __name__ == "__main__":
    thr = float(sys.argv[1]) if len(sys.argv) > 1 else 0.55
    main(threshold=thr)
