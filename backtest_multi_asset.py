"""Backtest multi-asset 1 mois : 8 actifs, balance commune, compound 20%.

Decision user 2026-05-17 :
- 1 mois aleatoire post 2025-09 (vrai OOS pour tous)
- Balance commune (compte unique MT5)
- Risk 20% par trade compound
- Max 3 trades concurrents
- Cooldown 15min par actif
- Tie-breaker chronologique
"""
import sys
sys.path.insert(0, 'c:/Users/Shadow/TradingBot')

import time
import json
import pickle
import random
import pandas as pd
from collections import defaultdict

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


# === CONFIG ===
INITIAL_BALANCE = 100.0
RISK_PCT = 0.20
COOLDOWN_SEC = 15 * 60
MAX_CONCURRENT = 3

ASSETS = ["XAUUSD", "NAS100", "GER40", "BTCUSD", "EURUSD", "GBPUSD", "AUDUSD", "USDJPY"]
OOS_START = pd.Timestamp("2025-09-01", tz="UTC")  # vrai OOS pour tous (post-train)


def load_model_for(instrument):
    """Charge model + features pour un actif."""
    model_path = f'c:/Users/Shadow/TradingBot/bot_v2/ml_model_{instrument}.pkl'
    feat_path = f'c:/Users/Shadow/TradingBot/bot_v2/ml_features_{instrument}.json'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    features = json.load(open(feat_path))['features']
    return model, features


def predict_proba(model, features, r, ob, instrument):
    feats = ml_filter._features_from_result(r, ob, instrument)
    X = pd.DataFrame([[feats.get(f, 0) for f in features]], columns=features)
    return float(model.predict_proba(X)[0, 1])


def detect_setups_for_asset(instrument, start, end):
    """Detecte tous les setups d'un actif sur la fenetre. Retourne liste de dicts."""
    try:
        df = load(instrument, "M1")
        df_htf = load(instrument, "M15")
        df_h1 = load(instrument, "H1")
        df_d1 = build_d1_from_h1(df_h1)
        htf_dfs = {}
        for tf in ["H1", "H4", "D1"]:
            try:
                htf_dfs[tf] = load(instrument, tf)
            except Exception:
                pass
        htf_swings = collect_htf_swings(htf_dfs, swing_strength=3)
    except Exception as e:
        print(f"  [{instrument}] data load error: {e}")
        return []

    mask = (df.index >= start) & (df.index <= end)
    df_ltf = df[mask]
    if len(df_ltf) < 100:
        return []

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
    try:
        cache["obs_htf2"] = detect_order_blocks(df_h1)
    except Exception:
        pass

    from bot_v2.concepts.mss_setup import detect_mss_setups, confirm_ob_with_mss
    mss_setups = detect_mss_setups(df_ltf, structure_breaks=cache["structure_breaks"], swings=cache["swings_ltf"])
    obs_confirmed = confirm_ob_with_mss(obs, mss_setups, window_bars=10)

    try:
        model, features = load_model_for(instrument)
    except Exception as e:
        print(f"  [{instrument}] no model: {e}")
        return []
    threshold = ml_filter.ML_THRESHOLDS.get(instrument, 0.55)

    setups = []
    for ob in obs_confirmed:
        try:
            r = evaluate_ob(
                ob, df_ltf, df_htf, df_d1, instrument,
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
        proba = predict_proba(model, features, r, ob, instrument)
        if proba < threshold:
            continue
        setups.append({
            "instrument": instrument,
            "ts": ob.validation_ts,
            "validation_index": ob.validation_index,
            "ob": ob,
            "r": r,
            "proba": proba,
            "df_ltf": df_ltf,
        })

    return setups


def main():
    # Periode : 30 jours aleatoire post OOS_START
    df_ref = load("XAUUSD", "M1")
    end_data = df_ref.index[-1]
    available_days = (end_data - OOS_START).days - 30
    if available_days <= 0:
        print("Fenetre OOS trop courte")
        return

    if len(sys.argv) > 1:
        seed = int(sys.argv[1])
    else:
        seed = random.randint(1, 100000)
    random.seed(seed)
    offset = random.randint(0, available_days)
    start = OOS_START + pd.Timedelta(days=offset)
    end = start + pd.Timedelta(days=30)

    print(f"=== MULTI-ASSET 1 MOIS (seed={seed}) ===")
    print(f"Periode : {start.date()} -> {end.date()}")
    print(f"Actifs  : {ASSETS}")
    print(f"Balance initial : {INITIAL_BALANCE}€, Risk {RISK_PCT*100:.0f}%, Max {MAX_CONCURRENT} concurrents\n")

    # 1. Detecte tous les setups sur tous les actifs
    t0 = time.time()
    all_setups = []
    for asset in ASSETS:
        print(f"  Detection {asset}...", flush=True)
        ts0 = time.time()
        setups = detect_setups_for_asset(asset, start, end)
        print(f"    {len(setups)} setups en {time.time()-ts0:.1f}s")
        all_setups.extend(setups)

    print(f"\nTotal setups detectes : {len(all_setups)} (en {time.time()-t0:.1f}s)\n")

    # 2. Tri chronologique
    all_setups.sort(key=lambda s: s["ts"])

    # 3. Boucle de trading
    balance = INITIAL_BALANCE
    max_balance = balance
    max_dd = 0
    last_trade_ts = {a: pd.Timestamp(0, tz="UTC") for a in ASSETS}
    active_trades = []  # liste de dicts avec exit_ts
    trades_log = []

    for s in all_setups:
        ts = s["ts"]
        inst = s["instrument"]

        # Cleanup active trades termines avant ts
        active_trades = [t for t in active_trades if t["exit_ts"] > ts]

        # Check cooldown
        if (ts - last_trade_ts[inst]).total_seconds() < COOLDOWN_SEC:
            continue

        # Check max concurrent
        if len(active_trades) >= MAX_CONCURRENT:
            continue

        setup = s["r"].trade_setup
        ob = s["ob"]
        df_ltf = s["df_ltf"]

        try:
            lots, risk_usd = compute_position_size(
                setup.entry_price, setup.stop_loss, inst,
                balance=60.0, risk_pct=0.10,
            )
            if lots <= 0:
                continue
            sim_setup = TradeSetup(
                instrument=inst, direction=setup.direction,
                entry_price=setup.entry_price, stop_loss=setup.stop_loss,
                take_profit=setup.take_profit, rr=setup.rr,
                risk_points=setup.risk_points, reward_points=setup.reward_points,
                risk_usd=risk_usd, reward_usd=risk_usd * setup.rr,
                position_size_lots=lots,
                ob_validation_ts=setup.ob_validation_ts,
                tp_source=setup.tp_source,
            )
            sim = simulate_trade(sim_setup, df_ltf, ob.validation_index + 1)
        except Exception as e:
            continue

        if sim.outcome == "NO_FILL":
            continue

        realized_rr = float(sim.pnl_usd) / max(risk_usd, 1e-9)
        risk_amount = balance * RISK_PCT
        if sim.outcome == "WIN":
            pnl_eur = risk_amount * realized_rr
        elif sim.outcome == "LOSS":
            pnl_eur = -risk_amount * abs(realized_rr) if realized_rr != 0 else -risk_amount
        else:
            pnl_eur = 0.0

        balance += pnl_eur
        max_balance = max(max_balance, balance)
        dd = (max_balance - balance) / max_balance * 100 if max_balance > 0 else 0
        max_dd = max(max_dd, dd)

        exit_ts = sim.exit_ts if sim.exit_ts else ts + pd.Timedelta(hours=2)
        active_trades.append({"exit_ts": exit_ts})
        last_trade_ts[inst] = ts

        trades_log.append({
            "ts": ts,
            "exit_ts": exit_ts,
            "instrument": inst,
            "direction": setup.direction,
            "entry": setup.entry_price,
            "sl": setup.stop_loss,
            "tp": setup.take_profit,
            "rr": setup.rr,
            "score": s["r"].score,
            "ml_proba": s["proba"],
            "killzone": s["r"].killzone_name,
            "outcome": sim.outcome,
            "pnl_eur": pnl_eur,
            "balance": balance,
        })

    # 4. Affichage tableau jour par jour
    print("\n" + "="*120)
    print(f"{'DATE':<12} {'TIME':<6} {'ASSET':<8} {'DIR':<6} {'RR':<5} {'ML':<6} {'KZ':<12} {'OUTCOME':<8} {'PnL€':<10} {'BALANCE€':<12}")
    print("="*120)

    by_day = defaultdict(list)
    for t in trades_log:
        day = t["ts"].date()
        by_day[day].append(t)

    days_sorted = sorted(by_day.keys())
    for day in days_sorted:
        for t in by_day[day]:
            time_str = t["ts"].strftime("%H:%M")
            pnl_str = f"{t['pnl_eur']:+.2f}€"
            print(f"{str(day):<12} {time_str:<6} {t['instrument']:<8} {t['direction'][:4].upper():<6} "
                  f"{t['rr']:<5.2f} {t['ml_proba']:<6.3f} {(t['killzone'] or 'None'):<12} "
                  f"{t['outcome']:<8} {pnl_str:<10} {t['balance']:<12.2f}")
        # ligne de separation par jour
        last_balance = by_day[day][-1]["balance"]
        n_trades_day = len(by_day[day])
        n_wins_day = sum(1 for t in by_day[day] if t["outcome"] == "WIN")
        print(f"  -> {day} : {n_trades_day} trades, {n_wins_day}W, balance fin de journee = {last_balance:.2f}€\n")

    # 5. Bilan global
    n_total = len(trades_log)
    n_wins = sum(1 for t in trades_log if t["outcome"] == "WIN")
    n_losses = sum(1 for t in trades_log if t["outcome"] == "LOSS")
    closed = n_wins + n_losses
    wr = n_wins / closed * 100 if closed > 0 else 0
    perf = (balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100

    print("="*120)
    print(f"\n=== BILAN GLOBAL ({start.date()} -> {end.date()}) ===")
    print(f"Trades totaux       : {n_total}")
    print(f"Wins / Losses       : {n_wins} / {n_losses}")
    print(f"WR global           : {wr:.1f}%")
    print(f"Balance finale      : {balance:.2f}€")
    print(f"Performance         : {perf:+.1f}%")
    print(f"Drawdown max        : -{max_dd:.1f}%")
    print()

    # Par actif
    print("=== PAR ACTIF ===")
    by_asset = defaultdict(list)
    for t in trades_log:
        by_asset[t["instrument"]].append(t)
    for asset in ASSETS:
        trades = by_asset[asset]
        if not trades:
            continue
        w = sum(1 for t in trades if t["outcome"] == "WIN")
        l = sum(1 for t in trades if t["outcome"] == "LOSS")
        c = w + l
        wr_a = w / c * 100 if c > 0 else 0
        pnl_a = sum(t["pnl_eur"] for t in trades)
        print(f"  {asset:<8} : {len(trades):>3} trades ({w}W/{l}L), WR={wr_a:.1f}%, PnL={pnl_a:+.2f}€")


if __name__ == "__main__":
    main()
