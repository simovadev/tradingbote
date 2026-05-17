"""Backtest demo de la semaine derniere : 10-17 mai 2026.

User specs : 100€ initial, 20% risk, tous les actifs.
Simule ce qu'aurait fait le bot live sur la semaine ecoulee.
"""
import sys
sys.path.insert(0, 'c:/Users/Shadow/TradingBot')

import time
import json
import pickle
import pandas as pd
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

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
MAX_CONCURRENT = 3

ASSETS = ["XAUUSD", "NAS100", "GER40", "BTCUSD", "EURUSD", "GBPUSD", "AUDUSD", "USDJPY"]

PERIOD_START = pd.Timestamp("2026-05-10", tz="UTC")
PERIOD_END = pd.Timestamp("2026-05-17", tz="UTC")


def load_model_for(instrument):
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


def detect_and_simulate_worker(args):
    instrument, start, end = args
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
    except Exception:
        return instrument, []

    mask = (df.index >= start) & (df.index <= end)
    df_ltf = df[mask]
    if len(df_ltf) < 100:
        return instrument, []

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
    except Exception:
        return instrument, []
    threshold = ml_filter.ML_THRESHOLDS.get(instrument, 0.55)

    sims = []
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
        except Exception:
            continue

        if sim.outcome == "NO_FILL":
            continue

        realized_rr = float(sim.pnl_usd) / max(risk_usd, 1e-9)
        exit_ts = sim.exit_ts if sim.exit_ts else ob.validation_ts + pd.Timedelta(hours=2)

        sims.append({
            "instrument": instrument,
            "ts": ob.validation_ts,
            "exit_ts": exit_ts,
            "direction": setup.direction,
            "entry": float(setup.entry_price),
            "sl": float(setup.stop_loss),
            "tp": float(setup.take_profit),
            "rr": float(setup.rr),
            "score": r.score,
            "ml_proba": float(proba),
            "killzone": r.killzone_name,
            "outcome": sim.outcome,
            "realized_rr": float(realized_rr),
        })
    return instrument, sims


def main():
    print(f"=== BACKTEST DEMO SEMAINE 10-17 MAI 2026 ===")
    print(f"Periode  : {PERIOD_START.date()} -> {PERIOD_END.date()}")
    print(f"Capital  : {INITIAL_BALANCE}€")
    print(f"Risk     : {RISK_PCT*100:.0f}% compound")
    print(f"Max conc : {MAX_CONCURRENT}, cooldown 15min\n")

    t0 = time.time()
    all_sims = []
    tasks = [(asset, PERIOD_START, PERIOD_END) for asset in ASSETS]
    with ProcessPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(detect_and_simulate_worker, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            asset_done = futures[fut]
            try:
                instrument, sims = fut.result()
                all_sims.extend(sims)
                print(f"  [{instrument}] {len(sims)} trades simules", flush=True)
            except Exception as e:
                print(f"  [{asset_done}] ERREUR : {e}", flush=True)

    print(f"\nTotal trades simules : {len(all_sims)} en {time.time()-t0:.1f}s\n")
    all_sims.sort(key=lambda s: s["ts"])

    balance = INITIAL_BALANCE
    last_trade_ts = {a: pd.Timestamp(0, tz="UTC") for a in ASSETS}
    active_trades = []
    trades_log = []

    for s in all_sims:
        ts = s["ts"]
        inst = s["instrument"]

        active_trades = [t for t in active_trades if t["exit_ts"] > ts]

        if (ts - last_trade_ts[inst]).total_seconds() < COOLDOWN_SEC:
            continue
        if len(active_trades) >= MAX_CONCURRENT:
            continue

        realized_rr = s["realized_rr"]
        risk_amount = balance * RISK_PCT
        if s["outcome"] == "WIN":
            pnl = risk_amount * realized_rr
        elif s["outcome"] == "LOSS":
            pnl = -risk_amount * abs(realized_rr) if realized_rr != 0 else -risk_amount
        else:
            pnl = 0.0

        balance += pnl
        active_trades.append({"exit_ts": s["exit_ts"]})
        last_trade_ts[inst] = ts

        trades_log.append({
            "ts": ts,
            "instrument": inst,
            "direction": s["direction"],
            "rr": s["rr"],
            "score": s["score"],
            "ml_proba": s["ml_proba"],
            "killzone": s["killzone"],
            "outcome": s["outcome"],
            "pnl": pnl,
            "balance": balance,
        })

    # Affichage
    print("=" * 110)
    print(f"{'DATE':<12} {'HEURE':<7} {'ASSET':<8} {'DIR':<6} {'RR':<5} {'ML':<6} {'KZ':<13} {'OUTCOME':<8} {'PnL':<10} {'BAL':<10}")
    print("=" * 110)
    by_day = defaultdict(list)
    for t in trades_log:
        by_day[t["ts"].date()].append(t)
    for day in sorted(by_day.keys()):
        for t in by_day[day]:
            print(f"{str(day):<12} {t['ts'].strftime('%H:%M'):<7} {t['instrument']:<8} "
                  f"{t['direction'][:4].upper():<6} {t['rr']:<5.2f} {t['ml_proba']:<6.3f} "
                  f"{(t['killzone'] or 'None'):<13} {t['outcome']:<8} {t['pnl']:+9.2f}€ {t['balance']:<10.2f}")
        nw = sum(1 for t in by_day[day] if t["outcome"] == "WIN")
        nl = sum(1 for t in by_day[day] if t["outcome"] == "LOSS")
        bal = by_day[day][-1]["balance"]
        print(f"  -> {day} : {len(by_day[day])} trades ({nw}W/{nl}L) | balance fin = {bal:.2f}€\n")

    print("=" * 110)
    n = len(trades_log)
    w = sum(1 for t in trades_log if t["outcome"] == "WIN")
    l = sum(1 for t in trades_log if t["outcome"] == "LOSS")
    wr = w / (w + l) * 100 if (w + l) > 0 else 0
    perf = (balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100
    print(f"\n=== BILAN SEMAINE ===")
    print(f"Trades totaux  : {n}")
    print(f"WR             : {wr:.1f}% ({w}W / {l}L)")
    print(f"Balance finale : {balance:.2f}€")
    print(f"Performance    : {perf:+.1f}%")

    print(f"\n=== PAR ACTIF ===")
    by_asset = defaultdict(list)
    for t in trades_log:
        by_asset[t["instrument"]].append(t)
    for a in ASSETS:
        trades = by_asset[a]
        if not trades:
            print(f"  {a:<8} : 0 trade")
            continue
        w_a = sum(1 for t in trades if t["outcome"] == "WIN")
        l_a = sum(1 for t in trades if t["outcome"] == "LOSS")
        wr_a = w_a / (w_a + l_a) * 100 if (w_a + l_a) > 0 else 0
        pnl_a = sum(t["pnl"] for t in trades)
        print(f"  {a:<8} : {len(trades):>3} trades ({w_a}W/{l_a}L) WR={wr_a:.1f}% PnL={pnl_a:+.2f}€")


if __name__ == "__main__":
    main()
