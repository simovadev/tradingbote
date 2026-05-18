"""Backtest 3 derniers mois (2026-02-18 -> 2026-05-18) - 15 actifs Phase 4.

Logs detailles : heures de trades, trades/jour, WR par actif, killzones.
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


# 15 actifs Phase 4 (8 anciens + 7 nouveaux, NZDUSD ecarte)
ASSETS = [
    "XAUUSD", "NAS100", "GER40", "BTCUSD", "EURUSD", "GBPUSD", "AUDUSD", "USDJPY",
    "SP500", "DJ30", "UK100", "FRA40", "JP225", "USDCAD", "USDCHF",
]
PERIOD_START = pd.Timestamp("2026-05-04", tz="UTC")
PERIOD_END = pd.Timestamp("2026-05-18", tz="UTC")


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
    except Exception as e:
        return instrument, [], 0, 0, 0

    mask = (df.index >= start) & (df.index <= end)
    df_ltf = df[mask]
    if len(df_ltf) < 100:
        return instrument, [], 0, 0, 0

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
        return instrument, [], 0, 0, len(obs_confirmed)
    threshold = ml_filter.ML_THRESHOLDS.get(instrument, 0.55)

    n_vizion_rejets = 0
    n_ml_rejets = 0
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
            n_vizion_rejets += 1
            continue
        proba = predict_proba(model, features, r, ob, instrument)
        if proba < threshold:
            n_ml_rejets += 1
            continue

        setup = r.trade_setup
        try:
            lots, risk_usd = compute_position_size(
                setup.entry_price, setup.stop_loss, instrument,
                balance=60.0, risk_pct=0.30,
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

        sims.append({
            "instrument": instrument,
            "ts": ob.validation_ts,
            "direction": setup.direction,
            "rr": setup.rr,
            "score": r.score,
            "ml_proba": float(proba),
            "killzone": r.killzone_name,
            "outcome": sim.outcome,
            "risk_usd": float(risk_usd),
            "pnl_usd": float(sim.pnl_usd),
        })

    return instrument, sims, n_vizion_rejets, n_ml_rejets, len(obs_confirmed)


def main():
    print(f"=== BACKTEST 3 MOIS ({PERIOD_START.date()} -> {PERIOD_END.date()}) - 15 ACTIFS ===\n", flush=True)

    t0 = time.time()
    all_sims = []
    detail = {}
    tasks = [(asset, PERIOD_START, PERIOD_END) for asset in ASSETS]
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(detect_and_simulate_worker, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            asset_done = futures[fut]
            try:
                instrument, sims, vizion_rej, ml_rej, total_ob = fut.result()
                all_sims.extend(sims)
                detail[instrument] = {
                    "trades": len(sims),
                    "vizion_rej": vizion_rej,
                    "ml_rej": ml_rej,
                    "total_ob": total_ob,
                }
                print(f"  [{instrument:7s}] {total_ob:>4} OB+MSS | Vizion rej={vizion_rej:>4} | ML rej={ml_rej:>3} | TRADES={len(sims)}", flush=True)
            except Exception as e:
                print(f"  [{asset_done}] ERREUR : {e}", flush=True)

    print(f"\nTotal trades : {len(all_sims)} en {time.time()-t0:.1f}s")
    duree_jours = (PERIOD_END - PERIOD_START).days
    print(f"Periode      : {duree_jours} jours = {len(all_sims)/duree_jours:.2f} trades/jour moyen\n")

    all_sims.sort(key=lambda s: s["ts"])

    # === RESUMA GLOBAL ===
    wins = sum(1 for s in all_sims if s["outcome"] == "WIN")
    losses = sum(1 for s in all_sims if s["outcome"] == "LOSS")
    wr = (wins / max(wins+losses, 1)) * 100
    pnl_total = sum(s["pnl_usd"] for s in all_sims)
    print(f"=== RESUME GLOBAL ===")
    print(f"  WIN  : {wins}")
    print(f"  LOSS : {losses}")
    print(f"  WR   : {wr:.1f}%")
    print(f"  PnL  : {pnl_total:+.2f} USD (risque base 60 USD x 30%)")
    print()

    # === PAR ACTIF ===
    print("=== PAR ACTIF ===")
    print(f"  {'Actif':<8} {'OB':>4} {'V.rej':>5} {'ML.rej':>6} {'Trades':>6} {'W':>3} {'L':>3} {'WR':>6} {'PnL':>10}")
    by_inst = defaultdict(list)
    for s in all_sims:
        by_inst[s["instrument"]].append(s)
    for asset in ASSETS:
        d = detail.get(asset, {})
        if not d:
            continue
        sims_a = by_inst[asset]
        w = sum(1 for t in sims_a if t["outcome"] == "WIN")
        l = sum(1 for t in sims_a if t["outcome"] == "LOSS")
        wr_a = (w / max(w+l, 1)) * 100
        pnl_a = sum(t["pnl_usd"] for t in sims_a)
        print(f"  {asset:<8} {d['total_ob']:>4} {d['vizion_rej']:>5} {d['ml_rej']:>6} {d['trades']:>6} {w:>3} {l:>3} {wr_a:>5.1f}% {pnl_a:>+10.2f}")
    print()

    # === TRADES PAR JOUR ===
    by_day = defaultdict(list)
    for s in all_sims:
        by_day[s["ts"].date()].append(s)
    print("=== TRADES PAR JOUR ===")
    for day in sorted(by_day.keys()):
        trades = by_day[day]
        w = sum(1 for t in trades if t["outcome"] == "WIN")
        l = sum(1 for t in trades if t["outcome"] == "LOSS")
        pnl_d = sum(t["pnl_usd"] for t in trades)
        print(f"  {day} ({day.strftime('%a'):<3}) : {len(trades):>2} trades ({w}W/{l}L)  PnL={pnl_d:+.2f}")
    print()

    # === HEURES UTC ===
    print("=== TRADES PAR HEURE (UTC -> Paris) ===")
    hc = defaultdict(int)
    hc_win = defaultdict(int)
    for s in all_sims:
        hc[s["ts"].hour] += 1
        if s["outcome"] == "WIN":
            hc_win[s["ts"].hour] += 1
    for h in sorted(hc.keys()):
        paris_h = (h + 2) % 24  # heure d'ete CEST
        n = hc[h]
        w = hc_win[h]
        wr_h = (w / n) * 100 if n else 0
        bar = "*" * n
        print(f"  {h:02d}h UTC = {paris_h:02d}h Paris : {n:>3} trades ({w}W) WR={wr_h:>5.1f}%  {bar}")
    print()

    # === PAR KILLZONE ===
    print("=== PAR KILLZONE ===")
    kz_count = defaultdict(lambda: {"n": 0, "w": 0, "l": 0, "pnl": 0.0})
    for s in all_sims:
        k = s["killzone"] or "None"
        kz_count[k]["n"] += 1
        if s["outcome"] == "WIN":
            kz_count[k]["w"] += 1
        elif s["outcome"] == "LOSS":
            kz_count[k]["l"] += 1
        kz_count[k]["pnl"] += s["pnl_usd"]
    for k, d in sorted(kz_count.items(), key=lambda x: -x[1]["n"]):
        wr_k = (d["w"] / max(d["w"]+d["l"], 1)) * 100
        print(f"  {k:<12} : {d['n']:>3} trades ({d['w']}W/{d['l']}L) WR={wr_k:>5.1f}%  PnL={d['pnl']:+.2f}")
    print()

    # === DETAIL TOUS LES TRADES ===
    print("=== DETAIL TRADES (chronologique) ===")
    for s in all_sims:
        paris_h = (s["ts"].hour + 2) % 24
        kz = s["killzone"] or "None"
        print(f"  {s['ts'].strftime('%Y-%m-%d %H:%M')} UTC ({paris_h:02d}h{s['ts'].minute:02d} Paris) | {s['instrument']:<7} {s['direction'][:4].upper():4} | KZ={kz:<10} | {s['outcome']:5} | ML={s['ml_proba']:.3f} | PnL={s['pnl_usd']:+8.2f}")


if __name__ == "__main__":
    main()
