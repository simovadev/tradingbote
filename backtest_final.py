"""Backtest FINAL multi-asset : 2025-09 -> 2026-05 (~8.5 mois OOS pur).

User specs (2026-05-17) :
- Balance commune compound
- 0-5000€   : risk 20% par trade (push initial)
- 5000€+    : risk 5% par trade (mode sur)
- Fin de mois : retrait de (balance - 5000) si balance > 5000, reset a 5000
- Max 3 trades concurrents
- Cooldown 15 min par actif
- Forex bloque 21h-02h NY (deja dans pipeline)
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


# === CONFIG ===
INITIAL_BALANCE = 100.0
RISK_PCT_AGGRESSIVE = 0.20  # 0-5000€
RISK_PCT_SAFE = 0.05         # 5000€+
THRESHOLD_SAFE_MODE = 5000.0
COOLDOWN_SEC = 15 * 60
MAX_CONCURRENT = 3

ASSETS = ["XAUUSD", "NAS100", "GER40", "BTCUSD", "EURUSD", "GBPUSD", "AUDUSD", "USDJPY"]

# Periode : 2025-09-01 -> 2026-05-17 (OOS pur post-train)
PERIOD_START = pd.Timestamp("2025-09-01", tz="UTC")
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


def detect_setups_for_asset(instrument, start, end):
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


def get_risk_pct(balance):
    """20% si balance < 5000, sinon 5%."""
    return RISK_PCT_AGGRESSIVE if balance < THRESHOLD_SAFE_MODE else RISK_PCT_SAFE


def detect_and_simulate_worker(args):
    """Worker process : detecte les setups d'un actif PUIS simule les trades.

    Retourne une liste de dicts (sims) serialisables pour merge global.
    La simulation ici donne juste outcome+realized_rr ; le balance/risk sera
    recalcule cote main (qui connait la balance globale au moment du trade).
    """
    instrument, start, end = args
    setups = detect_setups_for_asset(instrument, start, end)
    sims = []
    for s in setups:
        setup = s["r"].trade_setup
        ob = s["ob"]
        df_ltf = s["df_ltf"]
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
        exit_ts = sim.exit_ts if sim.exit_ts else s["ts"] + pd.Timedelta(hours=2)

        sims.append({
            "instrument": instrument,
            "ts": s["ts"],
            "exit_ts": exit_ts,
            "direction": setup.direction,
            "entry": float(setup.entry_price),
            "sl": float(setup.stop_loss),
            "tp": float(setup.take_profit),
            "rr": float(setup.rr),
            "score": s["r"].score,
            "ml_proba": float(s["proba"]),
            "killzone": s["r"].killzone_name,
            "outcome": sim.outcome,
            "realized_rr": float(realized_rr),
        })
    return instrument, sims


def main():
    print(f"=== BACKTEST FINAL MULTI-ASSET (OOS pur) ===")
    print(f"Periode : {PERIOD_START.date()} -> {PERIOD_END.date()}")
    print(f"Actifs  : {ASSETS}")
    print(f"Balance initial : {INITIAL_BALANCE}€")
    print(f"MM : 0-5000€ = 20% / 5000€+ = 5%, retrait fin de mois")
    print(f"Max {MAX_CONCURRENT} concurrents, cooldown 15min\n")

    # 1. Detection + simulation PARALLELE (6 workers, user 2026-05-17)
    t0 = time.time()
    all_sims = []
    n_workers = min(6, len(ASSETS))
    print(f"  Detection parallele : {n_workers} workers sur {len(ASSETS)} actifs\n", flush=True)
    tasks = [(asset, PERIOD_START, PERIOD_END) for asset in ASSETS]
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(detect_and_simulate_worker, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            asset_done = futures[fut]
            ts0 = time.time()
            try:
                instrument, sims = fut.result()
                all_sims.extend(sims)
                print(f"  [{instrument}] {len(sims)} trades simules", flush=True)
            except Exception as e:
                print(f"  [{asset_done}] ERREUR : {e}", flush=True)

    print(f"\nTotal trades simules : {len(all_sims)} (en {time.time()-t0:.1f}s)\n")
    all_sims.sort(key=lambda s: s["ts"])

    # 2. Boucle de portefeuille sequentielle (balance commune + cooldown + max concurrent)
    balance = INITIAL_BALANCE
    max_balance_ever = balance
    last_trade_ts = {a: pd.Timestamp(0, tz="UTC") for a in ASSETS}
    active_trades = []
    trades_log = []
    monthly_withdrawals = []
    monthly_balance_start = {}

    current_month = None

    for s in all_sims:
        ts = s["ts"]
        inst = s["instrument"]
        month_str = ts.strftime("%Y-%m")

        # Detection changement de mois -> retrait fin de mois precedent
        if current_month is not None and month_str != current_month:
            if balance > THRESHOLD_SAFE_MODE:
                withdrawn = balance - THRESHOLD_SAFE_MODE
                monthly_withdrawals.append({
                    "month": current_month,
                    "withdrawn": withdrawn,
                    "balance_before": balance,
                })
                balance = THRESHOLD_SAFE_MODE
        if month_str != current_month:
            monthly_balance_start[month_str] = balance
            current_month = month_str

        # Cleanup actifs trades termines
        active_trades = [t for t in active_trades if t["exit_ts"] > ts]

        # Cooldown
        if (ts - last_trade_ts[inst]).total_seconds() < COOLDOWN_SEC:
            continue

        # Max concurrent
        if len(active_trades) >= MAX_CONCURRENT:
            continue

        # Calcul PnL avec balance courante
        realized_rr = s["realized_rr"]
        risk_pct = get_risk_pct(balance)
        risk_amount = balance * risk_pct
        if s["outcome"] == "WIN":
            pnl_eur = risk_amount * realized_rr
        elif s["outcome"] == "LOSS":
            pnl_eur = -risk_amount * abs(realized_rr) if realized_rr != 0 else -risk_amount
        else:
            pnl_eur = 0.0

        balance += pnl_eur
        max_balance_ever = max(max_balance_ever, balance)

        active_trades.append({"exit_ts": s["exit_ts"]})
        last_trade_ts[inst] = ts

        trades_log.append({
            "ts": ts,
            "month": month_str,
            "instrument": inst,
            "direction": s["direction"],
            "rr": s["rr"],
            "score": s["score"],
            "ml_proba": s["ml_proba"],
            "killzone": s["killzone"],
            "outcome": s["outcome"],
            "risk_pct": risk_pct,
            "pnl_eur": pnl_eur,
            "balance": balance,
        })

    # Retrait final si dernier mois
    if current_month is not None and balance > THRESHOLD_SAFE_MODE:
        withdrawn = balance - THRESHOLD_SAFE_MODE
        monthly_withdrawals.append({
            "month": current_month,
            "withdrawn": withdrawn,
            "balance_before": balance,
        })
        balance = THRESHOLD_SAFE_MODE

    # === AFFICHAGE ===
    # Recap mois par mois
    by_month = defaultdict(list)
    for t in trades_log:
        by_month[t["month"]].append(t)

    print("=" * 120)
    print(f"{'MOIS':<10} {'TRADES':<8} {'W':<4} {'L':<4} {'WR':<7} {'BAL DEB':<12} {'BAL FIN':<14} {'PnL MOIS':<14} {'RETRAIT':<12}")
    print("=" * 120)

    total_withdrawn = 0.0
    months_sorted = sorted(by_month.keys())
    for month in months_sorted:
        trades = by_month[month]
        w = sum(1 for t in trades if t["outcome"] == "WIN")
        l = sum(1 for t in trades if t["outcome"] == "LOSS")
        c = w + l
        wr = w / c * 100 if c > 0 else 0
        bal_start = monthly_balance_start.get(month, 0)
        bal_end_avant_retrait = trades[-1]["balance"]
        pnl_mois = bal_end_avant_retrait - bal_start
        # Retrait associe
        retrait = next((wd["withdrawn"] for wd in monthly_withdrawals if wd["month"] == month), 0)
        bal_end_apres_retrait = bal_end_avant_retrait - retrait
        total_withdrawn += retrait
        retrait_str = f"-{retrait:.0f}€" if retrait > 0 else "-"
        print(f"{month:<10} {len(trades):<8} {w:<4} {l:<4} {wr:<6.1f}% "
              f"{bal_start:<11.2f}€ {bal_end_avant_retrait:<13.2f}€ "
              f"{pnl_mois:+13.2f}€ {retrait_str:<12}")

    print("=" * 120)
    print(f"\n=== BILAN GLOBAL ===")
    n_total = len(trades_log)
    n_wins = sum(1 for t in trades_log if t["outcome"] == "WIN")
    n_losses = sum(1 for t in trades_log if t["outcome"] == "LOSS")
    closed = n_wins + n_losses
    wr = n_wins / closed * 100 if closed > 0 else 0
    total_value = balance + total_withdrawn
    perf = (total_value - INITIAL_BALANCE) / INITIAL_BALANCE * 100

    print(f"Periode             : {PERIOD_START.date()} -> {PERIOD_END.date()}")
    print(f"Trades totaux       : {n_total}")
    print(f"Wins / Losses       : {n_wins} / {n_losses}")
    print(f"WR global           : {wr:.1f}%")
    print(f"Balance finale      : {balance:.2f}€ (compte)")
    print(f"Total retire        : {total_withdrawn:.2f}€")
    print(f"VALEUR TOTALE       : {total_value:.2f}€  (compte + retraits)")
    print(f"Performance         : {perf:+.1f}%")
    print(f"Trades/mois moy     : {n_total/len(months_sorted):.1f}")

    # Par actif
    print(f"\n=== PAR ACTIF ===")
    by_asset = defaultdict(list)
    for t in trades_log:
        by_asset[t["instrument"]].append(t)
    for asset in ASSETS:
        trades = by_asset[asset]
        if not trades:
            print(f"  {asset:<8} : 0 trade")
            continue
        w = sum(1 for t in trades if t["outcome"] == "WIN")
        l = sum(1 for t in trades if t["outcome"] == "LOSS")
        c = w + l
        wr_a = w / c * 100 if c > 0 else 0
        print(f"  {asset:<8} : {len(trades):>3} trades ({w}W/{l}L), WR={wr_a:.1f}%")

    # Retraits detail
    print(f"\n=== RETRAITS MENSUELS ===")
    for wd in monthly_withdrawals:
        print(f"  {wd['month']} : retire {wd['withdrawn']:.2f}€ (balance avant : {wd['balance_before']:.2f}€)")

    # Sauvegarde CSV de tous les trades (pour analyse jour par jour)
    df_trades = pd.DataFrame(trades_log)
    csv_path = "c:/Users/Shadow/TradingBot/backtest_final_trades.csv"
    df_trades.to_csv(csv_path, index=False)
    print(f"\nDetail des {len(trades_log)} trades sauvegarde : {csv_path}")


if __name__ == "__main__":
    main()
