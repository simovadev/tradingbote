"""Backtest exhaustif : balayage mois par mois sur 1 an OOS.

Reproduit EXACTEMENT la logique du dashboard ml_dashboard /api/replay_ml :
- Pipeline Vizion + ML filter v7 (M1) + M5 si dispo
- Cooldown 15min par TF
- TP fixe dynamique (HTF swing capé RR=3)
- Money management compose : 100 EUR initial, 20% risk par trade

Sortie : tableau mois par mois + stats globales.
"""
import sys
sys.path.insert(0, 'c:/Users/Shadow/TradingBot')

import time
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
from bot_v2.concepts.mss_setup import detect_mss_setups, confirm_ob_with_mss
from bot_v2.data_loader import load
from bot_v2.pipeline import evaluate_ob
from bot_v2.backtest import simulate_trade
from bot_v2.trade_setup import compute_position_size, TradeSetup
from bot_v2 import ml_filter


INITIAL_BALANCE = 100.0
RISK_PCT = 0.20
COOLDOWN_SEC = 15 * 60


def backtest_month(instrument, start, end, df_ltf_full, df_htf, df_htf2, df_d1, htf_swings,
                   df_m5_full=None, df_m5_htf=None, df_m5_htf2=None,
                   balance_start=100.0, threshold_m1=0.50, threshold_m5=0.50):
    """Backtest 1 mois. Retourne dict avec stats."""
    mask = (df_ltf_full.index >= start) & (df_ltf_full.index <= end)
    df_ltf = df_ltf_full[mask]
    if len(df_ltf) < 100:
        return None

    correlated_dfs = {}
    for corr_name, corr_type in SMT_PAIRS.get(instrument, []):
        try:
            df_c = load(corr_name, "M1")
            mask_c = (df_c.index >= start) & (df_c.index <= end)
            correlated_dfs[corr_name] = (df_c[mask_c], corr_type)
        except Exception:
            continue

    # === SCAN M1 ===
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

    # Collecte tous les setups M1
    setups_m1 = []
    for ob in obs:
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
        take, proba = ml_filter.should_take(r, ob, instrument)
        if not take:
            continue
        if proba < threshold_m1:
            continue
        setups_m1.append((ob.validation_ts, "M1", r, ob, proba))

    # === SCAN M5 (si modele dispo) ===
    setups_m5 = []
    if df_m5_full is not None and ml_filter.load_model_for_tf("M5") is not None:
        mask_m5 = (df_m5_full.index >= start) & (df_m5_full.index <= end)
        df_m5 = df_m5_full[mask_m5]
        if len(df_m5) > 20:
            obs_m5 = detect_order_blocks(df_m5, swing_strength=2, max_group_size=2)
            swings_m5 = find_swings(df_m5, strength=2)
            structure_breaks_m5 = detect_structure_breaks(df_m5, swings=swings_m5)
            mss_setups_m5 = detect_mss_setups(df_m5, structure_breaks=structure_breaks_m5, swings=swings_m5)
            obs_m5_confirmed = confirm_ob_with_mss(obs_m5, mss_setups_m5, window_bars=10)
            cache_m5 = {
                "swings_ltf": swings_m5,
                "fvgs_ltf": detect_fvg(df_m5),
                "breakers_ltf": detect_breakers(df_m5),
                "obs_htf": detect_order_blocks(df_m5_htf),
                "obs_htf2": detect_order_blocks(df_m5_htf2),
                "structure_breaks": structure_breaks_m5,
                "htf_trend": detect_trend(swings_m5, lookback=6),
            }
            for ob_m5 in obs_m5_confirmed:
                try:
                    r_m5 = evaluate_ob(
                        ob_m5, df_m5, df_m5_htf, df_d1, instrument,
                        ltf_name="M5", htf_name="H1",
                        df_htf2=df_m5_htf2, htf2_name="H4",
                        correlated_dfs={}, htf_swings=htf_swings,
                        df_h1=df_m5_htf, min_score=0, min_quality=0,
                        cache=cache_m5,
                    )
                except Exception:
                    continue
                if r_m5.verdict != "TRADE" or r_m5.trade_setup is None:
                    continue
                proba_m5 = ml_filter.predict_proba_for_tf(r_m5, ob_m5, instrument, tf="M5")
                if proba_m5 is None or proba_m5 < threshold_m5:
                    continue
                setups_m5.append((ob_m5.validation_ts, "M5", r_m5, ob_m5, proba_m5))

    # === ORDONNE LES SETUPS PAR TIMESTAMP + APPLIQUE COOLDOWN ===
    all_setups = sorted(setups_m1 + setups_m5, key=lambda s: s[0])

    balance = balance_start
    last_trade_by_tf = {"M1": pd.Timestamp(0, tz="UTC"), "M5": pd.Timestamp(0, tz="UTC")}
    trades = []
    n_skipped = 0

    for val_ts, tf, r, ob, proba in all_setups:
        # Cooldown 15 min par TF
        if (val_ts - last_trade_by_tf[tf]).total_seconds() < COOLDOWN_SEC:
            n_skipped += 1
            continue
        last_trade_by_tf[tf] = val_ts

        setup = r.trade_setup
        df_for_sim = df_ltf if tf == "M1" else df_m5_full[(df_m5_full.index >= start) & (df_m5_full.index <= end)]
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
            sim = simulate_trade(sim_setup, df_for_sim, ob.validation_index + 1)

            # Compute realized RR
            risk_unit = max(abs(setup.entry_price - setup.stop_loss), 1e-9)
            realized_rr = float(sim.pnl_usd) / max(risk_usd, 1e-9)

            # MM compose
            risk_amount = balance * RISK_PCT
            if sim.outcome == "WIN":
                pnl_eur = risk_amount * realized_rr
                balance += pnl_eur
            elif sim.outcome == "LOSS":
                pnl_eur = -risk_amount * abs(realized_rr) if realized_rr != 0 else -risk_amount
                balance += pnl_eur
            else:  # NO_FILL, PENDING
                pnl_eur = 0.0

            trades.append({
                "ts": val_ts, "tf": tf, "outcome": sim.outcome,
                "rr": setup.rr, "realized_rr": realized_rr,
                "pnl_eur": pnl_eur, "balance": balance,
            })
        except Exception:
            continue

    wins = sum(1 for t in trades if t["outcome"] == "WIN")
    losses = sum(1 for t in trades if t["outcome"] == "LOSS")
    no_fill = sum(1 for t in trades if t["outcome"] == "NO_FILL")
    closed = wins + losses
    wr = wins / closed * 100 if closed > 0 else 0
    total_pnl = sum(t["pnl_eur"] for t in trades)

    return {
        "start": start, "end": end,
        "n_setups_total": len(all_setups),
        "n_trades": len(trades),
        "n_skipped_cooldown": n_skipped,
        "wins": wins, "losses": losses, "no_fill": no_fill,
        "wr": wr, "balance_start": balance_start, "balance_end": balance,
        "total_pnl": total_pnl,
        "trades": trades,
    }


def main():
    instrument = "XAUUSD"
    print(f"=== BACKTEST EXHAUSTIF {instrument} ===\n", flush=True)
    print("Configuration :")
    print(f"  - Periode : 2025-06-11 -> 2026-05-15 (OOS test set du ML)")
    print(f"  - Balance initiale : {INITIAL_BALANCE}€")
    print(f"  - Risk par trade : {RISK_PCT*100}% de balance (compound)")
    print(f"  - Cooldown : 15 min par TF (M1 et M5 separes)")
    print(f"  - Multi-TF : M1 + M5")
    print(f"  - TP : dynamique HTF cape a RR=3")
    print(f"  - Seuil ML : 0.50\n", flush=True)

    # Charge toutes les data une seule fois
    print("Chargement data...", flush=True)
    t0 = time.time()
    df_ltf_full = load(instrument, "M1")
    df_htf = load(instrument, "M15")
    df_htf2 = load(instrument, "H1")
    df_h1 = load(instrument, "H1")
    df_d1 = build_d1_from_h1(df_h1)
    htf_dfs_swings = {}
    for tf in ["H1", "H4", "D1"]:
        try:
            htf_dfs_swings[tf] = load(instrument, tf)
        except Exception:
            pass
    htf_swings = collect_htf_swings(htf_dfs_swings, swing_strength=3)

    df_m5_full = load(instrument, "M5")
    df_m5_htf = load(instrument, "H1")
    df_m5_htf2 = load(instrument, "H4")
    print(f"Data charge en {time.time()-t0:.1f}s\n", flush=True)

    # Genere les fenetres mensuelles de 30 jours sur la periode OOS
    oos_start = pd.Timestamp("2025-06-11", tz="UTC")
    oos_end = pd.Timestamp("2026-05-15", tz="UTC")

    months = []
    cur = oos_start
    month_idx = 1
    while cur + pd.Timedelta(days=30) <= oos_end:
        months.append((f"Mois {month_idx}", cur, cur + pd.Timedelta(days=30)))
        cur = cur + pd.Timedelta(days=30)
        month_idx += 1

    print(f"Backtest {len(months)} mois consecutifs...\n", flush=True)

    balance = INITIAL_BALANCE
    all_months_stats = []
    cumul_trades = 0
    cumul_wins = 0
    cumul_losses = 0
    max_balance = balance
    max_drawdown = 0

    print(f"{'#':<4} {'Periode':<25} {'Trades':<8} {'WR':<8} {'Balance fin':<15} {'PnL mois':<12}", flush=True)
    print("-" * 80, flush=True)

    for idx, (name, start, end) in enumerate(months, 1):
        stats = backtest_month(
            instrument, start, end,
            df_ltf_full, df_htf, df_htf2, df_d1, htf_swings,
            df_m5_full, df_m5_htf, df_m5_htf2,
            balance_start=balance,
            threshold_m1=0.50, threshold_m5=0.50,
        )
        if stats is None:
            continue

        balance = stats["balance_end"]
        max_balance = max(max_balance, balance)
        dd = (max_balance - balance) / max_balance * 100 if max_balance > 0 else 0
        max_drawdown = max(max_drawdown, dd)

        cumul_trades += stats["n_trades"]
        cumul_wins += stats["wins"]
        cumul_losses += stats["losses"]

        pnl_mois = stats["balance_end"] - stats["balance_start"]
        period_str = f"{start.date()} -> {end.date()}"
        print(f"{idx:<4} {period_str:<25} {stats['n_trades']:<8} {stats['wr']:<7.1f}% {stats['balance_end']:<14.2f}€ {pnl_mois:>+10.2f}€", flush=True)
        all_months_stats.append(stats)

    print("-" * 80)
    closed_global = cumul_wins + cumul_losses
    wr_global = cumul_wins / closed_global * 100 if closed_global > 0 else 0
    perf_pct = (balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100

    months_positive = sum(1 for s in all_months_stats if s["balance_end"] > s["balance_start"])
    months_negative = sum(1 for s in all_months_stats if s["balance_end"] < s["balance_start"])

    print(f"\n=== BILAN GLOBAL ===")
    print(f"Trades totaux       : {cumul_trades}")
    print(f"Wins / Losses       : {cumul_wins} / {cumul_losses}")
    print(f"WR global           : {wr_global:.1f}%")
    print(f"Balance finale      : {balance:.2f}€ (initial {INITIAL_BALANCE}€)")
    print(f"Performance         : +{perf_pct:.1f}%")
    print(f"Mois positifs       : {months_positive} / {len(all_months_stats)}")
    print(f"Mois negatifs       : {months_negative} / {len(all_months_stats)}")
    print(f"Drawdown max        : -{max_drawdown:.1f}%")

    if all_months_stats:
        best = max(all_months_stats, key=lambda s: s["balance_end"] - s["balance_start"])
        worst = min(all_months_stats, key=lambda s: s["balance_end"] - s["balance_start"])
        print(f"Meilleur mois       : +{best['balance_end']-best['balance_start']:.2f}€ ({best['start'].date()})")
        print(f"Pire mois           : {worst['balance_end']-worst['balance_start']:+.2f}€ ({worst['start'].date()})")


if __name__ == "__main__":
    main()
