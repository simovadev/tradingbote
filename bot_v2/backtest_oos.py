"""Backtest Out-Of-Sample : test sur une fenetre temporelle DIFFERENTE de l'optimisation.

L'optimisation a porte sur les 7 derniers jours (~ 7-14 mai 2026).
On teste maintenant sur les 30 premiers jours (15 mars - 14 avril 2026) que le bot
n'a jamais vu.

Reuse la logique du portfolio_backtest mais avec une fenetre temporelle absolue.
"""
from __future__ import annotations

import sys
sys.path.insert(0, 'c:/Users/Shadow/TradingBot')

import pandas as pd

from bot_v2.backtest import simulate_trade
from bot_v2.config import INITIAL_BALANCE_USD, RISK_PER_TRADE_PCT, primary_instruments
from bot_v2.data_loader import load
from bot_v2.pipeline import (
    PipelineResult, _find_parent_ob,  # noqa
    evaluate_ob, run_pipeline,
)
from bot_v2.concepts.htf_swings import collect_htf_swings
from bot_v2.concepts.daily_bias import build_d1_from_h1
from bot_v2.concepts.order_block import detect_order_blocks
from bot_v2.config import SMT_PAIRS, get_param
from bot_v2.portfolio_backtest import GlobalTrade
from bot_v2.trade_setup import TradeSetup, compute_position_size


def run_pipeline_oos(instrument, start_ts, end_ts, ltf_name="M1", htf_name="M15", htf2_name="H1"):
    """Run pipeline sur une fenetre [start_ts, end_ts] precise."""
    df_ltf = load(instrument, ltf_name)
    df_htf = load(instrument, htf_name)
    df_htf2 = None
    if htf2_name:
        try:
            df_htf2 = load(instrument, htf2_name)
        except Exception:
            pass

    try:
        df_d1 = load(instrument, "D1")
        if len(df_d1) < 10:
            raise FileNotFoundError
    except (FileNotFoundError, Exception):
        df_h1 = load(instrument, "H1")
        df_d1 = build_d1_from_h1(df_h1)

    htf_dfs_swings = {}
    for tf in ["H1", "H4", "D1"]:
        try:
            df_t = load(instrument, tf)
            htf_dfs_swings[tf] = df_t
        except Exception:
            pass
    htf_swings = collect_htf_swings(htf_dfs_swings, swing_strength=3)

    # Filtre fenetre temporelle PRECISE
    mask = (df_ltf.index >= start_ts) & (df_ltf.index <= end_ts)
    df_ltf = df_ltf[mask]

    correlated_dfs = {}
    for corr_name, corr_type in SMT_PAIRS.get(instrument, []):
        for tf_try in [ltf_name, "M1"]:
            try:
                df_c = load(corr_name, tf_try)
                if len(df_c) > 0:
                    mask_c = (df_c.index >= start_ts) & (df_c.index <= end_ts)
                    correlated_dfs[corr_name] = (df_c[mask_c], corr_type)
                    break
            except Exception:
                continue

    swing_strength_ltf = get_param(instrument, "swing_strength_m1", 2)
    obs = detect_order_blocks(df_ltf, swing_strength=swing_strength_ltf, max_group_size=2)

    df_h1_for_feu_vert = None
    if htf2_name == "H1" and df_htf2 is not None:
        df_h1_for_feu_vert = df_htf2
    elif htf_name == "H1":
        df_h1_for_feu_vert = df_htf
    else:
        try:
            df_h1_for_feu_vert = load(instrument, "H1")
        except Exception:
            pass

    min_score = get_param(instrument, "min_score", 145)
    min_quality = get_param(instrument, "min_quality", 55)

    results = []
    for ob in obs:
        r = evaluate_ob(
            ob, df_ltf, df_htf, df_d1, instrument,
            ltf_name=ltf_name, htf_name=htf_name,
            df_htf2=df_htf2, htf2_name=htf2_name,
            correlated_dfs=correlated_dfs,
            htf_swings=htf_swings,
            df_h1=df_h1_for_feu_vert,
            min_score=min_score,
            min_quality=min_quality,
        )
        results.append(r)
    return results, df_ltf


def main():
    # Fenetre OOS : 15 mars - 14 avril 2026 (30 jours INEDITS pour le bot)
    start_ts = pd.Timestamp("2026-03-15 22:00:00+00:00")
    end_ts = pd.Timestamp("2026-04-14 22:00:00+00:00")

    instruments = primary_instruments()
    all_trades = []

    for inst in instruments:
        print(f"  [{inst}] pipeline OOS...", flush=True)
        try:
            results, df = run_pipeline_oos(inst, start_ts, end_ts)
        except Exception as e:
            print(f"    ERREUR : {e}")
            continue
        trades = [r for r in results if r.verdict == "TRADE"]
        for r in trades:
            all_trades.append((inst, r.ob.validation_ts, {"result": r, "df": df}))

    all_trades.sort(key=lambda x: x[1])
    print(f"  Total trades candidats : {len(all_trades)}")

    balance = 60.0
    peak = balance
    max_dd = 0.0
    open_positions = []
    global_trades = []
    risk_pct = 0.10  # 10% comme demande

    for inst, val_ts, payload in all_trades:
        open_positions = [(i, t) for i, t in open_positions if t > val_ts]
        if len(open_positions) >= 10:
            continue
        r = payload["result"]
        df = payload["df"]
        setup = r.trade_setup
        lots, risk_usd = compute_position_size(
            setup.entry_price, setup.stop_loss, inst,
            balance=balance, risk_pct=risk_pct,
        )
        if lots <= 0:
            continue
        sim_setup = TradeSetup(
            instrument=inst,
            direction=setup.direction,
            entry_price=setup.entry_price,
            stop_loss=setup.stop_loss,
            take_profit=setup.take_profit,
            rr=setup.rr,
            risk_points=setup.risk_points,
            reward_points=setup.reward_points,
            risk_usd=risk_usd,
            reward_usd=risk_usd * setup.rr,
            position_size_lots=lots,
            ob_validation_ts=setup.ob_validation_ts,
            tp_source=setup.tp_source,
        )
        tr = simulate_trade(sim_setup, df, r.ob.validation_index + 1)
        if tr.outcome == "NO_FILL":
            continue
        balance += tr.pnl_usd
        peak = max(peak, balance)
        dd = (peak - balance) / peak * 100 if peak > 0 else 0
        max_dd = max(max_dd, dd)
        if tr.exit_ts is not None:
            open_positions.append((inst, tr.exit_ts))
        global_trades.append(GlobalTrade(
            instrument=inst,
            fill_ts=tr.fill_ts,
            exit_ts=tr.exit_ts,
            direction=setup.direction,
            outcome=tr.outcome,
            entry=setup.entry_price,
            sl=setup.stop_loss,
            tp=setup.take_profit,
            rr=setup.rr,
            pnl_usd=tr.pnl_usd,
            score=r.score,
            balance_after=balance,
        ))

    wins = [t for t in global_trades if t.outcome == "WIN"]
    losses = [t for t in global_trades if t.outcome == "LOSS"]
    wr = len(wins) / (len(wins) + len(losses)) * 100 if (wins or losses) else 0
    wins_sum = sum(t.pnl_usd for t in wins)
    losses_sum = abs(sum(t.pnl_usd for t in losses))
    pf = wins_sum / losses_sum if losses_sum > 0 else 0.0

    print()
    print("=" * 70)
    print(f"BACKTEST OOS 30j : 15 mars - 14 avril 2026 (jamais vu en optim)")
    print("=" * 70)
    print(f"  Balance initial : 60.00 EUR")
    print(f"  Balance final   : {balance:.2f} EUR")
    print(f"  Profit          : {balance - 60:+.2f} EUR")
    print(f"  Return          : {(balance - 60) / 60 * 100:+.2f}%")
    print(f"  Trades          : {len(global_trades)}")
    print(f"    Wins          : {len(wins)}")
    print(f"    Losses        : {len(losses)}")
    print(f"  Win rate        : {wr:.1f}%")
    print(f"  Max DD          : {max_dd:.1f}%")
    print(f"  Profit factor   : {pf:.2f}")
    print(f"  Trades/jour     : {len(global_trades)/30:.2f}")
    print()
    print("=== Par actif ===")
    by_inst = {}
    for t in global_trades:
        d = by_inst.setdefault(t.instrument, {"n": 0, "wins": 0, "losses": 0, "pnl": 0.0})
        d["n"] += 1
        if t.outcome == "WIN":
            d["wins"] += 1
        elif t.outcome == "LOSS":
            d["losses"] += 1
        d["pnl"] += t.pnl_usd
    for inst, d in sorted(by_inst.items()):
        local_wr = d["wins"] / (d["wins"] + d["losses"]) * 100 if (d["wins"] + d["losses"]) else 0
        print(f"  {inst:8s} | {d['n']:3d} trades | win={local_wr:.0f}% | pnl={d['pnl']:+.2f}")


if __name__ == "__main__":
    main()
