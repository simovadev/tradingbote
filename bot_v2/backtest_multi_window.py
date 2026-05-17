"""Backtest sur 3 fenetres temporelles differentes pour mesurer la robustesse.

Objectif : verifier que la config tient sur des periodes variees, pas juste 7j d'optim.

Fenetres testees :
- W1 : 15 mars - 14 avril 2026 (30j OOS1)
- W2 : 1 avril - 30 avril 2026 (30j OOS2, chevauche W1 mais decale)
- W3 : 15 avril - 14 mai 2026 (30j OOS3, inclut zone in-sample)

Cible : WR 55-60%, PF > 1.2, DD < 35% sur la MOYENNE des 3 fenetres.
"""
from __future__ import annotations

import sys
sys.path.insert(0, 'c:/Users/Shadow/TradingBot')

from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd

from bot_v2.backtest import simulate_trade
from bot_v2.config import (
    INITIAL_BALANCE_USD, RISK_PER_TRADE_PCT, SMT_PAIRS, get_param,
    primary_instruments,
)
from bot_v2.concepts.daily_bias import build_d1_from_h1
from bot_v2.concepts.htf_swings import collect_htf_swings
from bot_v2.concepts.order_block import detect_order_blocks
from bot_v2.data_loader import load
from bot_v2.pipeline import evaluate_ob
from bot_v2.portfolio_backtest import GlobalTrade
from bot_v2.trade_setup import TradeSetup, compute_position_size


def _process_instrument_window(args):
    """Worker process : pipeline pour un actif sur une fenetre. Retourne trades."""
    inst, start_ts, end_ts = args
    try:
        from bot_v2.concepts.fvg import detect_fvg
        from bot_v2.concepts.breaker import detect_breakers
        from bot_v2.concepts.structure import detect_structure_breaks, detect_trend
        from bot_v2.concepts.liquidity import find_swings

        df_ltf = load(inst, "M1")
        df_htf = load(inst, "M15")
        try:
            df_htf2 = load(inst, "H1")
        except Exception:
            df_htf2 = None
        try:
            df_d1 = load(inst, "D1")
            if len(df_d1) < 10:
                raise FileNotFoundError
        except Exception:
            df_h1 = load(inst, "H1")
            df_d1 = build_d1_from_h1(df_h1)

        htf_dfs_swings = {}
        for tf in ["H1", "H4", "D1"]:
            try:
                htf_dfs_swings[tf] = load(inst, tf)
            except Exception:
                pass
        htf_swings = collect_htf_swings(htf_dfs_swings, swing_strength=3)

        mask = (df_ltf.index >= start_ts) & (df_ltf.index <= end_ts)
        df_ltf_w = df_ltf[mask]

        correlated_dfs = {}
        for corr_name, corr_type in SMT_PAIRS.get(inst, []):
            try:
                df_c = load(corr_name, "M1")
                if len(df_c) > 0:
                    mask_c = (df_c.index >= start_ts) & (df_c.index <= end_ts)
                    correlated_dfs[corr_name] = (df_c[mask_c], corr_type)
            except Exception:
                continue

        swing_strength_ltf = get_param(inst, "swing_strength_m1", 2)
        obs = detect_order_blocks(df_ltf_w, swing_strength=swing_strength_ltf, max_group_size=2)

        # OPTIM 1 : pre-calcul cache
        cache = {
            "swings_ltf": find_swings(df_ltf_w, strength=swing_strength_ltf),
            "fvgs_ltf": detect_fvg(df_ltf_w),
            "breakers_ltf": detect_breakers(df_ltf_w),
            "obs_htf": detect_order_blocks(df_htf),
        }
        cache["structure_breaks"] = detect_structure_breaks(df_ltf_w, swings=cache["swings_ltf"])
        cache["htf_trend"] = detect_trend(cache["swings_ltf"], lookback=6)
        if df_htf2 is not None:
            cache["obs_htf2"] = detect_order_blocks(df_htf2)

        df_h1_for_feu_vert = df_htf2 if df_htf2 is not None else None
        min_score = get_param(inst, "min_score", 145)
        min_quality = get_param(inst, "min_quality", 55)

        trades = []
        for ob in obs:
            r = evaluate_ob(
                ob, df_ltf_w, df_htf, df_d1, inst,
                ltf_name="M1", htf_name="M15",
                df_htf2=df_htf2, htf2_name="H1",
                correlated_dfs=correlated_dfs,
                htf_swings=htf_swings,
                df_h1=df_h1_for_feu_vert,
                min_score=min_score,
                min_quality=min_quality,
                cache=cache,
            )
            if r.verdict == "TRADE":
                trades.append((inst, r.ob.validation_ts, r, df_ltf_w))
        return trades, None
    except Exception as e:
        return [], str(e)


def run_window(start_ts, end_ts, instruments, label):
    """Run backtest sur une fenetre [start_ts, end_ts] - PARALLELE par actif."""
    all_trades = []
    tasks = [(inst, start_ts, end_ts) for inst in instruments]
    with ProcessPoolExecutor(max_workers=min(len(instruments), 4)) as executor:
        futures = {executor.submit(_process_instrument_window, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            trades, error = fut.result()
            if error:
                print(f"    ERREUR : {error}")
                continue
            all_trades.extend(trades)

    all_trades.sort(key=lambda x: x[1])
    balance = 60.0
    peak = balance
    max_dd = 0.0
    open_positions = []
    global_trades = []
    risk_pct = 0.10

    for inst, val_ts, r, df in all_trades:
        open_positions = [(i, t) for i, t in open_positions if t > val_ts]
        if len(open_positions) >= 10:
            continue
        setup = r.trade_setup
        lots, risk_usd = compute_position_size(
            setup.entry_price, setup.stop_loss, inst,
            balance=balance, risk_pct=risk_pct,
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
            instrument=inst, fill_ts=tr.fill_ts, exit_ts=tr.exit_ts,
            direction=setup.direction, outcome=tr.outcome,
            entry=setup.entry_price, sl=setup.stop_loss, tp=setup.take_profit,
            rr=setup.rr, pnl_usd=tr.pnl_usd, score=r.score, balance_after=balance,
        ))

    wins = [t for t in global_trades if t.outcome == "WIN"]
    losses = [t for t in global_trades if t.outcome == "LOSS"]
    wr = len(wins) / (len(wins) + len(losses)) * 100 if (wins or losses) else 0
    wins_sum = sum(t.pnl_usd for t in wins)
    losses_sum = abs(sum(t.pnl_usd for t in losses))
    pf = wins_sum / losses_sum if losses_sum > 0 else 0.0

    return {
        "label": label,
        "balance": balance,
        "trades": len(global_trades),
        "wins": len(wins),
        "losses": len(losses),
        "wr": wr,
        "pf": pf,
        "max_dd": max_dd,
        "by_inst": {
            t.instrument: t for t in global_trades
        },
        "details": global_trades,
    }


def main():
    instruments = primary_instruments()
    windows = [
        ("W1 (15 mars - 14 avr)", pd.Timestamp("2026-03-15 22:00:00+00:00"), pd.Timestamp("2026-04-14 22:00:00+00:00")),
        ("W2 (1 avr - 30 avr)",   pd.Timestamp("2026-04-01 00:00:00+00:00"), pd.Timestamp("2026-04-30 23:59:00+00:00")),
        ("W3 (15 avr - 14 mai)",  pd.Timestamp("2026-04-15 00:00:00+00:00"), pd.Timestamp("2026-05-14 14:00:00+00:00")),
    ]

    results = []
    for label, start_ts, end_ts in windows:
        print(f"\n=== {label} ===", flush=True)
        r = run_window(start_ts, end_ts, instruments, label)
        results.append(r)
        print(f"  BAL={r['balance']:.2f} WR={r['wr']:.1f}% TRADES={r['trades']} (W={r['wins']} L={r['losses']}) DD={r['max_dd']:.1f}% PF={r['pf']:.2f}")
        per_inst = {}
        for t in r["details"]:
            d = per_inst.setdefault(t.instrument, {"n": 0, "w": 0, "l": 0, "pnl": 0.0})
            d["n"] += 1
            if t.outcome == "WIN": d["w"] += 1
            else: d["l"] += 1
            d["pnl"] += t.pnl_usd
        for inst, d in sorted(per_inst.items()):
            local_wr = d["w"] / (d["w"] + d["l"]) * 100 if (d["w"] + d["l"]) else 0
            print(f"    {inst}: {d['n']} trades, WR={local_wr:.0f}%, pnl={d['pnl']:+.2f}")

    # Stats agregees
    total_trades = sum(r["trades"] for r in results)
    total_wins = sum(r["wins"] for r in results)
    total_losses = sum(r["losses"] for r in results)
    avg_wr = total_wins / (total_wins + total_losses) * 100 if (total_wins + total_losses) else 0
    avg_pf = sum(r["pf"] for r in results) / len(results)
    avg_dd = sum(r["max_dd"] for r in results) / len(results)

    print()
    print("=" * 70)
    print("MOYENNES SUR LES 3 FENETRES")
    print("=" * 70)
    print(f"  Total trades  : {total_trades}")
    print(f"  WR moyen      : {avg_wr:.1f}%")
    print(f"  PF moyen      : {avg_pf:.2f}")
    print(f"  DD moyen      : {avg_dd:.1f}%")
    print()
    print("Resultats par fenetre :")
    for r in results:
        print(f"  {r['label']:30s} | BAL={r['balance']:.0f} | WR={r['wr']:.0f}% | PF={r['pf']:.2f} | DD={r['max_dd']:.0f}%")


if __name__ == "__main__":
    main()
