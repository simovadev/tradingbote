"""Backtest multi-actif unifié — un seul portefeuille qui trade tous les actifs.

Differences avec backtest.py :
- Le balance est SHARED entre tous les actifs.
- Position sizing recalcule a chaque trade selon le balance courant.
- Equity curve consolidee.
- Pas de chevauchement : limite optionnelle au nombre de positions concurrentes.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import pandas as pd

from bot_v2.backtest import simulate_trade
from bot_v2.config import INITIAL_BALANCE_USD, RISK_PER_TRADE_PCT, primary_instruments
from bot_v2.data_loader import load
from bot_v2.pipeline import run_pipeline
from bot_v2.trade_setup import compute_position_size


def _run_pipeline_for_instrument(args):
    """Worker process : lance le pipeline pour un actif. Retourne (inst, results, df_ltf)."""
    inst, ltf_name, htf_name, htf2_name, days, min_score, min_quality = args
    try:
        results = run_pipeline(
            inst, ltf_name=ltf_name, htf_name=htf_name, htf2_name=htf2_name,
            days=days, min_score=min_score, min_quality=min_quality,
        )
        df = load(inst, ltf_name)
        mask = df.index >= (df.index.max() - pd.Timedelta(days=days))
        df = df[mask]
        return inst, results, df, None
    except Exception as e:
        return inst, None, None, str(e)


@dataclass
class GlobalTrade:
    """Trade dans un portefeuille global."""
    instrument: str
    fill_ts: pd.Timestamp
    exit_ts: pd.Timestamp | None
    direction: str
    outcome: str
    entry: float
    sl: float
    tp: float
    rr: float
    pnl_usd: float
    score: int
    balance_after: float


def run_portfolio_backtest(
    instruments: list[str] | None = None,
    days: int = 30,
    ltf_name: str = "M1",
    htf_name: str = "M15",
    htf2_name: str = "H1",
    max_concurrent: int = 5,
    min_score: int | None = None,
    min_quality: int | None = None,
) -> dict:
    """Backtest portefeuille unifié.

    Args:
        instruments: liste actifs (par defaut tous les primaires).
        max_concurrent: max de trades simultanes (None = pas de limite).
    """
    if instruments is None:
        instruments = primary_instruments()

    # 1. Lancer le pipeline pour chaque actif EN PARALLELE (OPTIM 2)
    # Avant : 4 actifs sequentiels = ~60s. Apres : 4 workers en parallele = ~15-20s.
    all_trades: list[tuple[str, pd.Timestamp, dict]] = []  # (instrument, validation_ts, payload)

    tasks = [
        (inst, ltf_name, htf_name, htf2_name, days, min_score, min_quality)
        for inst in instruments
    ]
    n_workers = min(len(instruments), 4)
    print(f"  Pipeline parallele : {n_workers} workers pour {len(instruments)} actifs...", flush=True)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_run_pipeline_for_instrument, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            inst, results, df, error = fut.result()
            if error:
                print(f"    ERREUR {inst} : {error}", flush=True)
                continue
            print(f"  [{inst}] {len(results)} candidats", flush=True)
            trades = [r for r in results if r.verdict == "TRADE"]
            for r in trades:
                all_trades.append((
                    inst, r.ob.validation_ts,
                    {"result": r, "df": df},
                ))

    # 2. Trier par ordre chronologique
    all_trades.sort(key=lambda x: x[1])
    print(f"  Total trades candidats : {len(all_trades)}")

    # 3. Simuler dans l'ordre, avec balance partagé
    balance = INITIAL_BALANCE_USD
    peak = balance
    max_dd = 0.0
    open_positions: list[tuple[str, pd.Timestamp]] = []  # (instrument, exit_ts attendu)
    global_trades: list[GlobalTrade] = []

    for inst, val_ts, payload in all_trades:
        # Verifier le max_concurrent : on supprime les positions deja closes
        open_positions = [(i, t) for i, t in open_positions if t > val_ts]
        if max_concurrent and len(open_positions) >= max_concurrent:
            continue   # Skip si trop de positions

        r = payload["result"]
        df = payload["df"]
        setup = r.trade_setup

        # Recalcule la position size avec le balance courant
        lots, risk_usd = compute_position_size(
            setup.entry_price, setup.stop_loss, inst,
            balance=balance, risk_pct=RISK_PER_TRADE_PCT,
        )
        if lots <= 0:
            continue

        # Simule le trade
        from bot_v2.trade_setup import TradeSetup
        # On recree un setup avec les bons lots pour la simulation
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

        # Update balance
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

    # 4. Stats
    wins = [t for t in global_trades if t.outcome == "WIN"]
    losses = [t for t in global_trades if t.outcome == "LOSS"]
    win_rate = len(wins) / (len(wins) + len(losses)) * 100 if (wins or losses) else 0
    wins_sum = sum(t.pnl_usd for t in wins)
    losses_sum = abs(sum(t.pnl_usd for t in losses))
    pf = wins_sum / losses_sum if losses_sum > 0 else 0.0

    # Repartition par actif
    by_inst: dict[str, dict] = {}
    for t in global_trades:
        d = by_inst.setdefault(t.instrument, {"n": 0, "wins": 0, "losses": 0, "pnl": 0.0})
        d["n"] += 1
        if t.outcome == "WIN":
            d["wins"] += 1
        elif t.outcome == "LOSS":
            d["losses"] += 1
        d["pnl"] += t.pnl_usd

    return {
        "initial_balance": INITIAL_BALANCE_USD,
        "final_balance": balance,
        "return_pct": (balance - INITIAL_BALANCE_USD) / INITIAL_BALANCE_USD * 100,
        "n_trades": len(global_trades),
        "n_wins": len(wins),
        "n_losses": len(losses),
        "win_rate": win_rate,
        "profit_factor": pf,
        "max_drawdown_pct": max_dd,
        "by_instrument": by_inst,
        "trades": global_trades,
        "n_candidates": len(all_trades),
        "days": days,
    }


def print_portfolio_report(rpt: dict) -> None:
    print(f"\n{'=' * 70}")
    print(f"BACKTEST PORTEFEUILLE UNIFIE — {rpt['days']} jours")
    print(f"{'=' * 70}")
    print(f"  Actifs           : {len(rpt['by_instrument'])}")
    print(f"  Candidats total  : {rpt['n_candidates']}")
    print(f"  Trades executes  : {rpt['n_trades']}")
    print(f"    Wins           : {rpt['n_wins']}")
    print(f"    Losses         : {rpt['n_losses']}")
    print(f"  Win rate         : {rpt['win_rate']:.1f}%")
    print(f"  Profit factor    : {rpt['profit_factor']:.2f}")
    print(f"  Max DD           : {rpt['max_drawdown_pct']:.1f}%")
    print(f"  Balance initial  : ${rpt['initial_balance']:.2f}")
    print(f"  Balance final    : ${rpt['final_balance']:.2f}")
    print(f"  Return           : {rpt['return_pct']:+.2f}%")

    days = rpt['days']
    annualized = (rpt['return_pct'] / days) * 365
    print(f"  Annualise (lin)  : {annualized:+.1f}%")

    print(f"\n  === Par actif ===")
    for inst, d in sorted(rpt["by_instrument"].items()):
        wr = d["wins"] / (d["wins"] + d["losses"]) * 100 if (d["wins"] + d["losses"]) else 0
        print(f"    {inst:8s} | {d['n']:3d} trades | win={wr:.0f}% | pnl=${d['pnl']:+8.2f}")


if __name__ == "__main__":
    import sys
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    rpt = run_portfolio_backtest(days=days)
    print_portfolio_report(rpt)
