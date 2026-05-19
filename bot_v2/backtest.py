"""Backtest module : replay les setups valides sur l'historique et calcule le PnL.

Pour chaque TradeSetup valide :
- On simule l'entree au prix limite (entry_price).
- On verifie chronologiquement quelle borne (SL ou TP) est touchee en 1ere.
- On compte le PnL et on met a jour le balance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from bot_v2.config import INITIAL_BALANCE_USD, INSTRUMENTS
from bot_v2.pipeline import PipelineResult, run_pipeline
from bot_v2.trade_setup import TradeSetup


TradeOutcome = Literal["WIN", "LOSS", "PENDING", "NO_FILL"]


@dataclass
class TradeResult:
    """Resultat d'un trade simule."""
    setup: TradeSetup
    outcome: TradeOutcome
    fill_index: int | None        # index ou l'entree a ete remplie
    fill_ts: pd.Timestamp | None
    exit_index: int | None        # index ou le trade s'est cloture
    exit_ts: pd.Timestamp | None
    exit_price: float | None
    pnl_usd: float
    pnl_pct: float                # vs balance avant entry
    score: int = 0                # score de la pipeline


@dataclass
class BacktestReport:
    """Rapport agrege d'un backtest."""
    instrument: str
    n_trades: int
    n_wins: int
    n_losses: int
    n_no_fill: int
    n_pending: int
    win_rate: float
    total_pnl_usd: float
    initial_balance: float
    final_balance: float
    return_pct: float
    avg_win_usd: float
    avg_loss_usd: float
    profit_factor: float          # somme(wins) / |somme(losses)|
    max_drawdown_pct: float
    trades: list[TradeResult]


def simulate_trade(
    setup: TradeSetup,
    df: pd.DataFrame,
    start_index: int,
    max_bars_to_fill: int = 30,
) -> TradeResult:
    """Simule un trade : cherche d'abord le FILL puis le SL/TP.

    Fill = le prix LTF touche entry_price apres start_index (pullback dans la zone OB).
    """
    highs = df["high"].values
    lows = df["low"].values
    entry = setup.entry_price
    sl = setup.stop_loss
    tp = setup.take_profit

    # 1. Cherche le FILL (price revient toucher l'entry)
    fill_idx = None
    for j in range(start_index, min(start_index + max_bars_to_fill, len(df))):
        if setup.direction == "bullish":
            # Long : on remplit si le LOW touche ou descend en-dessous d'entry
            if lows[j] <= entry:
                fill_idx = j
                break
        else:
            # Short : on remplit si le HIGH touche ou monte au-dessus d'entry
            if highs[j] >= entry:
                fill_idx = j
                break

    if fill_idx is None:
        return TradeResult(
            setup=setup,
            outcome="NO_FILL",
            fill_index=None, fill_ts=None,
            exit_index=None, exit_ts=None, exit_price=None,
            pnl_usd=0.0, pnl_pct=0.0,
        )

    # 2. Cherche SL ou TP apres le fill
    # Option A (user 2026-05-17) : on commence a fill_idx + 1 pour ne pas examiner
    # la bougie de fill (sa mèche couvre souvent SL+TP sur WTI/NAS volatiles,
    # ce qui causait des LOSS instantanes irrealistes).
    # FIX 2026-05-19 : cap a max_scan_bars (500 bougies = 8h M1) pour eviter
    # de scanner 2.6M bougies sur chaque setup (perf x100).
    max_scan_bars = 500
    scan_end = min(fill_idx + 1 + max_scan_bars, len(df))
    for j in range(fill_idx + 1, scan_end):
        h = highs[j]
        l = lows[j]
        if setup.direction == "bullish":
            sl_hit = l <= sl
            tp_hit = h >= tp
        else:
            sl_hit = h >= sl
            tp_hit = l <= tp

        # Si les deux touchent dans la meme bougie : on suppose SL touche en 1er (conservateur)
        if sl_hit and tp_hit:
            return TradeResult(
                setup=setup,
                outcome="LOSS",
                fill_index=fill_idx, fill_ts=df.index[fill_idx],
                exit_index=j, exit_ts=df.index[j], exit_price=sl,
                pnl_usd=-setup.risk_usd,
                pnl_pct=-setup.risk_usd / INITIAL_BALANCE_USD * 100,
            )
        if tp_hit:
            return TradeResult(
                setup=setup,
                outcome="WIN",
                fill_index=fill_idx, fill_ts=df.index[fill_idx],
                exit_index=j, exit_ts=df.index[j], exit_price=tp,
                pnl_usd=setup.reward_usd,
                pnl_pct=setup.reward_usd / INITIAL_BALANCE_USD * 100,
            )
        if sl_hit:
            return TradeResult(
                setup=setup,
                outcome="LOSS",
                fill_index=fill_idx, fill_ts=df.index[fill_idx],
                exit_index=j, exit_ts=df.index[j], exit_price=sl,
                pnl_usd=-setup.risk_usd,
                pnl_pct=-setup.risk_usd / INITIAL_BALANCE_USD * 100,
            )

    # 3. Toujours en cours
    return TradeResult(
        setup=setup,
        outcome="PENDING",
        fill_index=fill_idx, fill_ts=df.index[fill_idx],
        exit_index=None, exit_ts=None, exit_price=None,
        pnl_usd=0.0, pnl_pct=0.0,
    )


def simulate_trade_trailing(
    setup: TradeSetup,
    df: pd.DataFrame,
    start_index: int,
    max_bars_to_fill: int = 30,
    trailing_activate_pct: float = 0.50,
    swing_lookback: int = 10,
) -> TradeResult:
    """Simule un trade avec trailing SL au dernier swing low/high.

    Phases (user 2026-05-16, annule breakeven 30%) :
    1. Phase initiale (0-50% TP) : SL initial actif
    2. Phase trailing (50%+ TP) : SL = dernier swing low/high, capture runs

    Pas de breakeven force : on laisse le SL initial faire son boulot et on
    capture les gros runs via trailing apres 50% TP.
    """
    highs = df["high"].values
    lows = df["low"].values
    entry = setup.entry_price
    initial_sl = setup.stop_loss
    initial_tp = setup.take_profit

    # 1. Cherche le FILL
    fill_idx = None
    for j in range(start_index, min(start_index + max_bars_to_fill, len(df))):
        if setup.direction == "bullish":
            if lows[j] <= entry:
                fill_idx = j
                break
        else:
            if highs[j] >= entry:
                fill_idx = j
                break

    if fill_idx is None:
        return TradeResult(
            setup=setup, outcome="NO_FILL",
            fill_index=None, fill_ts=None,
            exit_index=None, exit_ts=None, exit_price=None,
            pnl_usd=0.0, pnl_pct=0.0,
        )

    # 2. Seuil d'activation du trailing (50% du chemin entry->TP)
    half_tp_price = entry + (initial_tp - entry) * trailing_activate_pct

    # 3. Etat dynamique
    current_sl = initial_sl
    trailing_active = False

    for j in range(fill_idx, len(df)):
        h = highs[j]
        l = lows[j]

        risk_unit = abs(entry - initial_sl)
        if risk_unit == 0:
            risk_unit = 1e-9  # safety

        if setup.direction == "bullish":
            # 0. CHECK TP PLEIN : si la bougie touche le TP, sortie WIN
            #    (Fix bug 2026-05-16 #1 : sans ca, on continue avec trailing meme apres TP atteint)
            if h >= initial_tp:
                # Le TP est atteint dans cette bougie -> WIN au prix TP
                pnl_pts = initial_tp - entry
                real_rr = pnl_pts / risk_unit
                pnl_usd = real_rr * setup.risk_usd
                return TradeResult(
                    setup=setup, outcome="WIN",
                    fill_index=fill_idx, fill_ts=df.index[fill_idx],
                    exit_index=j, exit_ts=df.index[j], exit_price=float(initial_tp),
                    pnl_usd=float(pnl_usd),
                    pnl_pct=pnl_usd / INITIAL_BALANCE_USD * 100,
                )

            # 1. Activer le trailing si 50% TP atteint
            if not trailing_active and h >= half_tp_price:
                trailing_active = True
                current_sl = max(current_sl, entry)  # breakeven minimum

            # 2. Si trailing actif, update SL au dernier swing low
            if trailing_active:
                lookback_start = max(fill_idx, j - swing_lookback)
                if j - lookback_start >= 3:
                    recent_low = min(lows[lookback_start:j])
                    if recent_low > current_sl:
                        current_sl = recent_low

            # 3. Check SL (potentiellement mis a jour par trailing)
            if l <= current_sl:
                pnl_pts = current_sl - entry
                real_rr = pnl_pts / risk_unit
                # WIN si profit, LOSS si perte, BREAKEVEN si nul (pas LOSS)
                if pnl_pts > 0.0001:
                    outcome = "WIN"
                elif pnl_pts < -0.0001:
                    outcome = "LOSS"
                else:
                    outcome = "BREAKEVEN"
                pnl_usd = real_rr * setup.risk_usd
                return TradeResult(
                    setup=setup, outcome=outcome,
                    fill_index=fill_idx, fill_ts=df.index[fill_idx],
                    exit_index=j, exit_ts=df.index[j], exit_price=float(current_sl),
                    pnl_usd=float(pnl_usd),
                    pnl_pct=pnl_usd / INITIAL_BALANCE_USD * 100,
                )
        else:
            # Bearish : check TP, trailing, SL
            if l <= initial_tp:
                pnl_pts = entry - initial_tp
                real_rr = pnl_pts / risk_unit
                pnl_usd = real_rr * setup.risk_usd
                return TradeResult(
                    setup=setup, outcome="WIN",
                    fill_index=fill_idx, fill_ts=df.index[fill_idx],
                    exit_index=j, exit_ts=df.index[j], exit_price=float(initial_tp),
                    pnl_usd=float(pnl_usd),
                    pnl_pct=pnl_usd / INITIAL_BALANCE_USD * 100,
                )

            if not trailing_active and l <= half_tp_price:
                trailing_active = True
                current_sl = min(current_sl, entry)

            if trailing_active:
                lookback_start = max(fill_idx, j - swing_lookback)
                if j - lookback_start >= 3:
                    recent_high = max(highs[lookback_start:j])
                    if recent_high < current_sl:
                        current_sl = recent_high

            if h >= current_sl:
                pnl_pts = entry - current_sl
                real_rr = pnl_pts / risk_unit
                if pnl_pts > 0.0001:
                    outcome = "WIN"
                elif pnl_pts < -0.0001:
                    outcome = "LOSS"
                else:
                    outcome = "BREAKEVEN"
                pnl_usd = real_rr * setup.risk_usd
                return TradeResult(
                    setup=setup, outcome=outcome,
                    fill_index=fill_idx, fill_ts=df.index[fill_idx],
                    exit_index=j, exit_ts=df.index[j], exit_price=float(current_sl),
                    pnl_usd=float(pnl_usd),
                    pnl_pct=pnl_usd / INITIAL_BALANCE_USD * 100,
                )

    return TradeResult(
        setup=setup, outcome="PENDING",
        fill_index=fill_idx, fill_ts=df.index[fill_idx],
        exit_index=None, exit_ts=None, exit_price=None,
        pnl_usd=0.0, pnl_pct=0.0,
    )


def backtest_instrument(
    instrument: str,
    ltf_name: str = "M1",
    htf_name: str = "M15",
    htf2_name: str = "H1",
    days: int = 30,
    min_score: int = 100,
    min_quality: int = 60,
) -> BacktestReport:
    """Lance pipeline (chaine Vizion D1->H1->M15->M1) + simulation."""
    results = run_pipeline(
        instrument,
        ltf_name=ltf_name, htf_name=htf_name, htf2_name=htf2_name,
        days=days, min_score=min_score, min_quality=min_quality,
    )
    trades_setups = [r for r in results if r.verdict == "TRADE"]

    # Recharge le LTF complet pour la simulation
    from bot_v2.data_loader import load
    df = load(instrument, ltf_name)
    mask = df.index >= (df.index.max() - pd.Timedelta(days=days))
    df = df[mask]

    trade_results: list[TradeResult] = []
    for r in trades_setups:
        setup = r.trade_setup
        start_idx = r.ob.validation_index + 1
        tr = simulate_trade(setup, df, start_idx)
        tr.score = r.score
        trade_results.append(tr)

    # Stats
    wins = [t for t in trade_results if t.outcome == "WIN"]
    losses = [t for t in trade_results if t.outcome == "LOSS"]
    no_fills = [t for t in trade_results if t.outcome == "NO_FILL"]
    pending = [t for t in trade_results if t.outcome == "PENDING"]
    closed = wins + losses

    total_pnl = sum(t.pnl_usd for t in trade_results)
    final_balance = INITIAL_BALANCE_USD + total_pnl

    win_rate = len(wins) / len(closed) * 100 if closed else 0
    avg_win = sum(t.pnl_usd for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t.pnl_usd for t in losses) / len(losses) if losses else 0
    wins_sum = sum(t.pnl_usd for t in wins)
    losses_sum = abs(sum(t.pnl_usd for t in losses))
    pf = wins_sum / losses_sum if losses_sum > 0 else 0.0

    # Equity curve + max DD
    balance = INITIAL_BALANCE_USD
    peak = balance
    max_dd = 0.0
    # On trie les trades par ordre chronologique de fill
    trade_results.sort(key=lambda t: t.fill_ts if t.fill_ts is not None else pd.Timestamp.min.tz_localize("UTC"))
    for t in trade_results:
        balance += t.pnl_usd
        peak = max(peak, balance)
        dd = (peak - balance) / peak * 100 if peak > 0 else 0
        max_dd = max(max_dd, dd)

    return BacktestReport(
        instrument=instrument,
        n_trades=len(trade_results),
        n_wins=len(wins),
        n_losses=len(losses),
        n_no_fill=len(no_fills),
        n_pending=len(pending),
        win_rate=win_rate,
        total_pnl_usd=total_pnl,
        initial_balance=INITIAL_BALANCE_USD,
        final_balance=final_balance,
        return_pct=(final_balance - INITIAL_BALANCE_USD) / INITIAL_BALANCE_USD * 100,
        avg_win_usd=avg_win,
        avg_loss_usd=avg_loss,
        profit_factor=pf,
        max_drawdown_pct=max_dd,
        trades=trade_results,
    )


def print_report(rpt: BacktestReport, show_trades: int = 20) -> None:
    print(f"\n{'=' * 60}")
    print(f"BACKTEST {rpt.instrument}")
    print(f"{'=' * 60}")
    print(f"  Trades total      : {rpt.n_trades}")
    print(f"    Wins            : {rpt.n_wins}")
    print(f"    Losses          : {rpt.n_losses}")
    print(f"    No fill         : {rpt.n_no_fill}")
    print(f"    Pending         : {rpt.n_pending}")
    print(f"  Win rate          : {rpt.win_rate:.1f}%")
    print(f"  Profit factor     : {rpt.profit_factor:.2f}")
    print(f"  Avg win           : ${rpt.avg_win_usd:.2f}")
    print(f"  Avg loss          : ${rpt.avg_loss_usd:.2f}")
    print(f"  Max DD            : {rpt.max_drawdown_pct:.1f}%")
    print(f"  Balance initial   : ${rpt.initial_balance:.2f}")
    print(f"  Balance final     : ${rpt.final_balance:.2f}")
    print(f"  Return            : {rpt.return_pct:+.2f}%")

    if show_trades and rpt.trades:
        print(f"\n  === {min(show_trades, len(rpt.trades))} premiers trades ===")
        for t in rpt.trades[:show_trades]:
            s = t.setup
            print(
                f"    {s.ob_validation_ts} | {s.direction:8s} | score={t.score:3d} | "
                f"{t.outcome:8s} | pnl=${t.pnl_usd:+8.2f} | RR={s.rr:.2f}"
            )


if __name__ == "__main__":
    import sys

    # Backtest XAUUSD par defaut
    instrument = sys.argv[1] if len(sys.argv) > 1 else "XAUUSD"
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    rpt = backtest_instrument(instrument, days=days)
    print_report(rpt, show_trades=20)
