"""Mode entrainement humain multi-actif :
- Pioche un jour aleatoire + un instrument aleatoire
- Scanne ce jour avec la strategy V3
- Retourne le meilleur trade
"""
from __future__ import annotations

import secrets
from datetime import timedelta

import pandas as pd

from bot.portfolio import FictivePortfolio, SimulatedTrade
from bot.strategy import scan_all
from config import INITIAL_BALANCE, INSTRUMENTS, TIMEFRAMES
from data.fetch import load_cached
from db.models import SessionLocal, TrainingTrade


MAX_SKIP_DAYS = 80


def load_all_dfs(instrument: str = "XAUUSD") -> dict[str, pd.DataFrame]:
    """Charge tous les TFs disponibles pour un instrument."""
    out: dict[str, pd.DataFrame] = {}
    for tf in TIMEFRAMES:
        try:
            out[tf] = load_cached(instrument, tf)
        except FileNotFoundError:
            continue
    return out


def load_correlates_m1(primary: str) -> dict[str, pd.DataFrame]:
    """Charge le M1 des actifs correles pour la detection SMT."""
    from config import SMT_PAIRS
    out: dict[str, pd.DataFrame] = {}
    for other, _ctype in SMT_PAIRS.get(primary, []):
        try:
            out[other] = load_cached(other, "M1")
        except FileNotFoundError:
            continue
    return out


def available_days(df_m1: pd.DataFrame) -> list[pd.Timestamp]:
    if df_m1.empty:
        return []
    return list(pd.Series(df_m1.index.normalize().unique()).tolist())


def pick_random_day(dfs: dict[str, pd.DataFrame]) -> pd.Timestamp | None:
    df_m1 = dfs.get("M1")
    if df_m1 is None or df_m1.empty:
        return None
    days = available_days(df_m1)
    if not days:
        return None
    # Exclure week-ends
    days = [d for d in days if (d.tz_localize("UTC") if d.tz is None else d).weekday() < 5]
    if not days:
        return None
    return days[secrets.randbelow(len(days))]


def slice_day(dfs: dict[str, pd.DataFrame], day: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """Decoupe les DFs pour ne garder que le contexte autour du jour cible."""
    if day.tz is None:
        day = day.tz_localize("UTC")
    end = day + timedelta(days=1)

    spans = {"M1": 2, "M5": 3, "M15": 7, "M30": 7, "H1": 30, "H4": 90}
    out: dict[str, pd.DataFrame] = {}
    for tf, df in dfs.items():
        days_back = spans.get(tf, 7)
        start = day - timedelta(days=days_back)
        sliced = df.loc[start:end]
        if not sliced.empty:
            out[tf] = sliced
    return out


def find_best_trade_of_day(
    dfs_day: dict[str, pd.DataFrame],
    instrument: str,
    correlates_m1: dict[str, pd.DataFrame],
) -> SimulatedTrade | None:
    """Scanne le jour et retourne le trade le plus pertinent (priorise filtres durs OK)."""
    portfolio = FictivePortfolio(balance=INITIAL_BALANCE)
    trades = scan_all(dfs_day, portfolio, instrument=instrument, correlates_m1=correlates_m1, score_threshold=0)
    if not trades:
        return None

    def hard_pass_count(t: SimulatedTrade) -> int:
        hf = (t.setup_analysis or {}).get("hard_filters", [])
        return sum(1 for f in hf if f.get("passed"))

    def hard_total(t: SimulatedTrade) -> int:
        return len((t.setup_analysis or {}).get("hard_filters", [])) or 1

    def sort_key(t: SimulatedTrade):
        pass_ratio = hard_pass_count(t) / hard_total(t)
        rr_capped = min(t.risk_reward, 10.0)
        return (pass_ratio, t.would_trade, t.score, rr_capped)

    return max(trades, key=sort_key)


def next_training_trade(instrument: str | None = None) -> tuple[TrainingTrade | None, pd.Timestamp | None, int]:
    """Pioche un jour + instrument, scanne, retourne le meilleur trade.

    Si instrument is None, pioche aussi un instrument au hasard.
    """
    # Choix de l'instrument
    if instrument is None:
        names = list(INSTRUMENTS.keys())
        instrument = names[secrets.randbelow(len(names))]

    dfs = load_all_dfs(instrument)
    if "M1" not in dfs:
        return None, None, 0

    correlates_m1 = load_correlates_m1(instrument)
    skipped = 0

    for _ in range(MAX_SKIP_DAYS):
        day = pick_random_day(dfs)
        if day is None:
            return None, None, skipped

        dfs_day = slice_day(dfs, day)
        trade = find_best_trade_of_day(dfs_day, instrument, correlates_m1)
        if trade is None:
            skipped += 1
            continue

        with SessionLocal() as s:
            count = s.query(TrainingTrade).count()
            record = TrainingTrade(
                id=trade.id,
                session_index=count,
                sampled_day=day.to_pydatetime().replace(tzinfo=None),
                direction=trade.direction,
                instrument=trade.instrument,
                entry_time=trade.entry_time.to_pydatetime().replace(tzinfo=None),
                entry_price=trade.entry_price,
                stop_loss=trade.stop_loss,
                take_profit=trade.take_profit,
                risk_reward=trade.risk_reward,
                lot_size=trade.lot_size,
                status=trade.status,
                exit_time=trade.exit_time.to_pydatetime().replace(tzinfo=None) if trade.exit_time is not None else None,
                exit_price=trade.exit_price,
                pnl=trade.pnl,
                score=trade.score,
                algo_verdict=trade.verdict,
                would_trade=trade.would_trade,
                htf_strength=trade.htf_strength,
                ob_type=trade.ob_type,
                touch_count=trade.touch_count,
                setup_analysis=trade.setup_analysis,
                chart_overlays=trade.chart_overlays,
                notes=trade.notes,
            )
            s.add(record)
            s.commit()
            s.refresh(record)
            return record, day, skipped

    return None, None, skipped
