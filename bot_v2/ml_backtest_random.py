"""Backtest random month avec ML filter — dashboard live.

Pour chaque appel :
- Pioche un mois aleatoire dans la fenetre OOS test (post 2025-06)
- Lance Vizion + ML filter (seuil par actif)
- Renvoie : liste trades + stats

OOS uniquement = 2025-06 -> 2026-05 (11 mois), le ML n'a JAMAIS vu ces donnees.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd

from bot_v2.config import SMT_PAIRS, get_param
from bot_v2.concepts.breaker import detect_breakers
from bot_v2.concepts.daily_bias import build_d1_from_h1
from bot_v2.concepts.fvg import detect_fvg
from bot_v2.concepts.htf_swings import collect_htf_swings
from bot_v2.concepts.liquidity import find_swings
from bot_v2.concepts.order_block import detect_order_blocks
from bot_v2.concepts.structure import detect_structure_breaks, detect_trend
from bot_v2.concepts.killzones import killzone_at
from bot_v2.data_loader import load
from bot_v2.pipeline import evaluate_ob
from bot_v2.backtest import simulate_trade
from bot_v2.trade_setup import compute_position_size, TradeSetup
from bot_v2 import ml_filter


OOS_START = pd.Timestamp("2025-06-11", tz="UTC")  # apres split train+val du ml_train


@dataclass
class RandomBacktestTrade:
    ts: str
    direction: str
    entry: float
    stop_loss: float
    take_profit: float
    rr: float
    score: int
    ml_proba: float
    outcome: str         # WIN / LOSS / NO_FILL
    pnl_usd: float
    bars_to_exit: int | None
    killzone: str | None


@dataclass
class RandomBacktestReport:
    instrument: str
    month: str           # ex "2025-09"
    period_start: str
    period_end: str
    threshold: float
    n_candidates_vizion: int   # trades Vizion brut (avant ML)
    n_taken_ml: int            # trades pris (ML proba >= threshold)
    n_wins: int
    n_losses: int
    n_no_fill: int
    win_rate: float
    total_pnl_usd: float
    avg_rr: float
    trades: list[dict]


def run_random_month(
    instrument: str = "XAUUSD",
    threshold: float | None = None,
    balance: float = 60.0,
    risk_pct: float = 0.10,
    seed: int | None = None,
) -> RandomBacktestReport:
    """Pioche un mois aleatoire OOS et backteste avec ML filter."""
    if seed is not None:
        random.seed(seed)

    # 1. Charge XAUUSD M1 et determine la fenetre OOS
    df_ltf = load(instrument, "M1")
    earliest_end = df_ltf.index[-1]
    if earliest_end < OOS_START:
        raise ValueError(f"Donnees ne couvrent pas la fenetre OOS post {OOS_START}")

    # 2. Pioche un mois aleatoire dans [OOS_START, earliest_end - 30j]
    available_days = (earliest_end - OOS_START).days - 30
    if available_days <= 0:
        raise ValueError("Fenetre OOS trop courte pour 1 mois")
    offset_days = random.randint(0, available_days)
    period_start = OOS_START + pd.Timedelta(days=offset_days)
    period_end = period_start + pd.Timedelta(days=30)
    month_label = period_start.strftime("%Y-%m-%d")

    # 3. Charge tout le contexte Vizion (HTF, SMT, etc.)
    df_htf = load(instrument, "M15")
    try:
        df_htf2 = load(instrument, "H1")
    except Exception:
        df_htf2 = None
    try:
        df_d1 = load(instrument, "D1")
        if len(df_d1) < 10:
            raise FileNotFoundError
    except Exception:
        df_h1 = load(instrument, "H1")
        df_d1 = build_d1_from_h1(df_h1)

    htf_dfs_swings = {}
    for tf in ["H1", "H4", "D1"]:
        try:
            htf_dfs_swings[tf] = load(instrument, tf)
        except Exception:
            pass
    htf_swings = collect_htf_swings(htf_dfs_swings, swing_strength=3)

    mask = (df_ltf.index >= period_start) & (df_ltf.index <= period_end)
    df_ltf_w = df_ltf[mask]
    if len(df_ltf_w) < 100:
        raise ValueError(f"Pas assez de bougies sur {month_label}")

    correlated_dfs = {}
    for corr_name, corr_type in SMT_PAIRS.get(instrument, []):
        try:
            df_c = load(corr_name, "M1")
            mask_c = (df_c.index >= period_start) & (df_c.index <= period_end)
            correlated_dfs[corr_name] = (df_c[mask_c], corr_type)
        except Exception:
            continue

    # 4. Detect OBs + cache
    sws = get_param(instrument, "swing_strength_m1", 2)
    obs = detect_order_blocks(df_ltf_w, swing_strength=sws, max_group_size=2)
    cache = {
        "swings_ltf": find_swings(df_ltf_w, strength=sws),
        "fvgs_ltf": detect_fvg(df_ltf_w),
        "breakers_ltf": detect_breakers(df_ltf_w),
        "obs_htf": detect_order_blocks(df_htf),
    }
    cache["structure_breaks"] = detect_structure_breaks(df_ltf_w, swings=cache["swings_ltf"])
    cache["htf_trend"] = detect_trend(cache["swings_ltf"], lookback=6)
    if df_htf2 is not None:
        cache["obs_htf2"] = detect_order_blocks(df_htf2)

    df_h1_for_feu_vert = df_htf2 if df_htf2 is not None else None

    # PAS de pre-filtre KZ (user 2026-05-16) : pipeline accepte tous les OB
    obs_kz = obs

    # 5. Threshold
    thr = threshold if threshold is not None else ml_filter.ML_THRESHOLDS.get(
        instrument, ml_filter.DEFAULT_THRESHOLD
    )

    # 6. Evalue chaque OB Vizion + filtre ML
    trades: list[RandomBacktestTrade] = []
    n_vizion_ok = 0
    for ob in obs_kz:
        r = evaluate_ob(
            ob, df_ltf_w, df_htf, df_d1, instrument,
            ltf_name="M1", htf_name="M15",
            df_htf2=df_htf2, htf2_name="H1",
            correlated_dfs=correlated_dfs,
            htf_swings=htf_swings,
            df_h1=df_h1_for_feu_vert,
            min_score=0, min_quality=0,
            cache=cache,
        )
        if r.verdict != "TRADE" or r.trade_setup is None:
            continue
        n_vizion_ok += 1

        # ML filter
        take, proba = ml_filter.should_take(r, ob, instrument)
        if not take:
            continue

        # Simulate trade
        setup = r.trade_setup
        try:
            lots, risk_usd = compute_position_size(
                setup.entry_price, setup.stop_loss, instrument,
                balance=balance, risk_pct=risk_pct,
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
            tr = simulate_trade(sim_setup, df_ltf_w, ob.validation_index + 1)
            bars = (tr.exit_index - tr.fill_index) if (tr.exit_index and tr.fill_index) else None
            trades.append(RandomBacktestTrade(
                ts=str(ob.validation_ts),
                direction=ob.direction,
                entry=float(setup.entry_price),
                stop_loss=float(setup.stop_loss),
                take_profit=float(setup.take_profit),
                rr=float(setup.rr),
                score=r.score,
                ml_proba=round(proba, 3),
                outcome=tr.outcome,
                pnl_usd=round(tr.pnl_usd, 2),
                bars_to_exit=bars,
                killzone=r.killzone_name,
            ))
        except Exception:
            continue

    # 7. Stats
    wins = sum(1 for t in trades if t.outcome == "WIN")
    losses = sum(1 for t in trades if t.outcome == "LOSS")
    no_fill = sum(1 for t in trades if t.outcome == "NO_FILL")
    closed = wins + losses
    wr = (wins / closed * 100) if closed > 0 else 0.0
    total_pnl = sum(t.pnl_usd for t in trades)
    avg_rr = (sum(t.rr for t in trades) / len(trades)) if trades else 0.0

    return RandomBacktestReport(
        instrument=instrument,
        month=month_label,
        period_start=str(period_start),
        period_end=str(period_end),
        threshold=thr,
        n_candidates_vizion=n_vizion_ok,
        n_taken_ml=len(trades),
        n_wins=wins,
        n_losses=losses,
        n_no_fill=no_fill,
        win_rate=round(wr, 1),
        total_pnl_usd=round(total_pnl, 2),
        avg_rr=round(avg_rr, 2),
        trades=[asdict(t) for t in trades],
    )


if __name__ == "__main__":
    import json
    rep = run_random_month("XAUUSD")
    print(json.dumps(asdict(rep), indent=2, default=str))
