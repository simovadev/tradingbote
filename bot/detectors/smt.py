"""SMT Divergence (Smart Money Technique) - confirmation OB via actif correle.

Principe :
- LONG sur XAU (sweep low) : XAU fait new low, DXY ne fait PAS new high (inverse) -> divergence -> LONG confirme
- SHORT sur XAU (sweep high) : XAU fait new high, DXY ne fait PAS new low -> divergence -> SHORT confirme
- LONG sur NAS (sweep low) : NAS fait new low, GER40 ne fait PAS new low (positive) -> divergence -> LONG confirme

Fenetre comparable : on regarde les memes ~N bougies sur les deux actifs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd


@dataclass
class SMTResult:
    has_divergence: bool
    primary_extremum_price: float | None      # extremum du primaire dans la fenetre
    primary_extremum_time: pd.Timestamp | None
    correlate_extremum_price: float | None
    correlate_extremum_time: pd.Timestamp | None
    correlation_type: str                       # "positive" / "inverse"
    detail: str


def check_smt_divergence(
    primary_df: pd.DataFrame,
    correlate_df: pd.DataFrame,
    sweep_time: pd.Timestamp,
    trade_direction: Literal["bullish", "bearish"],
    correlation_type: Literal["positive", "inverse"] = "positive",
    lookback_minutes: int = 60,
) -> SMTResult:
    """Verifie la divergence SMT au moment du sweep.

    Args:
        primary_df: bougies M1 de l'actif primaire (celui qu'on trade)
        correlate_df: bougies M1 de l'actif correle
        sweep_time: timestamp du sweep (= push_end_time typiquement)
        trade_direction: direction du trade envisage
        correlation_type: positive ou inverse
        lookback_minutes: fenetre d'analyse en minutes avant le sweep
    """
    if primary_df.empty or correlate_df.empty:
        return SMTResult(False, None, None, None, None, correlation_type, "Donnees manquantes")

    # Fenetre : lookback_minutes avant sweep + 5 min apres
    sweep_ts = sweep_time if hasattr(sweep_time, 'tz') else pd.Timestamp(sweep_time)
    if sweep_ts.tz is None:
        sweep_ts = sweep_ts.tz_localize("UTC")

    start = sweep_ts - pd.Timedelta(minutes=lookback_minutes)
    end = sweep_ts + pd.Timedelta(minutes=5)

    p_win = primary_df.loc[(primary_df.index >= start) & (primary_df.index <= end)]
    c_win = correlate_df.loc[(correlate_df.index >= start) & (correlate_df.index <= end)]

    if len(p_win) < 10 or len(c_win) < 10:
        return SMTResult(False, None, None, None, None, correlation_type, "Pas assez de bougies")

    # Pour un LONG : sweep des lows. On regarde le LOW MIN sur la fenetre du primaire.
    # Pour un SHORT : sweep des highs. On regarde le HIGH MAX.

    if trade_direction == "bullish":
        # Le primaire a fait un new low recent ? On regarde le low max recent vs le low du sweep
        p_low = p_win["low"].min()
        p_low_time = p_win["low"].idxmin()
        # Le low doit etre dans la zone proche du sweep_time (< 10 min)
        if abs((p_low_time - sweep_ts).total_seconds()) > 10 * 60:
            return SMTResult(False, None, None, None, None, correlation_type, "Pas de new low recent sur primaire")

        # Que fait l'actif correle au meme moment ?
        if correlation_type == "positive":
            # Pour correlation positive : si primaire fait new low, correle DEVRAIT aussi -> divergence si NON
            c_low = c_win["low"].min()
            c_low_time = c_win["low"].idxmin()
            # Verifie : le correle a-t-il aussi fait son low recent ?
            time_diff = abs((c_low_time - sweep_ts).total_seconds())
            if time_diff > 15 * 60:
                # Le correle a fait son low ailleurs (loin) = divergence
                return SMTResult(True, float(p_low), p_low_time, float(c_low), c_low_time,
                                 correlation_type,
                                 f"Primaire new low @ {p_low_time.strftime('%H:%M')}, correle low @ {c_low_time.strftime('%H:%M')}")
        else:  # inverse
            # XAU fait new low, DXY (inverse) DEVRAIT faire new high. Si non -> divergence
            c_high = c_win["high"].max()
            c_high_time = c_win["high"].idxmax()
            time_diff = abs((c_high_time - sweep_ts).total_seconds())
            if time_diff > 15 * 60:
                return SMTResult(True, float(p_low), p_low_time, float(c_high), c_high_time,
                                 correlation_type,
                                 f"Primaire new low @ {p_low_time.strftime('%H:%M')}, correle high @ {c_high_time.strftime('%H:%M')}")

    else:  # bearish
        p_high = p_win["high"].max()
        p_high_time = p_win["high"].idxmax()
        if abs((p_high_time - sweep_ts).total_seconds()) > 10 * 60:
            return SMTResult(False, None, None, None, None, correlation_type, "Pas de new high recent sur primaire")

        if correlation_type == "positive":
            c_high = c_win["high"].max()
            c_high_time = c_win["high"].idxmax()
            time_diff = abs((c_high_time - sweep_ts).total_seconds())
            if time_diff > 15 * 60:
                return SMTResult(True, float(p_high), p_high_time, float(c_high), c_high_time,
                                 correlation_type,
                                 f"Primaire new high @ {p_high_time.strftime('%H:%M')}, correle high @ {c_high_time.strftime('%H:%M')}")
        else:  # inverse
            c_low = c_win["low"].min()
            c_low_time = c_win["low"].idxmin()
            time_diff = abs((c_low_time - sweep_ts).total_seconds())
            if time_diff > 15 * 60:
                return SMTResult(True, float(p_high), p_high_time, float(c_low), c_low_time,
                                 correlation_type,
                                 f"Primaire new high @ {p_high_time.strftime('%H:%M')}, correle low @ {c_low_time.strftime('%H:%M')}")

    return SMTResult(False, None, None, None, None, correlation_type, "Pas de divergence (mouvement aligne)")
