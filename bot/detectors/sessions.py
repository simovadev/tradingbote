"""Detecteurs de niveaux journaliers ICT.

- Asia Range (00:00-04:00 UTC) : high + low = liquidite cible #1 pour LN/NY
- Daily Open : open de la session journaliere (00:00 UTC) = pivot psychologique
- PDH / PDL : Previous Day High / Low = liquidites majeures que le smart money chasse
- Session highs/lows : highs/lows de London et NY session

Ces niveaux sont des "magnets" pour le prix. Si le bot voit un sweep d'un de ces
niveaux pendant LN ou NY, c'est un signal beaucoup plus probabilisant qu'un
sweep d'un simple swing local.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Literal

import pandas as pd

SessionName = Literal["asia", "london", "ny", "london_close"]


@dataclass
class DailyLevels:
    """Niveaux du jour calcules a partir des bougies M1."""
    day: pd.Timestamp                # jour cible (UTC midnight)

    # Asia range (00:00 - 04:00 UTC)
    asia_high: float | None = None
    asia_low: float | None = None
    asia_high_time: pd.Timestamp | None = None
    asia_low_time: pd.Timestamp | None = None

    # Open journalier
    daily_open: float | None = None

    # Previous day high/low (PDH / PDL)
    pdh: float | None = None
    pdl: float | None = None
    pdh_time: pd.Timestamp | None = None
    pdl_time: pd.Timestamp | None = None

    # Highs/lows par session
    london_high: float | None = None
    london_low: float | None = None
    ny_high: float | None = None
    ny_low: float | None = None


SESSION_HOURS = {
    "asia":         (time(0, 0),  time(4, 0)),
    "london":       (time(7, 0),  time(10, 0)),
    "ny":           (time(12, 0), time(17, 0)),
    "london_close": (time(15, 0), time(17, 0)),
}


def _session_extremes(df_day: pd.DataFrame, start: time, end: time) -> tuple[float | None, float | None, pd.Timestamp | None, pd.Timestamp | None]:
    """Retourne high/low + leurs timestamps pour une session donnee."""
    if df_day.empty:
        return None, None, None, None
    mask = (df_day.index.time >= start) & (df_day.index.time < end)
    sub = df_day[mask]
    if sub.empty:
        return None, None, None, None
    return (
        float(sub["high"].max()),
        float(sub["low"].min()),
        sub["high"].idxmax(),
        sub["low"].idxmin(),
    )


def compute_daily_levels(df_m1: pd.DataFrame, day: pd.Timestamp) -> DailyLevels:
    """Calcule tous les niveaux journaliers a partir des bougies M1.

    Args:
        df_m1: bougies M1 avec index UTC.
        day: le jour cible (en UTC).
    """
    if day.tz is None:
        day = day.tz_localize("UTC")
    else:
        day = day.tz_convert("UTC")
    day = day.normalize()

    levels = DailyLevels(day=day)

    # Bougies du jour cible
    next_day = day + pd.Timedelta(days=1)
    df_today = df_m1.loc[(df_m1.index >= day) & (df_m1.index < next_day)]

    if df_today.empty:
        return levels

    # Daily open = 1ere bougie du jour
    levels.daily_open = float(df_today["open"].iloc[0])

    # Asia range
    (levels.asia_high, levels.asia_low,
     levels.asia_high_time, levels.asia_low_time) = _session_extremes(
        df_today, *SESSION_HOURS["asia"]
    )

    # London / NY session highs+lows
    levels.london_high, levels.london_low, *_ = _session_extremes(df_today, *SESSION_HOURS["london"])
    levels.ny_high, levels.ny_low, *_ = _session_extremes(df_today, *SESSION_HOURS["ny"])

    # PDH / PDL = high/low du jour precedent
    prev_day = day - pd.Timedelta(days=1)
    df_prev = df_m1.loc[(df_m1.index >= prev_day) & (df_m1.index < day)]
    if not df_prev.empty:
        levels.pdh = float(df_prev["high"].max())
        levels.pdl = float(df_prev["low"].min())
        levels.pdh_time = df_prev["high"].idxmax()
        levels.pdl_time = df_prev["low"].idxmin()

    return levels


def is_near(level: float | None, price: float, tolerance_pct: float = 0.05) -> bool:
    """Verifie si un prix est proche d'un niveau (tolerance en % du prix)."""
    if level is None:
        return False
    return abs(price - level) / price * 100 <= tolerance_pct


def find_matched_level(
    sweep_price: float,
    sweep_side: str,    # "high" ou "low"
    levels: DailyLevels,
    tolerance_pct: float = 0.05,
) -> tuple[str, float] | None:
    """Identifie quel niveau journalier majeur correspond au sweep.

    Retourne (nom_niveau, prix_niveau) ou None.
    Priorise : PDH/PDL > Asia high/low > London session > NY session
    """
    if sweep_side == "high":
        candidates = [
            ("PDH", levels.pdh),
            ("Asia High", levels.asia_high),
            ("London High", levels.london_high),
            ("NY High", levels.ny_high),
        ]
    else:
        candidates = [
            ("PDL", levels.pdl),
            ("Asia Low", levels.asia_low),
            ("London Low", levels.london_low),
            ("NY Low", levels.ny_low),
        ]

    for name, price in candidates:
        if price is None:
            continue
        if is_near(price, sweep_price, tolerance_pct):
            return name, price

    return None
