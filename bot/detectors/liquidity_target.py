"""Trouve la prochaine liquidite cible (TP) pour un trade donne.

Approche ICT :
- LONG : cherche la prochaine liquidite EN HAUT (high) du prix d'entree
- SHORT : cherche la prochaine liquidite EN BAS (low) du prix d'entree

Priorite :
1. PDH (Previous Day High) si LONG / PDL si SHORT
2. Asia High si LONG / Asia Low si SHORT
3. London/NY session High/Low
4. Plus haut swing M15 majeur (left/right 5)
5. Plus haut swing M1 majeur

Retourne la liquidite la PLUS PROCHE du prix d'entree dans la bonne direction.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time, timedelta
from typing import Literal

import pandas as pd

from bot.detectors.swings import detect_swings


@dataclass
class LiquidityTarget:
    price: float
    source: str               # "PDH", "PDL", "Asia High", "Asia Low", "Swing M15", "Swing M1"
    distance: float           # distance absolue au prix d'entree


def _session_extremes(df: pd.DataFrame, start: time, end: time, side: str) -> float | None:
    """Retourne high (side='high') ou low (side='low') d'une session sur la journee de df."""
    if df.empty:
        return None
    mask = (df.index.time >= start) & (df.index.time < end)
    sub = df[mask]
    if sub.empty:
        return None
    return float(sub["high"].max() if side == "high" else sub["low"].min())


def find_next_liquidity_target(
    df_m1: pd.DataFrame,
    df_m15: pd.DataFrame | None,
    entry_price: float,
    entry_time: pd.Timestamp,
    direction: Literal["bullish", "bearish"],
    min_distance_pct: float = 0.05,    # distance min en % du prix (eviter target trop proche)
    max_distance_pct: float = 5.0,      # distance max en % du prix (eviter target trop loin)
) -> LiquidityTarget | None:
    """Trouve la liquidite cible la plus proche dans la bonne direction.

    Returns:
        LiquidityTarget ou None si aucune cible raisonnable trouvee.
    """
    if entry_time.tz is None:
        entry_time = entry_time.tz_localize("UTC")

    candidates: list[LiquidityTarget] = []

    # ---------- 1. PDH / PDL ----------
    # Jour precedent = jour J-1 (en UTC, jours ouvres)
    today_start = entry_time.normalize()
    prev_day_start = today_start - timedelta(days=1)
    # Si veille = samedi/dimanche, on remonte
    while prev_day_start.weekday() >= 5:
        prev_day_start -= timedelta(days=1)
    prev_day_end = prev_day_start + timedelta(days=1)
    prev_day = df_m1.loc[(df_m1.index >= prev_day_start) & (df_m1.index < prev_day_end)]
    if not prev_day.empty:
        if direction == "bullish":
            pdh = float(prev_day["high"].max())
            if pdh > entry_price:
                candidates.append(LiquidityTarget(pdh, "PDH", pdh - entry_price))
        else:
            pdl = float(prev_day["low"].min())
            if pdl < entry_price:
                candidates.append(LiquidityTarget(pdl, "PDL", entry_price - pdl))

    # ---------- 2. Asia/London/NY de la journee ----------
    today_data = df_m1.loc[(df_m1.index >= today_start) & (df_m1.index < entry_time)]
    if not today_data.empty:
        sessions = [
            ("Asia", time(0, 0), time(4, 0)),
            ("London", time(7, 0), time(10, 0)),
            ("NY", time(12, 0), time(16, 0)),
        ]
        for name, start, end in sessions:
            if direction == "bullish":
                hi = _session_extremes(today_data, start, end, "high")
                if hi is not None and hi > entry_price:
                    candidates.append(LiquidityTarget(hi, f"{name} High", hi - entry_price))
            else:
                lo = _session_extremes(today_data, start, end, "low")
                if lo is not None and lo < entry_price:
                    candidates.append(LiquidityTarget(lo, f"{name} Low", entry_price - lo))

    # ---------- 3. Swings M15 majeurs (left/right 5) ----------
    if df_m15 is not None:
        m15_before = df_m15.loc[df_m15.index <= entry_time].tail(200)
        if len(m15_before) >= 20:
            m15_swings = detect_swings(m15_before, left=5, right=5)
            target_kind = "high" if direction == "bullish" else "low"
            for s in m15_swings:
                if s.kind != target_kind:
                    continue
                if direction == "bullish" and s.price > entry_price:
                    candidates.append(LiquidityTarget(s.price, "Swing M15", s.price - entry_price))
                elif direction == "bearish" and s.price < entry_price:
                    candidates.append(LiquidityTarget(s.price, "Swing M15", entry_price - s.price))

    # ---------- 4. Swings M1 majeurs (left/right 5) ----------
    entry_loc = df_m1.index.get_indexer([entry_time], method="nearest")[0]
    m1_before = df_m1.iloc[max(0, entry_loc - 500):entry_loc]
    if len(m1_before) >= 30:
        m1_swings = detect_swings(m1_before, left=5, right=5)
        target_kind = "high" if direction == "bullish" else "low"
        for s in m1_swings[-20:]:    # 20 derniers swings
            if s.kind != target_kind:
                continue
            if direction == "bullish" and s.price > entry_price:
                candidates.append(LiquidityTarget(s.price, "Swing M1", s.price - entry_price))
            elif direction == "bearish" and s.price < entry_price:
                candidates.append(LiquidityTarget(s.price, "Swing M1", entry_price - s.price))

    # ---------- Filtre distance ----------
    min_dist = entry_price * min_distance_pct / 100
    max_dist = entry_price * max_distance_pct / 100

    valid = [c for c in candidates if min_dist <= c.distance <= max_dist]
    if not valid:
        return None

    # ---------- Priorise : PDH/PDL > Asia > Session > Swing M15 > Swing M1 ----------
    priority = {"PDH": 1, "PDL": 1, "Asia High": 2, "Asia Low": 2,
                "London High": 3, "London Low": 3, "NY High": 3, "NY Low": 3,
                "Swing M15": 4, "Swing M1": 5}

    # On groupe par priorite, dans chaque groupe on prend la plus proche
    valid.sort(key=lambda c: (priority.get(c.source, 99), c.distance))
    return valid[0]
