"""Recherche du target TP intelligent - approche Vizion Trading.

Priorite des targets pour un trade :
1. PDH/PDL (Previous Day High/Low) si dans la bonne direction et raisonnable
2. Asia High/Low (si pas encore sweep ce jour-la)
3. London/NY session high/low
4. Swing structurel majeur (left/right >= 5) M15 dans la bonne direction
5. Fallback : prochain swing M1 left/right >= 3 dans la bonne direction
6. Si rien : pas de target (refus du trade Vizion)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from bot.detectors.sessions import DailyLevels
from bot.detectors.swings import Swing, detect_swings


@dataclass
class TargetCandidate:
    price: float
    source: str
    distance_pts: float


def find_best_target(
    direction: Literal["bullish", "bearish"],
    entry_price: float,
    entry_index: int,
    df_m1: pd.DataFrame,
    df_m15: pd.DataFrame | None,
    daily_levels: DailyLevels,
    max_distance_atr_mult: float = 30.0,    # max 30x ATR comme distance (evite targets trop loins)
) -> TargetCandidate | None:
    """Trouve le meilleur target dans la direction du trade.

    Approche Vizion : on cherche la prochaine VRAIE liquidite qui n'a pas
    encore ete prise dans la direction du trade.
    """
    candidates: list[TargetCandidate] = []

    # ATR rapide pour borner la recherche
    if entry_index >= 14:
        recent = df_m1.iloc[max(0, entry_index - 14):entry_index]
        atr = (recent["high"] - recent["low"]).mean()
        max_dist = atr * max_distance_atr_mult
    else:
        max_dist = entry_price * 0.05  # 5% du prix max

    def consider(price: float | None, source: str) -> None:
        if price is None:
            return
        # Doit etre du bon cote de l'entree
        if direction == "bullish" and price <= entry_price:
            return
        if direction == "bearish" and price >= entry_price:
            return
        dist = abs(price - entry_price)
        if dist > max_dist:
            return
        candidates.append(TargetCandidate(price, source, dist))

    # 1. PDH/PDL
    if direction == "bullish":
        consider(daily_levels.pdh, "PDH")
        consider(daily_levels.asia_high, "Asia High")
        consider(daily_levels.london_high, "London High")
        consider(daily_levels.ny_high, "NY High")
    else:
        consider(daily_levels.pdl, "PDL")
        consider(daily_levels.asia_low, "Asia Low")
        consider(daily_levels.london_low, "London Low")
        consider(daily_levels.ny_low, "NY Low")

    # 2. Swings structurels M15 (left/right = 5, plus solide)
    if df_m15 is not None and not df_m15.empty:
        # Filtre M15 jusqu'au temps de l'entry
        entry_time = df_m1.index[entry_index]
        df_m15_sub = df_m15.loc[df_m15.index <= entry_time]
        if len(df_m15_sub) >= 20:
            swings_m15 = detect_swings(df_m15_sub, left=5, right=5)
            target_kind = "high" if direction == "bullish" else "low"
            recent_m15_swings = [s for s in swings_m15 if s.kind == target_kind][-10:]
            for s in recent_m15_swings:
                consider(s.price, f"Swing M15 majeur")

    # 3. Swings M1 majeurs (left/right = 5)
    if entry_index >= 30:
        sub_m1 = df_m1.iloc[max(0, entry_index - 200):entry_index]
        swings_m1 = detect_swings(sub_m1, left=5, right=5)
        target_kind = "high" if direction == "bullish" else "low"
        recent_swings = [s for s in swings_m1 if s.kind == target_kind][-5:]
        for s in recent_swings:
            consider(s.price, "Swing M1 majeur")

    if not candidates:
        return None

    # On prend LE PLUS PROCHE (= la 1ere liquidite a etre prise)
    return min(candidates, key=lambda c: c.distance_pts)
