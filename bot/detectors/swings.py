"""Detection des swing highs / swing lows (pivots).

Un swing high a l'index i = bougie dont le high est strictement superieur
aux N bougies a gauche ET aux N bougies a droite.

C'est la fondation de tout le reste (sweep, BOS, OB se calculent par rapport
aux swings).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

SwingKind = Literal["high", "low"]


@dataclass
class Swing:
    """Un pivot identifie sur la serie."""
    index: int           # index positionnel dans le DataFrame
    timestamp: pd.Timestamp
    price: float
    kind: SwingKind


def detect_swings(df: pd.DataFrame, left: int = 3, right: int = 3) -> list[Swing]:
    """Detecte les swing highs et swing lows.

    Args:
        df: DataFrame OHLC indexe par timestamp.
        left: nombre de bougies a gauche que le pivot doit dominer.
        right: nombre de bougies a droite (= "confirmation lag").
                Avec right=3, un swing n'est confirme que 3 bougies plus tard
                -> evite de detecter des pivots qui n'en sont pas encore.

    Returns:
        Liste de Swing tries par index croissant.
    """
    if len(df) < left + right + 1:
        return []

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    times = df.index

    swings: list[Swing] = []

    for i in range(left, len(df) - right):
        h = highs[i]
        # Swing high: domine strictement gauche et droite
        if h > highs[i - left:i].max() and h > highs[i + 1:i + right + 1].max():
            swings.append(Swing(i, times[i], float(h), "high"))
            continue

        lo = lows[i]
        # Swing low: domine strictement (en bas) gauche et droite
        if lo < lows[i - left:i].min() and lo < lows[i + 1:i + right + 1].min():
            swings.append(Swing(i, times[i], float(lo), "low"))

    return swings


def last_swings(swings: list[Swing], kind: SwingKind | None = None, n: int = 5) -> list[Swing]:
    """Recupere les N derniers swings (filtrable par type)."""
    filtered = [s for s in swings if kind is None or s.kind == kind]
    return filtered[-n:]


def equal_levels(swings: list[Swing], kind: SwingKind, tolerance_pct: float = 0.05) -> list[list[Swing]]:
    """Identifie les groupes de swings de meme niveau (equal highs / equal lows).

    Args:
        tolerance_pct: ecart max entre 2 swings pour les considerer egaux (en %).
                       0.05 = 0.05% (typique sur XAUUSD = ~2-3$ d'ecart a 4700).

    Returns:
        Liste de groupes. Chaque groupe = >= 2 swings consideres egaux.
        Ce sont les pools de liquidite typiques en ICT.
    """
    filtered = [s for s in swings if s.kind == kind]
    if len(filtered) < 2:
        return []

    groups: list[list[Swing]] = []
    used: set[int] = set()

    for i, s1 in enumerate(filtered):
        if i in used:
            continue
        group = [s1]
        for j in range(i + 1, len(filtered)):
            if j in used:
                continue
            s2 = filtered[j]
            diff_pct = abs(s2.price - s1.price) / s1.price * 100
            if diff_pct <= tolerance_pct:
                group.append(s2)
                used.add(j)
        if len(group) >= 2:
            groups.append(group)
            used.add(i)

    return groups
