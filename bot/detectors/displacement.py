"""Detection du Displacement = qualite de la bougie de cassure.

Un VRAI OB est suivi d'un mouvement IMPULSIF : grosse bougie avec gros corps,
peu de meche, qui casse la structure d'un coup.

Un FAUX OB est suivi d'un mouvement mou (petits corps, mèches partout).

Filtre essentiel pour eliminer les OB "qui passent pas le test".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

DisplacementStrength = Literal["strong", "moderate", "weak"]


@dataclass
class DisplacementAnalysis:
    strength: DisplacementStrength
    body_ratio: float        # corps / range total de la bougie de cassure
    range_vs_avg: float      # range bougie cassure / ATR
    detail: str


def analyze_displacement(
    df: pd.DataFrame,
    break_index: int,
    atr_lookback: int = 14,
) -> DisplacementAnalysis:
    """Analyse la qualite du mouvement de cassure.

    Args:
        df: OHLC.
        break_index: index de la bougie qui a casse la structure.
        atr_lookback: nb bougies pour calculer l'ATR (volatilite moyenne).

    Returns:
        DisplacementAnalysis avec strength + metriques.
    """
    if break_index >= len(df) or break_index < 1:
        return DisplacementAnalysis("weak", 0.0, 0.0, "Index invalide")

    row = df.iloc[break_index]
    body = abs(row["close"] - row["open"])
    rng = row["high"] - row["low"]
    body_ratio = (body / rng) if rng > 0 else 0.0

    # ATR simplifie (vraie definition: True Range, ici on simplifie sur high-low)
    start = max(0, break_index - atr_lookback)
    avg_range = df.iloc[start:break_index]["high"].sub(df.iloc[start:break_index]["low"]).mean()
    range_vs_avg = (rng / avg_range) if avg_range and avg_range > 0 else 0.0

    if body_ratio >= 0.7 and range_vs_avg >= 1.5:
        return DisplacementAnalysis(
            "strong", float(body_ratio), float(range_vs_avg),
            f"Corps {body_ratio*100:.0f}% du range, taille x{range_vs_avg:.1f} vs moyenne",
        )
    if body_ratio >= 0.5 and range_vs_avg >= 1.0:
        return DisplacementAnalysis(
            "moderate", float(body_ratio), float(range_vs_avg),
            f"Corps {body_ratio*100:.0f}% du range, taille x{range_vs_avg:.1f} vs moyenne",
        )
    return DisplacementAnalysis(
        "weak", float(body_ratio), float(range_vs_avg),
        f"Mouvement mou : corps {body_ratio*100:.0f}%, taille x{range_vs_avg:.1f}",
    )
