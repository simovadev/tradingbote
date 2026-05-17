"""Optimal Trade Entry (OTE) — concept ICT.

Apres une cassure de structure (BOS), le prix retrace souvent jusqu'aux
niveaux Fibonacci 62%-79%. La zone 62-79% est la "OTE zone" : zone d'entree
optimale qui combine retracement profond + risque/reward favorable.

- 62% = niveau d'entree minimum (deja interessant)
- 70.5% = sweet spot
- 79% = limite extreme (au dela = setup invalide, on n'aurait pas du voir
        le retracement aller si loin)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class OTEAnalysis:
    """Analyse de la position de l'entree dans la zone OTE."""
    swing_low: float
    swing_high: float
    entry: float
    retracement_pct: float
    in_ote: bool
    quality: Literal["sweet_spot", "premium", "extreme", "shallow", "out"]
    detail: str


def analyze_ote(
    entry_price: float,
    swing_high: float,
    swing_low: float,
    direction: Literal["bullish", "bearish"],
) -> OTEAnalysis:
    """Calcule la position de l'entree dans la zone OTE.

    Pour LONG :
        retracement = 1 - (entry - low) / (high - low)
        On veut retracement entre 62% et 79% (= entry sous le mid plus serieusement)

    Pour SHORT :
        retracement = (entry - low) / (high - low)
    """
    rng = swing_high - swing_low
    if rng <= 0:
        return OTEAnalysis(
            swing_low, swing_high, entry_price, 0.0,
            False, "out", "Range invalide",
        )

    if direction == "bullish":
        # Entry sous high, plus c'est bas plus c'est profond
        retr = (swing_high - entry_price) / rng
    else:
        # Entry au-dessus low, plus c'est haut plus c'est profond
        retr = (entry_price - swing_low) / rng

    if 0.68 <= retr <= 0.73:
        quality = "sweet_spot"
    elif 0.62 <= retr <= 0.79:
        quality = "premium"
    elif 0.79 < retr <= 0.88:
        quality = "extreme"
    elif 0.50 <= retr < 0.62:
        quality = "shallow"
    else:
        quality = "out"

    in_ote = 0.62 <= retr <= 0.79

    return OTEAnalysis(
        swing_low=swing_low,
        swing_high=swing_high,
        entry=entry_price,
        retracement_pct=float(retr),
        in_ote=in_ote,
        quality=quality,
        detail=f"Retracement {retr*100:.0f}% ({quality})",
    )
