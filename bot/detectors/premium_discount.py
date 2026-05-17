"""Premium / Discount zone (concept ICT).

Sur un range donne (du dernier swing oppose au sweep) :
- 50% = equilibrium
- Au-dessus de 50% = PREMIUM
- En-dessous de 50% = DISCOUNT

Pour un long : on veut acheter en zone DISCOUNT (sous le mid).
Pour un short : on veut vendre en zone PREMIUM (au-dessus du mid).

Une entree contre cette logique = setup plus faible.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ZonePosition = Literal["deep_discount", "discount", "equilibrium", "premium", "deep_premium"]


@dataclass
class PremiumDiscountAnalysis:
    position: ZonePosition
    range_high: float
    range_low: float
    range_mid: float
    entry_pct: float          # ou se situe l'entree dans le range (0.0 = bottom, 1.0 = top)
    favorable: bool           # True si la zone est favorable au trade
    detail: str


def analyze_position(
    entry_price: float,
    range_high: float,
    range_low: float,
    direction: Literal["bullish", "bearish"],
) -> PremiumDiscountAnalysis:
    """Position de l'entree dans le range courant."""
    if range_high <= range_low:
        return PremiumDiscountAnalysis(
            "equilibrium", range_high, range_low, range_high, 0.5, False,
            "Range invalide",
        )

    rng = range_high - range_low
    pct = (entry_price - range_low) / rng
    mid = (range_high + range_low) / 2

    if pct <= 0.25: pos: ZonePosition = "deep_discount"
    elif pct <= 0.4: pos = "discount"
    elif pct < 0.6: pos = "equilibrium"
    elif pct < 0.75: pos = "premium"
    else: pos = "deep_premium"

    # Favorable pour long = discount, pour short = premium
    if direction == "bullish":
        favorable = pos in ("deep_discount", "discount")
    else:
        favorable = pos in ("premium", "deep_premium")

    return PremiumDiscountAnalysis(
        position=pos,
        range_high=range_high,
        range_low=range_low,
        range_mid=mid,
        entry_pct=pct,
        favorable=favorable,
        detail=f"Entree a {pct*100:.0f}% du range ({pos})",
    )
