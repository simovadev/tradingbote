"""Detection des Order Blocks (OB) et Breaker Blocks (BB).

REGLE #2 et #6 du trader :
- OB = derniere bougie OPPOSEE au mouvement impulsif qui a cause le BOS.
  Long  : derniere bougie BAISSIERE avant la cassure haussiere -> OB haussier.
  Short : derniere bougie HAUSSIERE avant la cassure baissiere -> OB baissier.

- Mèche = OB (regle 6) : si la bougie du sweep est elle-meme un long candle
  avec grande meche, on traite la meche elle-meme comme zone OB.

- Breaker Block (regle 5) : un OB qui a ete CASSE par le prix devient un BB
  au retour. Souvent plus probabilisant.

La "zone" de l'OB :
- OB haussier : [low_meche ; high_corps] de la bougie OB (zone d'achat).
- OB baissier : [low_corps ; high_meche] de la bougie OB (zone de vente).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from bot.detectors.structure import StructureBreak

OBKind = Literal["bullish", "bearish"]
OBType = Literal["order_block", "wick_ob", "breaker_block"]


@dataclass
class OrderBlock:
    """Une zone d'Order Block (ou Breaker)."""
    kind: OBKind             # sens du trade attendu au retour
    ob_type: OBType          # OB classique, OB de meche, ou BB
    candle_index: int        # bougie source
    candle_time: pd.Timestamp
    zone_high: float
    zone_low: float
    structure_break: StructureBreak  # le BOS qui valide cet OB

    @property
    def zone_mid(self) -> float:
        """Le 50% de la zone (niveau d'entree typique)."""
        return (self.zone_high + self.zone_low) / 2

    @property
    def zone_size(self) -> float:
        return self.zone_high - self.zone_low


def _is_bullish_candle(row: pd.Series) -> bool:
    return row["close"] > row["open"]


def _is_bearish_candle(row: pd.Series) -> bool:
    return row["close"] < row["open"]


def _body_size(row: pd.Series) -> float:
    return abs(row["close"] - row["open"])


def _candle_range(row: pd.Series) -> float:
    return row["high"] - row["low"]


def _is_long_wick_candle(row: pd.Series, ratio: float = 0.6) -> bool:
    """Bougie avec une meche significative (>= ratio * range total)."""
    rng = _candle_range(row)
    if rng == 0:
        return False
    body = _body_size(row)
    wick_total = rng - body
    return wick_total >= ratio * rng


def detect_ob_from_break(
    df: pd.DataFrame,
    sb: StructureBreak,
    max_lookback: int = 15,
) -> OrderBlock | None:
    """Trouve l'Order Block correspondant a un Structure Break.

    Logique :
    - On part de la bougie de cassure et on remonte en arriere.
    - Long  (sb bullish) : 1ere bougie BAISSIERE rencontree = OB haussier.
    - Short (sb bearish) : 1ere bougie HAUSSIERE rencontree = OB baissier.
    - Si la bougie du sweep elle-meme est un long-wick candle, on prend ce wick.

    Args:
        max_lookback: nb max de bougies a remonter avant d'abandonner.

    Returns:
        OrderBlock ou None.
    """
    # Cas special : meche = OB (regle 6)
    sweep_candle = df.iloc[sb.sweep.candle_index]
    if _is_long_wick_candle(sweep_candle):
        if sb.direction == "bullish":
            # Mèche basse = OB haussier (zone d'achat dans la meche)
            zone_low = float(sweep_candle["low"])
            zone_high = float(min(sweep_candle["open"], sweep_candle["close"]))
        else:
            # Mèche haute = OB baissier (zone de vente dans la meche)
            zone_high = float(sweep_candle["high"])
            zone_low = float(max(sweep_candle["open"], sweep_candle["close"]))

        if zone_high > zone_low:
            return OrderBlock(
                kind=sb.direction,
                ob_type="wick_ob",
                candle_index=sb.sweep.candle_index,
                candle_time=sb.sweep.candle_time,
                zone_high=zone_high,
                zone_low=zone_low,
                structure_break=sb,
            )

    # Cas classique : on cherche la derniere bougie opposee avant le BOS
    start = sb.break_index - 1
    stop = max(0, start - max_lookback)

    for i in range(start, stop - 1, -1):
        row = df.iloc[i]

        if sb.direction == "bullish" and _is_bearish_candle(row):
            # OB haussier : zone = [low ; high] de cette bougie baissiere
            return OrderBlock(
                kind="bullish",
                ob_type="order_block",
                candle_index=i,
                candle_time=df.index[i],
                zone_high=float(row["high"]),
                zone_low=float(row["low"]),
                structure_break=sb,
            )

        if sb.direction == "bearish" and _is_bullish_candle(row):
            return OrderBlock(
                kind="bearish",
                ob_type="order_block",
                candle_index=i,
                candle_time=df.index[i],
                zone_high=float(row["high"]),
                zone_low=float(row["low"]),
                structure_break=sb,
            )

    return None


def check_breaker_block(df: pd.DataFrame, ob: OrderBlock) -> bool:
    """Verifie si un OB a ete casse, le transformant en Breaker Block.

    Casse :
    - OB haussier : prix CLOTURE en-dessous de zone_low apres ob.candle_index.
    - OB baissier : prix CLOTURE au-dessus de zone_high apres ob.candle_index.

    Si oui, on retourne True et le caller peut changer ob.ob_type = "breaker_block".
    """
    if ob.candle_index + 1 >= len(df):
        return False

    sub = df.iloc[ob.candle_index + 1:]
    if ob.kind == "bullish":
        return bool((sub["close"] < ob.zone_low).any())
    else:
        return bool((sub["close"] > ob.zone_high).any())
