"""Detection des Fair Value Gaps (FVG) et Inverse FVG (iFVG).

FVG haussier (BISI - Buy Side Imbalance, Sell Side Inefficiency) :
    3 bougies consecutives ou high[i-1] < low[i+1]
    -> il existe un "gap" entre [high(i-1), low(i+1)]
    -> zone d'achat potentielle

FVG baissier (SIBI - Sell Side Imbalance, Buy Side Inefficiency) :
    low[i-1] > high[i+1] -> gap baissier

iFVG : un FVG qui a ete rempli, puis le prix s'est inverse.
       Le FVG "inverse" devient une zone tradable dans l'autre sens.

REGLE TRADER : indicateur "FVG/iFVG (Nephew_Sam_)" actif sur TradingView.
FVG en confluence avec OB = setup beaucoup plus fort.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

FVGKind = Literal["bullish", "bearish"]


@dataclass
class FVG:
    kind: FVGKind
    candle_index: int        # index de la bougie centrale (i)
    candle_time: pd.Timestamp
    zone_high: float
    zone_low: float
    filled: bool = False     # True si le prix est revenu dans la zone
    inverted: bool = False   # True si iFVG (rempli + retournement)


def detect_fvgs(df: pd.DataFrame, lookback: int = 200) -> list[FVG]:
    """Detecte tous les FVG dans les `lookback` dernieres bougies.

    Pour chaque triplet (i-1, i, i+1) :
    - FVG bullish si high[i-1] < low[i+1]
      zone = [high(i-1), low(i+1)]
    - FVG bearish si low[i-1] > high[i+1]
      zone = [high(i+1), low(i-1)]
    """
    if len(df) < 3:
        return []

    start = max(1, len(df) - lookback)
    end = len(df) - 1
    out: list[FVG] = []
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    times = df.index

    for i in range(start, end):
        # bullish
        if highs[i - 1] < lows[i + 1]:
            out.append(FVG(
                kind="bullish",
                candle_index=i,
                candle_time=times[i],
                zone_high=float(lows[i + 1]),
                zone_low=float(highs[i - 1]),
            ))
        # bearish
        elif lows[i - 1] > highs[i + 1]:
            out.append(FVG(
                kind="bearish",
                candle_index=i,
                candle_time=times[i],
                zone_high=float(lows[i - 1]),
                zone_low=float(highs[i + 1]),
            ))

    # Calcule filled/inverted pour chaque FVG
    for fvg in out:
        sub = df.iloc[fvg.candle_index + 2:]
        if sub.empty:
            continue
        if fvg.kind == "bullish":
            entered = (sub["low"] <= fvg.zone_high).any()
            fvg.filled = bool(entered)
            # iFVG : apres remplissage, close en-dessous de zone_low -> inverte
            if fvg.filled:
                fvg.inverted = bool((sub["close"] < fvg.zone_low).any())
        else:
            entered = (sub["high"] >= fvg.zone_low).any()
            fvg.filled = bool(entered)
            if fvg.filled:
                fvg.inverted = bool((sub["close"] > fvg.zone_high).any())

    return out


def find_fvg_in_zone(
    fvgs: list[FVG],
    zone_high: float,
    zone_low: float,
    direction: FVGKind,
    before_index: int,
) -> FVG | None:
    """Cherche un FVG du bon sens qui chevauche la zone donnee (OB) et qui
    a ete forme avant `before_index`."""
    for fvg in fvgs:
        if fvg.kind != direction:
            continue
        if fvg.candle_index > before_index:
            continue
        # chevauchement
        if fvg.zone_high < zone_low or fvg.zone_low > zone_high:
            continue
        return fvg
    return None
