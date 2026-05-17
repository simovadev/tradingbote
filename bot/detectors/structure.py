"""Detection de Break of Structure (BOS) et Market Structure Shift (MSS).

REGLE #3 : apres le sweep, le prix DOIT casser un niveau de structure
dans le sens du trade. Sans cassure -> REJECT.

Definitions simplifiees :
- Apres un sweep de liquidite "low" -> on cherche une cassure de structure
  HAUSSIERE (le prix casse le dernier swing high formé avant le sweep).
- Apres un sweep de liquidite "high" -> cassure BAISSIERE
  (le prix casse le dernier swing low formé avant le sweep).

BOS vs MSS : pour notre algo on les unifie (les deux confirment un retournement
ou continuation), avec un flag pour distinguer plus tard si besoin.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from bot.detectors.liquidity import Sweep
from bot.detectors.swings import Swing

BreakDirection = Literal["bullish", "bearish"]


@dataclass
class StructureBreak:
    """Une cassure de structure confirmee."""
    direction: BreakDirection
    broken_swing: Swing      # le swing qui a ete casse
    break_index: int         # index de la bougie qui a casse
    break_time: pd.Timestamp
    break_price: float       # close de la bougie qui casse
    sweep: Sweep             # le sweep qui a precede (cause -> effet)


def find_structure_break(
    df: pd.DataFrame,
    sweep: Sweep,
    swings: list[Swing],
    max_lookforward: int = 50,
) -> StructureBreak | None:
    """Cherche une cassure de structure apres un sweep.

    Args:
        df: OHLC complet.
        sweep: le sweep qui sert de point de depart.
        swings: tous les swings detectes sur df.
        max_lookforward: nombre de bougies max apres le sweep pour trouver la cassure.

    Logique :
    - Si sweep cote "low" (sweep des lows -> retournement haussier attendu) :
        * Identifier le dernier swing HIGH AVANT le sweep
        * Chercher la 1ere bougie apres le sweep qui CLOTURE au-dessus
    - Si sweep cote "high" (sweep des highs -> retournement baissier attendu) :
        * Identifier le dernier swing LOW AVANT le sweep
        * Chercher la 1ere bougie apres le sweep qui CLOTURE en-dessous

    Returns:
        StructureBreak ou None.
    """
    direction: BreakDirection = "bullish" if sweep.side == "low" else "bearish"
    target_kind = "high" if direction == "bullish" else "low"

    # Swings du bon type formes AVANT le sweep
    candidates = [
        s for s in swings
        if s.kind == target_kind and s.index < sweep.candle_index
    ]
    if not candidates:
        return None

    # Le dernier swing avant le sweep = le niveau a casser
    target_swing = candidates[-1]

    # Fenetre de recherche apres le sweep
    end_index = min(sweep.candle_index + 1 + max_lookforward, len(df))
    sub = df.iloc[sweep.candle_index + 1:end_index]
    if sub.empty:
        return None

    if direction == "bullish":
        mask = sub["close"] > target_swing.price
    else:
        mask = sub["close"] < target_swing.price

    if not mask.any():
        return None

    break_time = mask.idxmax()  # 1er True
    break_candle = sub.loc[break_time]
    break_idx = int(df.index.get_loc(break_time))

    return StructureBreak(
        direction=direction,
        broken_swing=target_swing,
        break_index=break_idx,
        break_time=break_time,
        break_price=float(break_candle["close"]),
        sweep=sweep,
    )
