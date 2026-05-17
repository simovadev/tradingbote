"""Mitigation Order Block.

Un OB "fresh" (non mitigated) = jamais retouche depuis sa formation = haute proba.
Un OB deja mitigated = deja teste = la liquidite a deja ete consommee, proba reduite.

On verifie : entre l'index de l'OB et le BOS, est-ce que le prix est revenu
dans la zone OB ?
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from bot.detectors.order_blocks import OrderBlock


@dataclass
class MitigationStatus:
    is_fresh: bool
    touches_before_break: int
    detail: str


def check_mitigation(df: pd.DataFrame, ob: OrderBlock) -> MitigationStatus:
    """Verifie si l'OB a ete touche entre sa formation et le BOS.

    Si le prix est revenu dans la zone OB entre ob.candle_index et break_index
    (exclus), alors l'OB n'est PAS fresh.
    """
    start = ob.candle_index + 1
    end = ob.structure_break.break_index
    if start >= end:
        return MitigationStatus(True, 0, "Pas de bougies entre OB et BOS")

    sub = df.iloc[start:end]
    touches = (
        (sub["high"] >= ob.zone_low) & (sub["low"] <= ob.zone_high)
    ).sum()

    is_fresh = touches == 0
    return MitigationStatus(
        is_fresh=is_fresh,
        touches_before_break=int(touches),
        detail=f"{touches} touches avant BOS" if touches else "OB fresh (jamais retouche)",
    )
