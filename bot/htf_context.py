"""Analyse du contexte HTF (H1) pour la regle 8 :
analyse top-down ponderee par la force du H1.

4 cas du trader :
1. H1 contradictoire -> REJECT absolu
2. H1 fortement aligne (OB + BB) -> flexible sur M1
3. H1 moderement aligne (OB seul) -> exiger setup M1 propre
4. H1 incertain -> exiger setup M1 textbook strict
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from bot.detectors.order_blocks import OrderBlock, check_breaker_block, detect_ob_from_break
from bot.detectors.structure import find_structure_break
from bot.detectors.liquidity import detect_all_sweeps
from bot.detectors.swings import detect_swings

HTFStrength = Literal["strong", "moderate", "weak", "contradictory", "neutral"]
Direction = Literal["bullish", "bearish"]


@dataclass
class HTFAnalysis:
    bias: Direction | None
    strength: HTFStrength
    active_obs: list[OrderBlock]
    notes: str


def analyze_htf(df_htf: pd.DataFrame, ltf_direction: Direction) -> HTFAnalysis:
    """Analyse le H1 et retourne sa force par rapport a la direction LTF voulue.

    Args:
        df_htf: bougies H1.
        ltf_direction: direction du trade envisage sur M1.

    Returns:
        HTFAnalysis avec strength qui pilotera l'exigence sur M1.
    """
    swings = detect_swings(df_htf, left=3, right=3)
    sweeps = detect_all_sweeps(df_htf, swings)

    # On regarde les BOS recents (= les 5 derniers)
    breaks = []
    for sw in sweeps[-10:]:
        sb = find_structure_break(df_htf, sw, swings)
        if sb is not None:
            breaks.append(sb)

    if not breaks:
        return HTFAnalysis(None, "neutral", [], "Aucun BOS recent en H1")

    # Bias HTF = direction du dernier BOS
    last_break = breaks[-1]
    htf_bias: Direction = last_break.direction

    # Verifie alignement avec direction LTF
    if htf_bias != ltf_direction:
        return HTFAnalysis(
            bias=htf_bias,
            strength="contradictory",
            active_obs=[],
            notes=f"H1 bias {htf_bias} contre LTF {ltf_direction}",
        )

    # Construit les OB du H1 dans le sens du LTF
    obs: list[OrderBlock] = []
    has_breaker = False
    for sb in breaks:
        if sb.direction != ltf_direction:
            continue
        ob = detect_ob_from_break(df_htf, sb)
        if ob is None:
            continue
        if check_breaker_block(df_htf, ob):
            ob.ob_type = "breaker_block"
            has_breaker = True
        obs.append(ob)

    if not obs:
        return HTFAnalysis(htf_bias, "weak", [], "Bias H1 OK mais pas d'OB exploitable")

    if has_breaker and len(obs) >= 2:
        return HTFAnalysis(htf_bias, "strong", obs, "OB H1 + BB H1 alignes")
    if has_breaker or len(obs) >= 1:
        return HTFAnalysis(htf_bias, "moderate", obs, "OB H1 aligne, pas de BB")

    return HTFAnalysis(htf_bias, "weak", obs, "Contexte H1 faible")
