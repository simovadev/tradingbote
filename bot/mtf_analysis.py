"""Analyse Multi-Timeframe complete sur H4 / H1 / M30 / M15 / M5 / M1.

Le bot regarde TOUS les TFs pour avoir une lecture complete du marche.
Cela ne veut pas dire que TOUS doivent etre alignes pour trader : on score.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from bot.detectors.liquidity import detect_all_sweeps
from bot.detectors.order_blocks import OrderBlock, check_breaker_block, detect_ob_from_break
from bot.detectors.structure import find_structure_break
from bot.detectors.swings import detect_swings

Direction = Literal["bullish", "bearish", "range"]


@dataclass
class TimeframeSnapshot:
    """Snapshot d'un TF a un instant donne."""
    tf: str
    bias: Direction
    last_bos_direction: Direction | None
    active_obs: list[OrderBlock] = field(default_factory=list)
    has_breaker: bool = False
    notes: str = ""


@dataclass
class MultiTFAnalysis:
    snapshots: dict[str, TimeframeSnapshot]    # tf -> snapshot
    htf_bias: Direction                         # bias agrege du HTF (H4+H1)
    mtf_alignment: bool                         # M30/M15 alignes avec HTF
    ltf_alignment: bool                         # M5 aligne avec HTF
    aligned_count: int                          # nb de TFs alignes avec direction trade
    total_count: int

    def is_contradictory(self, trade_direction: Direction) -> bool:
        """True si le HTF est franchement contradictoire."""
        return self.htf_bias != "range" and self.htf_bias != trade_direction

    def alignment_strength(self, trade_direction: Direction) -> Literal["full", "strong", "moderate", "weak", "contradictory"]:
        """Force d'alignement total des TFs avec la direction du trade."""
        if self.is_contradictory(trade_direction):
            return "contradictory"
        ratio = self.aligned_count / self.total_count if self.total_count else 0
        if ratio >= 0.85: return "full"
        if ratio >= 0.65: return "strong"
        if ratio >= 0.45: return "moderate"
        return "weak"


def _bias_from_df(df: pd.DataFrame) -> tuple[Direction, Direction | None, list[OrderBlock], bool]:
    """Calcule bias + dernier BOS + OBs actifs sur un DataFrame TF.

    Bias = composite de :
    - Direction du dernier BOS
    - Position du prix vs SMA50 (sous SMA = bearish, au-dessus = bullish)
    - Force du mouvement recent (si chute > rebond recent => range, pas bull)

    On retourne "range" si signaux contradictoires (evite le piege ICT classique).
    """
    swings = detect_swings(df, left=3, right=3)
    if not swings:
        return "range", None, [], False

    sweeps = detect_all_sweeps(df, swings)
    breaks = []
    for sw in sweeps[-15:]:
        sb = find_structure_break(df, sw, swings)
        if sb is not None:
            breaks.append(sb)

    if not breaks:
        return "range", None, [], False

    last_break = breaks[-1]
    bos_direction: Direction = last_break.direction

    # Signal SMA : position relative du dernier close vs moyenne longue
    sma_signal: Direction = "range"
    if len(df) >= 50:
        sma = df["close"].rolling(50).mean().iloc[-1]
        last_close = df["close"].iloc[-1]
        # Marge de 0.1% pour eviter les faux signaux
        if last_close > sma * 1.001:
            sma_signal = "bullish"
        elif last_close < sma * 0.999:
            sma_signal = "bearish"

    # Signal range : compare amplitude descente vs montee recente
    range_signal: Direction = "range"
    if len(df) >= 20:
        recent = df.tail(20)
        # Plus haut atteint dans la fenetre et plus bas
        rh, rl = recent["high"].max(), recent["low"].min()
        last_close = recent["close"].iloc[-1]
        pos_in_range = (last_close - rl) / (rh - rl) if rh > rl else 0.5
        # On regarde de quel cote on est venu
        idx_high = recent["high"].idxmax()
        idx_low = recent["low"].idxmin()
        # Si on est venu d'en haut (descente recente puis rebond), pos bas, signal range/baissier
        if idx_high < idx_low:
            # high arrive AVANT low -> on a chute -> bearish/range
            range_signal = "bearish" if pos_in_range < 0.6 else "range"
        else:
            # low arrive AVANT high -> on est monte -> bullish/range
            range_signal = "bullish" if pos_in_range > 0.4 else "range"

    # Combine : si 2 signaux sur 3 d'accord, on suit. Sinon range (prudence).
    votes = [bos_direction, sma_signal, range_signal]
    bullish_votes = sum(1 for v in votes if v == "bullish")
    bearish_votes = sum(1 for v in votes if v == "bearish")
    if bullish_votes >= 2:
        bias: Direction = "bullish"
    elif bearish_votes >= 2:
        bias = "bearish"
    else:
        bias = "range"

    obs: list[OrderBlock] = []
    has_breaker = False
    for sb in breaks[-5:]:
        ob = detect_ob_from_break(df, sb)
        if ob is None:
            continue
        if check_breaker_block(df, ob):
            ob.ob_type = "breaker_block"
            has_breaker = True
        obs.append(ob)

    return bias, last_break.direction, obs, has_breaker


def analyze_multi_tf(
    dfs: dict[str, pd.DataFrame],
    asof: pd.Timestamp,
    trade_direction: Direction,
) -> MultiTFAnalysis:
    """Analyse complete multi-TF a un instant donne (asof).

    Args:
        dfs: mapping TF -> DataFrame OHLC (deja charge).
        asof: instant d'evaluation (typiquement le moment du BOS sur M1).
        trade_direction: direction du trade envisage.
    """
    snapshots: dict[str, TimeframeSnapshot] = {}

    # On respecte un ordre raisonnable, les TFs manquants sont ignores
    for tf in ["H4", "H1", "M30", "M15", "M5", "M1"]:
        df = dfs.get(tf)
        if df is None or df.empty:
            continue
        sub = df.loc[:asof]
        if len(sub) < 20:
            continue
        bias, last_bos, obs, has_bb = _bias_from_df(sub)
        snapshots[tf] = TimeframeSnapshot(
            tf=tf,
            bias=bias,
            last_bos_direction=last_bos,
            active_obs=obs,
            has_breaker=has_bb,
            notes=f"Bias {bias}, {len(obs)} OB actifs, BB={has_bb}",
        )

    # HTF bias = priorite H4 > H1 si dispo, sinon H1 seul
    h4 = snapshots.get("H4")
    h1 = snapshots.get("H1")
    if h4 and h1 and h4.bias == h1.bias:
        htf_bias = h4.bias
    elif h4:
        htf_bias = h4.bias
    elif h1:
        htf_bias = h1.bias
    else:
        htf_bias = "range"

    # Comptage alignement
    aligned = 0
    total = 0
    for tf, snap in snapshots.items():
        if snap.bias == "range":
            continue
        total += 1
        if snap.bias == trade_direction:
            aligned += 1

    mtf_aligned = all(
        snapshots[tf].bias in (trade_direction, "range")
        for tf in ("M30", "M15") if tf in snapshots
    )
    ltf_aligned = snapshots.get("M5") is None or snapshots["M5"].bias in (trade_direction, "range")

    return MultiTFAnalysis(
        snapshots=snapshots,
        htf_bias=htf_bias,
        mtf_alignment=mtf_aligned,
        ltf_alignment=ltf_aligned,
        aligned_count=aligned,
        total_count=total or 1,
    )
