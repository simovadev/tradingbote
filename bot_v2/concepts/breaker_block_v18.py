"""Breaker Block V18 — derive d'un OB casse (Soufiane 2026-05-26).

Definition :

BB BULLISH (issu d'un OB BEARISH casse) :
  1. Un OB bearish existe avec bornes [ob_low ; ob_high] selon la def V18
  2. Pendant que l'OB bearish est en pending (avant cloture < ob_low pour valider),
     une bougie cloture > ob_high → l'OB est INVALIDE en tant qu'OB bearish
     MAIS devient un BB BULLISH
  3. BB bullish herite des bornes [ob_low ; ob_high] de l'ex-OB
  4. PENDING indefini pour le BB
  5. RETEST : le prix doit redescendre dans la zone pour qu'on puisse trader
  6. REINVALIDATION : si une bougie a son LOW < BB_low (meche suffit) → BB MORT A VIE

BB BEARISH : symetrique a partir d'un OB BULLISH casse (close < ob_low).

NOTE IMPLEMENTATION :
- L'OB V18 dans order_block_v18.py est VALIDE quand close < ob_low (bearish) ou
  close > ob_high (bullish). Si pendant le pending la MECHE OPPOSEE est cassee,
  on `return None`. Pour generer les BB, on doit re-parcourir en cherchant cet
  evenement (la cassure inverse) au lieu de retourner None.

Donc detect_breaker_blocks_v18 reprend la logique de detect_order_blocks_v18 mais
inverse les critères d'invalidation / validation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from bot_v2.concepts.liquidity import Sweep, Swing, find_sweeps, find_swings


BBDirection = Literal["bullish", "bearish"]


@dataclass(frozen=True)
class BreakerBlockV18:
    """BB issu d'un OB casse (bornes heritees de l'ex-OB)."""
    direction: BBDirection                  # "bullish" si ex-OB bearish casse, etc.
    # Bornes (heritees de l'ex-OB)
    bb_high: float                          # = ob_high de l'ex-OB
    bb_low: float                           # = ob_low de l'ex-OB
    # Le groupe OB d'origine (pour traçabilite)
    group_start_index: int
    group_end_index: int
    group_start_ts: pd.Timestamp
    group_end_ts: pd.Timestamp
    group_size: int
    # La bougie qui a CASSE l'OB → declenche le BB
    break_index: int
    break_ts: pd.Timestamp
    break_close: float
    # Le sweep d'origine de l'ex-OB
    sweep: Sweep


def detect_breaker_blocks_v18(
    df: pd.DataFrame,
    swings: list[Swing] | None = None,
    sweeps: list[Sweep] | None = None,
    swing_strength: int = 2,
    min_group_size: int = 2,
    max_lookback_break: int = 10000,
    min_sweep_depth_atr: float = 0.0,
) -> list[BreakerBlockV18]:
    """Detecte les BB V18.

    Logique :
      - Pour chaque sweep bullish : on construit un OB bearish CANDIDAT
        (bougie de sweep haussiere + haussieres consecutives avant).
        Pendant son pending, si meche > ob_high (= ce qui invalide l'OB bearish),
        ON IGNORE.
        Si une bougie cloture > ob_high (cassure haussiere), on cree un BB BULLISH.

      - Pour chaque sweep bearish : symetrique → BB BEARISH si cassure < ob_low.

    ⚠ Subtilite : selon la def V18 de l'OB bearish, la bougie de sweep doit etre
    haussiere. Donc on cherche les meme structures que detect_order_blocks_v18,
    mais on remplace le critere de validation par le critere de cassure inverse.
    """
    if swings is None:
        swings = find_swings(df, strength=swing_strength)
    if sweeps is None:
        sweeps = find_sweeps(df, swings, min_depth_atr=min_sweep_depth_atr)

    bbs: list[BreakerBlockV18] = []

    for sweep in sweeps:
        if sweep.direction == "bullish":
            # Sweep bullish (low pris) → cherche un OB bearish CANDIDAT...
            # Non : sweep bullish → OB bullish candidat (bougie sweep baissiere).
            # Un OB bullish casse vers le BAS (close < ob_low) → BB BEARISH.
            bb = _try_bb_from_bullish_ob_break(df, sweep, min_group_size, max_lookback_break)
        else:
            # Sweep bearish (high pris) → OB bearish candidat (bougie sweep haussiere).
            # Un OB bearish casse vers le HAUT (close > ob_high) → BB BULLISH.
            bb = _try_bb_from_bearish_ob_break(df, sweep, min_group_size, max_lookback_break)
        if bb is not None:
            bbs.append(bb)

    return _deduplicate_bb(bbs)


def _try_bb_from_bearish_ob_break(
    df: pd.DataFrame,
    sweep: Sweep,
    min_group_size: int,
    max_lookback: int,
) -> BreakerBlockV18 | None:
    """OB bearish (sweep d'un swing high par bougie haussiere) CASSE vers le HAUT
    → BB BULLISH.
    """
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    n = len(df)
    sweep_idx = sweep.sweep_index

    # Bougie de sweep doit etre HAUSSIERE (pour OB bearish)
    if closes[sweep_idx] <= opens[sweep_idx]:
        return None

    # Groupe haussier consecutif terminant par le sweep
    group_end = sweep_idx
    group_start = group_end
    while group_start > 0 and closes[group_start - 1] > opens[group_start - 1]:
        group_start -= 1

    group_size = group_end - group_start + 1
    if group_size < min_group_size:
        return None

    # Bornes de l'OB bearish (selon V18)
    ob_high = float(highs[sweep_idx])  # meche INCLUSE
    bodies_low = [min(opens[i], closes[i]) for i in range(group_start, group_end + 1)]
    ob_low = float(min(bodies_low))
    if ob_high <= ob_low:
        return None

    # On cherche une CASSURE HAUSSIERE (close > ob_high) AVANT une validation
    # bearish (close < ob_low). Si pendant le pending la meche depasse ob_high
    # ça invaliderait l'OB bearish — mais ici justement, c'est ce qu'on attend
    # pour declencher le BB. Subtilite : la regle V18 de l'OB invalide si meche
    # depasse, donc la CLOTURE > ob_high est notre signal de "casse + BB".
    val_end = min(sweep_idx + 1 + max_lookback, n)
    break_idx = None
    for j in range(sweep_idx + 1, val_end):
        # Si l'OB bearish est valide d'abord (close < ob_low) → c'etait un vrai OB
        # bearish, pas un BB. On stoppe.
        if closes[j] < ob_low:
            return None
        # CASSURE : close > ob_high → naissance du BB bullish
        if closes[j] > ob_high:
            break_idx = j
            break

    if break_idx is None:
        return None

    return BreakerBlockV18(
        direction="bullish",
        bb_high=ob_high,
        bb_low=ob_low,
        group_start_index=int(group_start),
        group_end_index=int(group_end),
        group_start_ts=df.index[group_start],
        group_end_ts=df.index[group_end],
        group_size=int(group_size),
        break_index=int(break_idx),
        break_ts=df.index[break_idx],
        break_close=float(closes[break_idx]),
        sweep=sweep,
    )


def _try_bb_from_bullish_ob_break(
    df: pd.DataFrame,
    sweep: Sweep,
    min_group_size: int,
    max_lookback: int,
) -> BreakerBlockV18 | None:
    """OB bullish casse vers le BAS → BB BEARISH (symetrique)."""
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    n = len(df)
    sweep_idx = sweep.sweep_index

    # Bougie de sweep DOIT etre baissiere (pour OB bullish)
    if closes[sweep_idx] >= opens[sweep_idx]:
        return None

    # Groupe baissier consecutif terminant par le sweep
    group_end = sweep_idx
    group_start = group_end
    while group_start > 0 and closes[group_start - 1] < opens[group_start - 1]:
        group_start -= 1

    group_size = group_end - group_start + 1
    if group_size < min_group_size:
        return None

    # Bornes de l'OB bullish (V18)
    ob_low = float(lows[sweep_idx])  # meche INCLUSE en bas
    bodies_high = [max(opens[i], closes[i]) for i in range(group_start, group_end + 1)]
    ob_high = float(max(bodies_high))
    if ob_high <= ob_low:
        return None

    # On cherche une CASSURE BAISSIERE (close < ob_low) avant validation bullish
    val_end = min(sweep_idx + 1 + max_lookback, n)
    break_idx = None
    for j in range(sweep_idx + 1, val_end):
        # Si OB bullish valide d'abord (close > ob_high) → vrai OB, pas BB
        if closes[j] > ob_high:
            return None
        # Cassure baissiere → BB bearish
        if closes[j] < ob_low:
            break_idx = j
            break

    if break_idx is None:
        return None

    return BreakerBlockV18(
        direction="bearish",
        bb_high=ob_high,
        bb_low=ob_low,
        group_start_index=int(group_start),
        group_end_index=int(group_end),
        group_start_ts=df.index[group_start],
        group_end_ts=df.index[group_end],
        group_size=int(group_size),
        break_index=int(break_idx),
        break_ts=df.index[break_idx],
        break_close=float(closes[break_idx]),
        sweep=sweep,
    )


def _deduplicate_bb(bbs: list[BreakerBlockV18]) -> list[BreakerBlockV18]:
    seen: set[tuple] = set()
    out: list[BreakerBlockV18] = []
    for bb in bbs:
        key = (bb.direction, bb.group_start_index, bb.group_end_index, bb.break_index)
        if key in seen:
            continue
        seen.add(key)
        out.append(bb)
    return out
