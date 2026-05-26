"""Order Block V18 — Detection STRICTE selon definition Soufiane 2026-05-26.

Definition verrouillee (cf V18_DEFINITIONS_ICT_SMC.md) :

OB BULLISH :
  1. Sweep d'un swing low : meche d'une bougie descend sous un swing low recent
  2. La bougie de sweep DOIT etre BAISSIERE (close < open) — sinon ignore
  3. Groupe OB = bougie de sweep + bougies BAISSIERES CONSECUTIVES juste avant
     - MIN 2 bougies dans le groupe
     - MAX illimite (tant que consecutivement baissieres)
  4. Bornes :
     - ob_low  = LOW de la meche de la bougie de sweep (meche INCLUSE en bas)
     - ob_high = MAX(open, close) de toutes les bougies du groupe
                 (corps uniquement, meches du haut EXCLUES)
  5. PENDING indefini (aucun timer)
  6. VALIDATION : une bougie cloture > ob_high → OB tradable
  7. REJECTION : si pendant le pending, une bougie a son LOW < ob_low
                 (meche suffit) → OB REJECTED A VIE

OB BEARISH : symetrique.

Difference cle vs V17 :
- V17 utilisait mode "mitigation" (low touche ob_high) = INVERSE de la regle Soufiane
- V17 n'avait pas le check d'invalidation "low < ob_low" pendant le pending
- V17 limitait a max_bars_after_sweep = pas indefini
- V17 max_group_size = 5 (Soufiane = illimite)
- V17 min_group_size = 1 (Soufiane = 2)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from bot_v2.concepts.liquidity import Sweep, Swing, find_sweeps, find_swings


OBDirection = Literal["bullish", "bearish"]


@dataclass(frozen=True)
class OrderBlockV18:
    """OB strict selon la definition Soufiane V18."""
    direction: OBDirection
    # Bornes du GROUPE
    group_start_index: int          # iloc 1ere bougie du groupe (la + ancienne)
    group_end_index: int            # iloc bougie de sweep (inclusif, fin du groupe)
    group_start_ts: pd.Timestamp
    group_end_ts: pd.Timestamp
    group_size: int                 # nombre de bougies dans le groupe (>=2)
    # Niveaux de prix de la ZONE TRADABLE
    ob_high: float                  # haut de la zone (selon definition stricte)
    ob_low: float                   # bas de la zone (selon definition stricte)
    # La bougie de VALIDATION (close > ob_high pour bullish)
    validation_index: int
    validation_ts: pd.Timestamp
    validation_close: float
    # Le sweep qui a declenche l'OB
    sweep: Sweep
    # Bornes raw du groupe (pour debug)
    group_max_high: float           # high max meches incluses du groupe
    group_min_low: float            # low min meches incluses du groupe


def detect_order_blocks_v18(
    df: pd.DataFrame,
    swings: list[Swing] | None = None,
    sweeps: list[Sweep] | None = None,
    swing_strength: int = 2,
    min_group_size: int = 2,
    max_lookback_validation: int = 10000,
    min_sweep_depth_atr: float = 0.0,
) -> list[OrderBlockV18]:
    """Detecte tous les OB V18 valides dans le DataFrame.

    Args:
        df: DataFrame avec colonnes open, high, low, close (index temporel).
        swings, sweeps: optionnels, calcules sinon.
        swing_strength: pour find_swings si recalcul (def=2).
        min_group_size: minimum 2 (decision Soufiane). Pas de max.
        max_lookback_validation: garde-fou pour eviter de boucler trop loin (def=10000 bougies).
        min_sweep_depth_atr: profondeur min du sweep en ATR (0 = pas de filtre).
    """
    if swings is None:
        swings = find_swings(df, strength=swing_strength)
    if sweeps is None:
        sweeps = find_sweeps(df, swings, min_depth_atr=min_sweep_depth_atr)

    obs: list[OrderBlockV18] = []

    for sweep in sweeps:
        if sweep.direction == "bullish":
            ob = _try_bullish_ob_v18(df, sweep, min_group_size, max_lookback_validation)
        else:
            ob = _try_bearish_ob_v18(df, sweep, min_group_size, max_lookback_validation)
        if ob is not None:
            obs.append(ob)

    return _deduplicate(obs)


def _try_bullish_ob_v18(
    df: pd.DataFrame,
    sweep: Sweep,
    min_group_size: int,
    max_lookback: int,
) -> OrderBlockV18 | None:
    """Cherche un OB bullish V18 a partir d'un sweep bullish (low pris)."""
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    n = len(df)
    sweep_idx = sweep.sweep_index

    # ===== ETAPE 1 : la bougie de sweep DOIT etre BAISSIERE =====
    if closes[sweep_idx] >= opens[sweep_idx]:
        # Sweep en bougie verte (close >= open) = pas d'OB selon Soufiane
        return None

    # ===== ETAPE 2 : construction du groupe baissier =====
    # La bougie de sweep est la FIN du groupe. On remonte les baissieres consecutives.
    group_end = sweep_idx
    group_start = group_end
    while group_start > 0 and closes[group_start - 1] < opens[group_start - 1]:
        group_start -= 1

    group_size = group_end - group_start + 1
    if group_size < min_group_size:
        return None

    # ===== ETAPE 3 : bornes de la zone OB =====
    # ob_low = LOW de la meche de la bougie de sweep (meche INCLUSE en bas)
    ob_low = float(lows[sweep_idx])
    # ob_high = MAX(open, close) de toutes les bougies du groupe (corps uniquement)
    bodies_high = []
    for i in range(group_start, group_end + 1):
        bodies_high.append(max(opens[i], closes[i]))
    ob_high = float(max(bodies_high))

    if ob_high <= ob_low:
        return None  # OB degenere

    # Pour debug / cohérence
    group_max_high = float(max(highs[group_start:group_end + 1]))
    group_min_low = float(min(lows[group_start:group_end + 1]))

    # ===== ETAPE 4-5 : pending indefini, validation par close > ob_high
    # ===== ET rejection si meche < ob_low avant validation =====
    val_end = min(sweep_idx + 1 + max_lookback, n)
    validation_idx = None
    for j in range(sweep_idx + 1, val_end):
        # CHECK 1 (priorite) : rejection si meche descend sous ob_low
        if lows[j] < ob_low:
            return None  # OB REJECTED A VIE
        # CHECK 2 : validation si close > ob_high
        if closes[j] > ob_high:
            validation_idx = j
            break

    if validation_idx is None:
        return None  # pas valide dans le lookback

    return OrderBlockV18(
        direction="bullish",
        group_start_index=int(group_start),
        group_end_index=int(group_end),
        group_start_ts=df.index[group_start],
        group_end_ts=df.index[group_end],
        group_size=int(group_size),
        ob_high=ob_high,
        ob_low=ob_low,
        validation_index=int(validation_idx),
        validation_ts=df.index[validation_idx],
        validation_close=float(closes[validation_idx]),
        sweep=sweep,
        group_max_high=group_max_high,
        group_min_low=group_min_low,
    )


def _try_bearish_ob_v18(
    df: pd.DataFrame,
    sweep: Sweep,
    min_group_size: int,
    max_lookback: int,
) -> OrderBlockV18 | None:
    """Symetrique strict du bullish : sweep d'un swing high par bougie HAUSSIERE."""
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    n = len(df)
    sweep_idx = sweep.sweep_index

    # ===== ETAPE 1 : la bougie de sweep DOIT etre HAUSSIERE =====
    if closes[sweep_idx] <= opens[sweep_idx]:
        return None

    # ===== ETAPE 2 : groupe haussier consecutif terminant par le sweep =====
    group_end = sweep_idx
    group_start = group_end
    while group_start > 0 and closes[group_start - 1] > opens[group_start - 1]:
        group_start -= 1

    group_size = group_end - group_start + 1
    if group_size < min_group_size:
        return None

    # ===== ETAPE 3 : bornes =====
    # ob_high = HIGH de la meche du sweep (meche INCLUSE en haut)
    ob_high = float(highs[sweep_idx])
    # ob_low = MIN(open, close) du groupe (corps uniquement, meches basses EXCLUES)
    bodies_low = []
    for i in range(group_start, group_end + 1):
        bodies_low.append(min(opens[i], closes[i]))
    ob_low = float(min(bodies_low))

    if ob_high <= ob_low:
        return None

    group_max_high = float(max(highs[group_start:group_end + 1]))
    group_min_low = float(min(lows[group_start:group_end + 1]))

    # ===== ETAPE 4-5 : pending + validation + rejection =====
    val_end = min(sweep_idx + 1 + max_lookback, n)
    validation_idx = None
    for j in range(sweep_idx + 1, val_end):
        # CHECK 1 : rejection si meche monte au-dessus de ob_high
        if highs[j] > ob_high:
            return None
        # CHECK 2 : validation si close < ob_low
        if closes[j] < ob_low:
            validation_idx = j
            break

    if validation_idx is None:
        return None

    return OrderBlockV18(
        direction="bearish",
        group_start_index=int(group_start),
        group_end_index=int(group_end),
        group_start_ts=df.index[group_start],
        group_end_ts=df.index[group_end],
        group_size=int(group_size),
        ob_high=ob_high,
        ob_low=ob_low,
        validation_index=int(validation_idx),
        validation_ts=df.index[validation_idx],
        validation_close=float(closes[validation_idx]),
        sweep=sweep,
        group_max_high=group_max_high,
        group_min_low=group_min_low,
    )


def _deduplicate(obs: list[OrderBlockV18]) -> list[OrderBlockV18]:
    """Supprime les OB doublons (meme groupe + meme direction)."""
    seen: set[tuple] = set()
    out: list[OrderBlockV18] = []
    for ob in obs:
        key = (ob.direction, ob.group_start_index, ob.group_end_index, ob.validation_index)
        if key in seen:
            continue
        seen.add(key)
        out.append(ob)
    return out
