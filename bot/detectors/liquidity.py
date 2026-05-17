"""Detection de liquidite (pools) et de sweep.

REGLE #1 du trader : sans sweep confirme = REJECT automatique.

Definitions :
- Pool de liquidite = niveau ou plusieurs swings se concentrent (equal H/L)
  OU swing isole suffisamment marquant.
- Sweep = bougie dont la meche depasse le niveau, mais qui CLOTURE de l'autre cote.
  C'est la "prise de liquidite" : les stops ont saute, le prix a rejete.

Sans cette confirmation (depassement + retour), c'est juste une cassure, pas un sweep.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from bot.detectors.swings import Swing, equal_levels

SweepSide = Literal["high", "low"]


@dataclass
class LiquidityPool:
    """Niveau de liquidite (un equal level ou un swing marquant)."""
    price: float
    side: SweepSide          # "high" = pool au-dessus, "low" = pool en-dessous
    source_swings: list[Swing]
    first_index: int         # index de la 1ere bougie qui a forme ce pool
    last_index: int          # index de la derniere bougie qui a forme ce pool


@dataclass
class Sweep:
    """Une prise de liquidite confirmee."""
    pool: LiquidityPool
    candle_index: int        # index de la bougie qui a fait le sweep
    candle_time: pd.Timestamp
    wick_extreme: float      # extreme de la meche (le plus haut/bas atteint)
    close: float             # close de la bougie (doit etre du bon cote)
    side: SweepSide          # "high" = sweep au-dessus, "low" = sweep en-dessous


def build_pools(swings: list[Swing], tolerance_pct: float = 0.05) -> list[LiquidityPool]:
    """Construit les pools de liquidite (equal highs et equal lows)."""
    pools: list[LiquidityPool] = []

    for side in ("high", "low"):
        groups = equal_levels(swings, side, tolerance_pct)
        for group in groups:
            avg_price = sum(s.price for s in group) / len(group)
            pools.append(LiquidityPool(
                price=float(avg_price),
                side=side,
                source_swings=group,
                first_index=group[0].index,
                last_index=group[-1].index,
            ))

    # Ajoute aussi les swings "isoles marquants" (un swing seul mais bien net)
    # car parfois la liquidite c'est juste un swing high recent
    for s in swings:
        # Skip si deja dans un pool equal
        in_pool = any(s in p.source_swings for p in pools)
        if in_pool:
            continue
        pools.append(LiquidityPool(
            price=s.price,
            side=s.kind,
            source_swings=[s],
            first_index=s.index,
            last_index=s.index,
        ))

    return pools


def detect_sweep(
    df: pd.DataFrame,
    pool: LiquidityPool,
    from_index: int | None = None,
    wick_buffer_pct: float = 0.001,
) -> Sweep | None:
    """Cherche un sweep du pool dans le df.

    Args:
        df: OHLC.
        pool: le niveau a sweeper.
        from_index: ne regarde qu'a partir de cet index (defaut = apres la derniere
                    bougie du pool).
        wick_buffer_pct: le sweep doit depasser d'au moins ce % (filtre le bruit).
                         0.001% = ~0.05$ sur XAUUSD a 4700 (tres permissif).

    Returns:
        Le 1er sweep trouve, ou None.

    Logique :
    - Pool "high" : bougie dont high > pool.price (avec buffer) MAIS close < pool.price
    - Pool "low"  : bougie dont low  < pool.price (avec buffer) MAIS close > pool.price
    """
    start = from_index if from_index is not None else pool.last_index + 1
    if start >= len(df):
        return None

    sub = df.iloc[start:]
    buffer = pool.price * wick_buffer_pct / 100

    if pool.side == "high":
        threshold = pool.price + buffer
        mask = (sub["high"] > threshold) & (sub["close"] < pool.price)
    else:
        threshold = pool.price - buffer
        mask = (sub["low"] < threshold) & (sub["close"] > pool.price)

    if not mask.any():
        return None

    # Premiere occurence
    idx_rel = mask.idxmax()  # premier True
    candle = sub.loc[idx_rel]
    idx_abs = df.index.get_loc(idx_rel)

    return Sweep(
        pool=pool,
        candle_index=int(idx_abs),
        candle_time=idx_rel,
        wick_extreme=float(candle["high"] if pool.side == "high" else candle["low"]),
        close=float(candle["close"]),
        side=pool.side,
    )


def detect_all_sweeps(df: pd.DataFrame, swings: list[Swing]) -> list[Sweep]:
    """Detecte tous les sweeps de tous les pools dans le df."""
    pools = build_pools(swings)
    sweeps: list[Sweep] = []
    for pool in pools:
        sweep = detect_sweep(df, pool)
        if sweep is not None:
            sweeps.append(sweep)
    # Tri par index chronologique
    sweeps.sort(key=lambda s: s.candle_index)
    return sweeps
