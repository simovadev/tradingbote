"""Detection des conditions d'entree sur un OB - approche Vizion Trading.

Principes :
- SL serre : sous le low (long) ou au-dessus du high (short) de l'OB + buffer adaptatif
- TP intelligent : prochaine VRAIE liquidite (PDH/PDL/Asia high-low/swing structurel)
  Si pas de target majeur visible, fallback sur le prochain swing structurel.
  PAS de RR fixe.
- Entree au mid OB (50%) au retour
- Accepte 1er ou 2eme touch (Vizion : si setup M1 propre, 1er touch valide)
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from bot.detectors.order_blocks import OrderBlock


@dataclass
class EntrySignal:
    ob: OrderBlock
    touch_count: int
    entry_index: int
    entry_time: pd.Timestamp
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    target_source: str          # "PDH" / "Asia High" / "swing M1 majeur" / ...


def _candle_touches_zone(row: pd.Series, ob: OrderBlock) -> bool:
    return not (row["high"] < ob.zone_low or row["low"] > ob.zone_high)


def find_entry(
    df: pd.DataFrame,
    ob: OrderBlock,
    target_price: float | None = None,
    target_source: str = "swing M1",
    min_rr: float = 1.0,
    max_rr: float = 5.0,            # Plafond : si liquidite trop loin, on rabote
    sl_buffer_pct: float = 0.10,
    sl_min_points: float = 3.0,
    require_second_touch: bool = False,
    max_lookforward: int = 120,
) -> EntrySignal | None:
    """Cherche le signal d'entree sur un OB - logique Vizion.

    Args:
        target_price: prix cible (= liquidite). Si None -> pas d'entree.
        target_source: nom humain du target (pour le log).
        min_rr: RR minimum. 1.0 par defaut (Vizion).
    """
    if target_price is None:
        return None

    start = ob.structure_break.break_index + 1
    end = min(start + max_lookforward, len(df))
    if start >= end:
        return None

    sub = df.iloc[start:end]
    touches: list[int] = []
    in_zone = False

    for ts, row in sub.iterrows():
        if _candle_touches_zone(row, ob):
            if not in_zone:
                idx_abs = int(df.index.get_loc(ts))
                touches.append(idx_abs)
                in_zone = True
        else:
            in_zone = False

    if not touches:
        return None

    target_touch_n = 2 if require_second_touch else 1
    if len(touches) < target_touch_n:
        return None

    target_touch_idx = touches[target_touch_n - 1]
    target_row = df.iloc[target_touch_idx]

    if not (target_row["low"] <= ob.zone_mid <= target_row["high"]):
        return None

    entry_price = ob.zone_mid

    # SL : sous le low OB (long) / au-dessus du high (short) + buffer
    sl_buffer = max(ob.zone_size * sl_buffer_pct, sl_min_points)
    if ob.kind == "bullish":
        stop_loss = ob.zone_low - sl_buffer
    else:
        stop_loss = ob.zone_high + sl_buffer

    risk = abs(entry_price - stop_loss)
    reward = abs(target_price - entry_price)
    if risk <= 0 or reward <= 0:
        return None

    rr = reward / risk
    if rr < min_rr:
        return None

    # Plafond : si liquidite trop loin, on rabote au max_rr
    final_tp = float(target_price)
    if rr > max_rr:
        if ob.kind == "bullish":
            final_tp = entry_price + risk * max_rr
        else:
            final_tp = entry_price - risk * max_rr
        rr = max_rr

    return EntrySignal(
        ob=ob,
        touch_count=target_touch_n,
        entry_index=target_touch_idx,
        entry_time=df.index[target_touch_idx],
        entry_price=float(entry_price),
        stop_loss=float(stop_loss),
        take_profit=final_tp,
        risk_reward=float(rr),
        target_source=target_source,
    )
