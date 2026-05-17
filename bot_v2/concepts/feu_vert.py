"""Feu Vert H1 - Vizion bible V2 §2 (video 03 VXnCchk8NTM).

Concept Vizion : quand le daily bias est NEUTRAL, on ne reste pas bloque.
Le "feu vert" H1 prend le relais et autorise la chasse de setup.

Conditions strictes :
1. Sweep PDH (pour direction bearish) ou PDL (pour direction bullish) sur H1 recent.
2. OB H1 valide dans la direction OPPOSEE au sweep (= retournement).
3. Optionnel boost : SMT divergence + OB H1 forme en killzone.

Application bot intraday (decision user 2026-05-15) : assouplir le filtre
daily_bias=neutral en autorisant la chasse via feu vert H1.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from bot_v2.concepts.killzones import killzone_at
from bot_v2.concepts.order_block import OrderBlock, detect_order_blocks


@dataclass(frozen=True)
class FeuVert:
    """Resultat de la detection du feu vert H1."""
    is_green: bool                       # True si feu vert pour la direction
    direction: Literal["bullish", "bearish"]
    parent_ob_h1: OrderBlock | None      # OB H1 qui constitue le feu vert
    swept_level: Literal["PDH", "PDL"] | None
    in_killzone: bool                    # OB H1 forme en KZ ? (boost)
    reason: str


def _get_previous_day_high_low(
    df_d1: pd.DataFrame,
    target_ts: pd.Timestamp,
) -> tuple[float, float] | None:
    """Retourne (PDH, PDL) = high/low de la bougie daily PRECEDENTE."""
    prev = df_d1[df_d1.index < target_ts.normalize()]
    if len(prev) < 1:
        return None
    last = prev.iloc[-1]
    return (float(last["high"]), float(last["low"]))


def check_feu_vert_h1(
    df_h1: pd.DataFrame,
    df_d1: pd.DataFrame,
    target_ts: pd.Timestamp,
    direction: Literal["bullish", "bearish"],
    lookback_h1_bars: int = 24,
) -> FeuVert:
    """Verifie si le feu vert H1 est active pour `direction` au moment `target_ts`.

    Args:
        df_h1: bougies H1.
        df_d1: bougies D1 (pour PDH/PDL).
        target_ts: timestamp de l'OB LTF qu'on veut valider.
        direction: direction du trade envisage.
        lookback_h1_bars: combien de bougies H1 en arriere chercher l'evenement.

    Returns:
        FeuVert avec is_green=True si feu vert detecte.
    """
    # 1. Recupere PDH/PDL
    levels = _get_previous_day_high_low(df_d1, target_ts)
    if levels is None:
        return FeuVert(
            is_green=False, direction=direction, parent_ob_h1=None,
            swept_level=None, in_killzone=False,
            reason="PDH/PDL non calculable",
        )
    pdh, pdl = levels

    # 2. Fenetre H1 a verifier : `lookback_h1_bars` bougies avant target_ts
    mask = (df_h1.index <= target_ts) & (
        df_h1.index >= target_ts - pd.Timedelta(hours=lookback_h1_bars)
    )
    df_h1_window = df_h1[mask]
    if len(df_h1_window) < 3:
        return FeuVert(
            is_green=False, direction=direction, parent_ob_h1=None,
            swept_level=None, in_killzone=False,
            reason="Pas assez de bougies H1 dans la fenetre",
        )

    # 3. Selon direction, chercher sweep + OB H1 inverse
    # - bullish trade : on cherche PDL pris (sweep low) puis OB H1 BULLISH (retournement)
    # - bearish trade : PDH pris puis OB H1 BEARISH
    swept: Literal["PDH", "PDL"] | None = None
    if direction == "bullish":
        # Cherche une bougie H1 qui a casse PDL en meche (low < pdl)
        # et clos AU-DESSUS (rejet)
        sweep_mask = (df_h1_window["low"] < pdl) & (df_h1_window["close"] > pdl)
        if sweep_mask.any():
            swept = "PDL"
    else:
        sweep_mask = (df_h1_window["high"] > pdh) & (df_h1_window["close"] < pdh)
        if sweep_mask.any():
            swept = "PDH"

    if swept is None:
        return FeuVert(
            is_green=False, direction=direction, parent_ob_h1=None,
            swept_level=None, in_killzone=False,
            reason=f"Pas de sweep {'PDL' if direction=='bullish' else 'PDH'} sur H1",
        )

    # 4. Cherche un OB H1 dans la direction `direction`, dans la fenetre
    obs_h1 = detect_order_blocks(df_h1_window)
    matching = [
        ob for ob in obs_h1
        if ob.direction == direction and ob.validation_ts <= target_ts
    ]
    if not matching:
        return FeuVert(
            is_green=False, direction=direction, parent_ob_h1=None,
            swept_level=swept, in_killzone=False,
            reason=f"Sweep {swept} OK mais pas d'OB H1 {direction} pour retournement",
        )

    # Prend le plus recent
    parent_ob = max(matching, key=lambda ob: ob.validation_ts)
    in_kz = killzone_at(parent_ob.validation_ts) is not None

    return FeuVert(
        is_green=True,
        direction=direction,
        parent_ob_h1=parent_ob,
        swept_level=swept,
        in_killzone=in_kz,
        reason=f"Feu vert {direction} : sweep {swept} + OB H1 valide{' en KZ' if in_kz else ''}",
    )


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent))
    from bot_v2.data_loader import load
    from bot_v2.concepts.daily_bias import build_d1_from_h1

    df_h1 = load("NAS100", "H1")
    df_d1 = build_d1_from_h1(df_h1)

    # Test : feu vert pour la derniere bougie H1
    last_ts = df_h1.index[-1]
    for d in ["bullish", "bearish"]:
        fv = check_feu_vert_h1(df_h1, df_d1, last_ts, d)
        print(f"\n=== Feu Vert {d} @ {last_ts} ===")
        print(f"  is_green : {fv.is_green}")
        print(f"  swept    : {fv.swept_level}")
        print(f"  in_KZ    : {fv.in_killzone}")
        print(f"  reason   : {fv.reason}")
