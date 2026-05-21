"""Breaker Blocks — bible §16 (vidéo 9yChQ3V7u_o complete).

Definition Vizion stricte :
- Un **Breaker** est un ancien OB INVERSE en cloture (corps de bougie, pas meche).
- Breaker BULLISH = ancien OB BEARISH traverse a la hausse en cloture.
- Breaker BEARISH = ancien OB BULLISH traverse a la baisse en cloture.
- Devient une zone de support (bullish) / resistance (bearish) au retest.

3 conditions pour un Breaker HIGH PROBABILITY (9yChQ3V7u_o) :
1. **Qualite OB d'origine** : OB qui avait pris une liquidite (high probability).
2. **Association des PDR** : presence de IFVG, busy, OB sur le chemin de validation
   du breaker.
3. **Displacement** : cloture franche au-dela de l'OB origine (mouvement puissant).

2 types :
- **Breaker de CONTINUATION** : dans le sens de la tendance, objectif HTF pas
  encore atteint.
- **Breaker REVERSAL** : change la tendance, apparait apres prise d'un swing HTF
  pertinent.

SL = MECHE du breaker (decision user 2026-05-15).
Entry = corps de bougie chez Vizion (mais on prend zone breaker comme prix limite).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from bot_v2.concepts.order_block import OrderBlock, detect_order_blocks


Direction = Literal["bullish", "bearish"]
BreakerType = Literal["continuation", "reversal"]


@dataclass(frozen=True)
class Breaker:
    """Un ancien OB devenu Breaker apres inversion en cloture."""
    direction: Direction              # bullish = ancien OB bearish inverse
    # L'OB d'origine
    origin_ob: OrderBlock
    # La bougie qui inverse l'OB (clos au-dela du high pour bullish)
    inverse_index: int
    inverse_ts: pd.Timestamp
    inverse_close: float
    # Niveau du breaker = bornes de l'OB d'origine
    top: float
    bottom: float
    # Score qualite (3 conditions HP)
    has_quality_ob_origin: bool       # OB d'origine avait pris une liquidite (toujours True ici)
    has_displacement: bool            # displacement clair au moment de l'inversion
    associated_pdr_count: int = 0     # nombre de PDR (IFVG/OB/FVG) sur le chemin


def _has_displacement(df: pd.DataFrame, inv_idx: int, direction: Direction) -> bool:
    """Verifie un displacement franc au moment de l'inversion.

    Definition simple : la bougie d'inversion + sa suivante representent un
    mouvement > 1.5x l'ATR(14) des 14 bougies precedentes.
    """
    if inv_idx < 14 or inv_idx >= len(df) - 1:
        return False

    # ATR simple = mean(high-low) sur 14 bougies precedentes
    prev = df.iloc[inv_idx - 14:inv_idx]
    atr = float((prev["high"] - prev["low"]).mean())
    if atr == 0:
        return False

    # Mouvement = high-low de la bougie d'inversion
    inv_bar = df.iloc[inv_idx]
    move = float(inv_bar["high"] - inv_bar["low"])
    return move > 1.5 * atr


def _has_displacement_fast(highs, lows, inv_idx: int, n: int) -> bool:
    """Version vectorisee de _has_displacement : opere sur arrays numpy.

    Pour rester BIT-IDENTIQUE a _has_displacement, on garde le meme ordre de
    sommation (pandas Series.mean()) au lieu de numpy.mean() qui peut differer
    d'un ULP et basculer un move > 1.5*atr a la limite.
    Le param `direction` original n'etait pas utilise dans le corps, on l'omet.
    """
    if inv_idx < 14 or inv_idx >= n - 1:
        return False
    # Reproduction exacte de l'original : pd.Series.mean() sur la difference
    # high-low de 14 bougies.
    hl = pd.Series(highs[inv_idx - 14:inv_idx] - lows[inv_idx - 14:inv_idx])
    atr = float(hl.mean())
    if atr == 0:
        return False
    move = float(highs[inv_idx] - lows[inv_idx])
    return move > 1.5 * atr


def detect_breakers(
    df: pd.DataFrame,
    obs: list[OrderBlock] | None = None,
) -> list[Breaker]:
    """Detecte les Breakers : OB qui ont ete inverses en cloture.

    VERSION OPTIMISEE NUMPY : pour chaque OB, on cherche la 1ere bougie
    qui clos de l'autre cote via np.argmax au lieu d'une boucle Python.

    Pour chaque OB :
    - Cherche apres validation si une bougie clos de l'autre cote.
    - OB bullish + close < ob_low => devient breaker BEARISH.
    - OB bearish + close > ob_high => devient breaker BULLISH.
    """
    if obs is None:
        obs = detect_order_blocks(df)
    if not obs:
        return []

    import numpy as np
    from bot_v2.concepts._fast import first_breach

    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    timestamps = df.index
    n = len(df)
    breakers: list[Breaker] = []

    # OPTIM V5.6 (2026-05-21) : suffixe min/max sur CLOSES (pas highs/lows !)
    # pour rejet O(1) des OB jamais inverses. Avant : closes[start_idx:] < ob_low
    # materialisait toute la queue par OB = O(N^2). Maintenant : O(1) reject +
    # first_breach galloping.
    suffix_min_close = np.minimum.accumulate(closes[::-1])[::-1]
    suffix_max_close = np.maximum.accumulate(closes[::-1])[::-1]

    for ob in obs:
        start_idx = ob.validation_index + 1
        if start_idx >= n:
            continue

        if ob.direction == "bullish":
            # Cherche close < ob_low (strict)
            if suffix_min_close[start_idx] >= ob.ob_low:
                continue
            j = first_breach(closes, start_idx, ob.ob_low, np.less)
        else:
            if suffix_max_close[start_idx] <= ob.ob_high:
                continue
            j = first_breach(closes, start_idx, ob.ob_high, np.greater)

        if j < 0:
            continue

        if ob.direction == "bullish":
            disp = _has_displacement_fast(highs, lows, j, n)
            breaker_dir = "bearish"
        else:
            disp = _has_displacement_fast(highs, lows, j, n)
            breaker_dir = "bullish"

        breakers.append(Breaker(
            direction=breaker_dir,
            origin_ob=ob,
            inverse_index=j,
            inverse_ts=timestamps[j],
            inverse_close=float(closes[j]),
            top=ob.ob_high,
            bottom=ob.ob_low,
            has_quality_ob_origin=True,
            has_displacement=disp,
        ))

    return breakers


def breaker_stop_loss(br: Breaker) -> float:
    """SL en MECHE (decision user) : sous le bottom pour bullish, au-dessus du top pour bearish."""
    if br.direction == "bullish":
        return br.bottom
    return br.top


def breaker_entry(br: Breaker) -> float:
    """Entree au niveau breaker : retest du top (bullish) ou bottom (bearish)."""
    if br.direction == "bullish":
        return br.top
    return br.bottom


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent))
    from bot_v2.data_loader import load

    df = load("XAUUSD", "M5")
    mask = df.index >= (df.index.max() - pd.Timedelta(days=7))
    df_week = df[mask]
    print(f"XAUUSD M5 (7 derniers jours) : {len(df_week)} bougies\n")

    obs = detect_order_blocks(df_week)
    print(f"OB detectes : {len(obs)}")

    brks = detect_breakers(df_week, obs=obs)
    print(f"Breakers detectes : {len(brks)}")
    print(f"  - Bullish : {sum(1 for b in brks if b.direction == 'bullish')}")
    print(f"  - Bearish : {sum(1 for b in brks if b.direction == 'bearish')}")
    print(f"  - Avec displacement : {sum(1 for b in brks if b.has_displacement)}")

    print(f"\n=== 5 derniers Breakers ===")
    for br in brks[-5:]:
        sl = breaker_stop_loss(br)
        entry = breaker_entry(br)
        disp = " [disp]" if br.has_displacement else ""
        print(
            f"  {br.inverse_ts} | breaker {br.direction:8s} | "
            f"entry={entry:.3f} sl={sl:.3f}{disp}"
        )
