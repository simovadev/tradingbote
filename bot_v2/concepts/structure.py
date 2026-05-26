"""Structure de marche : BOS, MSS, SISD — bible §5.

Definitions Vizion (7q1cQyvvQKI, 3RAGDN6TfTA, P66QVQNegvo, kyk9Y3EYeE8) :

- **BOS (Break of Structure)** : cassure d'un swing dans le sens de la TENDANCE
  en cours. = CONTINUATION.

- **MSS (Market Structure Shift)** : cassure d'un swing CONTRE la tendance,
  avec une bougie qui CLOS de l'autre cote (corps de bougie, pas meche).
  = CHANGEMENT DE DIRECTION. Necessite IDEALEMENT un FVG dans le displacement
  pour etre tradable (modele 2022 ICT, P66QVQNegvo).

- **SISD (Sised / Change in State of Delivery)** : un OB qui acte un changement
  de PHASE de marche (retracement -> expansion, ou expansion -> reversal).
  Souvent plus precoce que le MSS (7q1cQyvvQKI).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from bot_v2.concepts.fvg import FVG, detect_fvg
from bot_v2.concepts.liquidity import Swing, find_swings


StructureKind = Literal["BOS", "MSS"]
Direction = Literal["bullish", "bearish"]


@dataclass(frozen=True)
class StructureBreak:
    """Une cassure de structure (BOS ou MSS)."""
    kind: StructureKind
    direction: Direction          # bullish = cassure vers le haut
    # Le swing qui a ete casse
    swing: Swing
    # La bougie qui casse (close > swing_high pour bullish)
    break_index: int
    break_ts: pd.Timestamp
    break_close: float
    # Pour le MSS : presence d'un FVG dans le displacement (modele 2022)
    has_displacement_fvg: bool = False


def detect_trend(swings: list[Swing], lookback: int = 4) -> Direction | None:
    """Estime la tendance actuelle en regardant les N derniers swings.

    - Tendance HAUSSIERE : derniers highs croissants ET derniers lows croissants
    - Tendance BAISSIERE : symetrique
    - Sinon : range (None).
    """
    if len(swings) < lookback:
        return None

    recent = swings[-lookback:]
    highs = [s for s in recent if s.kind == "high"]
    lows = [s for s in recent if s.kind == "low"]

    if len(highs) < 2 or len(lows) < 2:
        return None

    highs_rising = all(h2.price > h1.price for h1, h2 in zip(highs, highs[1:]))
    lows_rising = all(l2.price > l1.price for l1, l2 in zip(lows, lows[1:]))
    highs_falling = all(h2.price < h1.price for h1, h2 in zip(highs, highs[1:]))
    lows_falling = all(l2.price < l1.price for l1, l2 in zip(lows, lows[1:]))

    if highs_rising and lows_rising:
        return "bullish"
    if highs_falling and lows_falling:
        return "bearish"
    return None


def detect_structure_breaks(
    df: pd.DataFrame,
    swings: list[Swing] | None = None,
    swing_strength: int = 2,
    fvgs: list[FVG] | None = None,
) -> list[StructureBreak]:
    """Detecte BOS et MSS.

    Logique :
    1. Pour chaque swing, on cherche la 1ere bougie qui CLOS au-dela du swing.
       - Si swing HIGH et close > swing.price -> cassure haussiere.
       - Si swing LOW et close < swing.price  -> cassure baissiere.
    2. On regarde la TENDANCE au moment de la cassure :
       - Cassure dans le sens de la tendance = BOS.
       - Cassure contre la tendance = MSS.
    3. Pour MSS : on verifie aussi la presence d'un FVG dans les 3 bougies
       autour de la cassure (modele 2022).

    Args:
        fvgs: FVGs pre-calcules (evite un detect_fvg redondant si l'appelant
              les a deja). Si None, calcule en interne.
    """
    if swings is None:
        swings = find_swings(df, strength=swing_strength)

    closes = df["close"].values
    breaks: list[StructureBreak] = []
    if fvgs is None:
        fvgs = detect_fvg(df)

    import bisect as _bisect
    _swing_indices = [s.index for s in swings]  # deja trie (swings tries par index)

    import numpy as _np

    # OPTIM V5.5 (2026-05-21) : index FVG par center_index pour le check
    # "FVG dans displacement". Avant : `for f in fvgs` par MSS = O(N_mss * N_fvg)
    # = 16k * 22k = 350M iterations Python (~5s). Maintenant : lookup direct
    # par index dans 2 listes pre-bucketees (bull/bear). O(1) par MSS.
    _n = len(closes)
    _fvg_bull_at: list[bool] = [False] * (_n + 1)
    _fvg_bear_at: list[bool] = [False] * (_n + 1)
    for f in fvgs:
        ci = f.center_index
        if 0 <= ci <= _n:
            if f.direction == "bullish":
                _fvg_bull_at[ci] = True
            else:
                _fvg_bear_at[ci] = True

    for k, swing in enumerate(swings):
        target = swing.price
        # numpy argmax pour trouver la 1ere cassure (vectorise).
        _start = swing.index + 1
        if _start >= _n:
            continue
        _seg = closes[_start:]
        if swing.kind == "high":
            _hits = _seg > target
        else:
            _hits = _seg < target
        if not _hits.any():
            continue  # swing jamais casse
        j = _start + int(_np.argmax(_hits))  # 1ere bougie qui casse

        broke_up = swing.kind == "high"
        direction: Direction = "bullish" if broke_up else "bearish"

        # Tendance basee sur les swings AVANT la cassure (bisect = O(log N)).
        _cut = _bisect.bisect_left(_swing_indices, j)
        swings_before = swings[max(0, _cut - 20):_cut]
        trend = detect_trend(swings_before, lookback=4)

        if trend is None:
            kind: StructureKind = "MSS"
        elif trend == direction:
            kind = "BOS"
        else:
            kind = "MSS"

        # V17 FIX-5 : Pour MSS, check FVG dans displacement SUR LE PASSE UNIQUEMENT.
        # Avant : fenetre [j-3, j+3] -> leak +3 bougies futures (training vs live).
        # Maintenant : [j-3, j] (passe/present uniquement).
        has_fvg = False
        if kind == "MSS":
            window_start = max(0, j - 3)
            window_end = j + 1  # inclus j, EXCLUS j+1, j+2, j+3
            _at = _fvg_bull_at if direction == "bullish" else _fvg_bear_at
            for _ci in range(window_start, window_end):
                if _at[_ci]:
                    has_fvg = True
                    break

        breaks.append(StructureBreak(
            kind=kind,
            direction=direction,
            swing=swing,
            break_index=j,
            break_ts=df.index[j],
            break_close=float(closes[j]),
            has_displacement_fvg=has_fvg,
        ))

    return breaks


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent))
    from bot_v2.data_loader import load

    df = load("XAUUSD", "M5")
    mask = df.index >= (df.index.max() - pd.Timedelta(days=7))
    df_week = df[mask]
    print(f"XAUUSD M5 (7 derniers jours) : {len(df_week)} bougies\n")

    swings = find_swings(df_week, strength=2)
    trend_now = detect_trend(swings, lookback=4)
    print(f"Tendance actuelle (4 swings) : {trend_now}")

    breaks = detect_structure_breaks(df_week, swings=swings)
    print(f"\nCassures detectees : {len(breaks)}")
    print(f"  - BOS : {sum(1 for b in breaks if b.kind == 'BOS')}")
    print(f"  - MSS : {sum(1 for b in breaks if b.kind == 'MSS')}")
    mss_with_fvg = sum(1 for b in breaks if b.kind == 'MSS' and b.has_displacement_fvg)
    mss_total = sum(1 for b in breaks if b.kind == 'MSS')
    print(f"  - MSS avec FVG displacement (tradable modele 2022) : {mss_with_fvg}/{mss_total}")

    print(f"\n=== 5 dernieres cassures ===")
    for b in breaks[-5:]:
        fvg_tag = " [FVG-OK]" if b.has_displacement_fvg else ""
        print(f"  {b.break_ts} | {b.kind:3s} {b.direction:8s} | "
              f"break swing {b.swing.kind} @ {b.swing.price:.3f}{fvg_tag}")
