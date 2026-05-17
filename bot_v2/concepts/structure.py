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
    """
    if swings is None:
        swings = find_swings(df, strength=swing_strength)

    closes = df["close"].values
    breaks: list[StructureBreak] = []
    fvgs = detect_fvg(df)

    for k, swing in enumerate(swings):
        target = swing.price
        # Cherche la 1ere bougie qui CLOS au-dela
        for j in range(swing.index + 1, len(df)):
            broke_up = swing.kind == "high" and closes[j] > target
            broke_down = swing.kind == "low" and closes[j] < target
            if not (broke_up or broke_down):
                continue

            direction: Direction = "bullish" if broke_up else "bearish"

            # Tendance basee sur les swings AVANT la cassure
            swings_before = [s for s in swings if s.index < j]
            trend = detect_trend(swings_before, lookback=4)

            if trend is None:
                # Pas de tendance claire : on considere par defaut comme MSS
                kind: StructureKind = "MSS"
            elif trend == direction:
                kind = "BOS"
            else:
                kind = "MSS"

            # Pour MSS : check FVG dans displacement (3 bougies avant -> 3 apres la cassure)
            has_fvg = False
            if kind == "MSS":
                window_start = max(0, j - 3)
                window_end = min(len(df), j + 4)
                for f in fvgs:
                    if window_start <= f.center_index < window_end and f.direction == direction:
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
            break  # un swing = une seule cassure

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
