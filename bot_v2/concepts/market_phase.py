"""4 phases de marche — bible §11.1 (Pm29OIifOns, AE0K6W9uiSY, h0Dc_28mZhY).

Concept Vizion central :
- **Accumulation** (range, faible volume) -> PAS TRADABLE
- **Manipulation** (faux mouvement contre le bias) -> PAS TRADABLE
- **Distribution** (vraie expansion dans le sens du bias) -> ON TRADE ICI
- **Reversal** -> ON TRADE (retournement majeur)

Regle stricte : "On trade UNIQUEMENT l'Expansion et le Reversal."

Detection heuristique :
- Range = derniers N bougies oscillent dans une zone etroite (variance basse).
- Expansion = la bougie courante OU les 3 dernieres ont du displacement
  (volatilite > moyenne, corps direction nette).
- Manipulation = sweep recent + retour rapide dans le range.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd


Phase = Literal["accumulation", "manipulation", "expansion", "reversal", "undetermined"]


@dataclass(frozen=True)
class MarketPhaseAnalysis:
    """Analyse de la phase de marche au moment de l'OB."""
    phase: Phase
    is_tradable: bool                # True seulement si expansion ou reversal
    range_size: float                # range des N dernieres bougies
    avg_body_size: float             # corps moyen des N dernieres bougies
    last_displacement: float         # ratio (corps dernière bougie / range moyen)
    reason: str                      # explication textuelle


def analyze_phase(
    df: pd.DataFrame,
    at_index: int,
    lookback: int = 20,
) -> MarketPhaseAnalysis:
    """Analyse la phase de marche AU MOMENT de la bougie at_index.

    Returns:
        MarketPhaseAnalysis avec is_tradable=True seulement si expansion/reversal.
    """
    if at_index < lookback:
        return MarketPhaseAnalysis(
            phase="undetermined",
            is_tradable=True,  # par defaut on laisse passer si pas assez de data
            range_size=0.0,
            avg_body_size=0.0,
            last_displacement=0.0,
            reason="Pas assez de donnees (< lookback)",
        )

    window = df.iloc[at_index - lookback:at_index + 1]
    highs = window["high"].values
    lows = window["low"].values
    opens = window["open"].values
    closes = window["close"].values

    # Range total des N dernieres bougies
    range_total = float(highs.max() - lows.min())
    if range_total <= 0:
        return MarketPhaseAnalysis(
            phase="undetermined", is_tradable=False, range_size=0.0,
            avg_body_size=0.0, last_displacement=0.0,
            reason="Range nul",
        )

    # Taille moyenne de corps des bougies
    bodies = [abs(closes[i] - opens[i]) for i in range(len(window))]
    avg_body = float(sum(bodies) / len(bodies))

    # ATR simple (high-low moyen) pour normaliser
    atr = float(sum(highs[i] - lows[i] for i in range(len(window))) / len(window))
    if atr <= 0:
        atr = range_total / lookback

    # Corps de la bougie ACTUELLE (celle qui valide l'OB)
    last_body = bodies[-1]
    last_displacement = last_body / atr if atr > 0 else 0.0

    # === DETECTION DE RANGE / ACCUMULATION ===
    # Si le range des N dernieres bougies est SERRE (< 3x atr), et que les bougies
    # oscillent (corps moyen << range), c'est de l'accumulation.
    if range_total < 3.0 * atr and avg_body < atr * 0.6:
        return MarketPhaseAnalysis(
            phase="accumulation",
            is_tradable=False,
            range_size=range_total,
            avg_body_size=avg_body,
            last_displacement=last_displacement,
            reason=f"Range etroit (range={range_total:.2f} < 3*ATR={3*atr:.2f}) et corps faibles",
        )

    # === DETECTION D'EXPANSION ===
    # Si la bougie courante a un corps > 1.0x ATR ET la moyenne des 3 dernieres > 0.7x ATR,
    # on est en expansion.
    last_3_avg = sum(bodies[-3:]) / 3
    if last_displacement > 1.0 and last_3_avg > 0.7 * atr:
        return MarketPhaseAnalysis(
            phase="expansion",
            is_tradable=True,
            range_size=range_total,
            avg_body_size=avg_body,
            last_displacement=last_displacement,
            reason=f"Displacement franc (last={last_displacement:.2f}xATR, last_3_avg={last_3_avg/atr:.2f}xATR)",
        )

    # === DETECTION DE MANIPULATION ===
    # Bougie longue MAIS direction contraire aux N-1 precedentes = manipulation.
    # Si les N-1 bougies precedentes sont haussieres en moyenne mais la derniere est tres baissiere (ou inverse)
    prev_dir = sum(1 if closes[i] > opens[i] else -1 for i in range(len(window) - 1))
    last_dir = 1 if closes[-1] > opens[-1] else -1
    if last_displacement > 1.2 and prev_dir * last_dir < 0 and abs(prev_dir) > lookback * 0.4:
        return MarketPhaseAnalysis(
            phase="manipulation",
            is_tradable=False,
            range_size=range_total,
            avg_body_size=avg_body,
            last_displacement=last_displacement,
            reason=f"Manipulation (bougie contraire forte {last_displacement:.2f}xATR vs prev_dir={prev_dir})",
        )

    # === REVERSAL ===
    # Apres une longue tendance + bougie d'inversion forte
    if last_displacement > 1.5 and abs(prev_dir) > lookback * 0.5 and prev_dir * last_dir < 0:
        return MarketPhaseAnalysis(
            phase="reversal",
            is_tradable=True,
            range_size=range_total,
            avg_body_size=avg_body,
            last_displacement=last_displacement,
            reason=f"Reversal (forte inversion {last_displacement:.2f}xATR)",
        )

    # === DEFAUT : indetermine ===
    # Mouvement mou, ni expansion claire ni range serre.
    # Par defaut on dit NON TRADABLE pour eviter les trades douteux dans des zones de retracement.
    return MarketPhaseAnalysis(
        phase="undetermined",
        is_tradable=False,
        range_size=range_total,
        avg_body_size=avg_body,
        last_displacement=last_displacement,
        reason=f"Phase indeterminee (last_disp={last_displacement:.2f}xATR, avg_body/atr={avg_body/atr:.2f})",
    )


def has_displacement_at_validation(
    df: pd.DataFrame,
    validation_index: int,
    direction: Literal["bullish", "bearish"],
    lookback_atr: int = 14,
    min_displacement_ratio: float = 1.2,
) -> tuple[bool, float]:
    """Verifie que la bougie de validation OB a un VRAI displacement (bible §5).

    Vizion (kyk9Y3EYeE8, P66QVQNegvo) :
    > "Sans displacement, il n'y a pas de volonte de changement de structure
    >  quantifiable sur les bougies, donc PAS DE MSS."

    Applique aussi a l'OB : la bougie qui valide doit avoir un corps clair > 1.2x ATR
    dans le sens du trade.

    Returns:
        (has_displacement, ratio_corps_vs_atr)
    """
    if validation_index < lookback_atr or validation_index >= len(df):
        return False, 0.0

    val_bar = df.iloc[validation_index]
    body = abs(val_bar["close"] - val_bar["open"])

    # ATR sur les `lookback_atr` bougies precedentes
    prev = df.iloc[validation_index - lookback_atr:validation_index]
    atr = float((prev["high"] - prev["low"]).mean())
    if atr <= 0:
        return False, 0.0

    ratio = body / atr

    # Verifie aussi que la bougie est du bon sens
    correct_direction = (
        (direction == "bullish" and val_bar["close"] > val_bar["open"])
        or (direction == "bearish" and val_bar["close"] < val_bar["open"])
    )

    return (ratio >= min_displacement_ratio and correct_direction), ratio


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent))
    from bot_v2.data_loader import load

    df = load("XAUUSD", "M5")
    mask = df.index >= (df.index.max() - pd.Timedelta(days=3))
    df = df[mask]
    print(f"XAUUSD M5 (3j) : {len(df)} bougies")

    # Test sur 10 derniers points
    print("\n=== Phases des 10 dernieres bougies ===")
    for i in range(max(0, len(df) - 10), len(df)):
        a = analyze_phase(df, i, lookback=20)
        print(f"  {df.index[i]} | phase={a.phase:14s} | tradable={str(a.is_tradable):5s} | disp={a.last_displacement:.2f}x | {a.reason[:60]}")

    # Stats sur l'ensemble du dataset
    from collections import Counter
    cnt = Counter()
    for i in range(20, len(df)):
        a = analyze_phase(df, i, lookback=20)
        cnt[a.phase] += 1
    print("\n=== Repartition phases sur 3j ===")
    for phase, c in cnt.most_common():
        pct = c / sum(cnt.values()) * 100
        print(f"  {phase:14s} : {c:4d} ({pct:.0f}%)")
