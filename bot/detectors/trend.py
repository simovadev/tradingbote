"""Detection du trend sur un timeframe donne.

Methode ICT pure : on regarde les swing highs/lows recents.
- HH + HL = uptrend
- LL + LH = downtrend
- Mixte = range

On regarde les N derniers swings et on classifie.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from bot.detectors.swings import Swing, detect_swings

TrendDirection = Literal["bullish", "bearish", "range"]


@dataclass
class TrendAnalysis:
    direction: TrendDirection
    confidence: float          # 0 a 1
    last_high: float | None
    last_low: float | None
    detail: str


def analyze_trend(df: pd.DataFrame, lookback_bars: int = 80, left: int = 3, right: int = 3) -> TrendAnalysis:
    """Analyse le trend sur les `lookback_bars` dernieres bougies.

    Methode combinee :
    1. Swings HH/HL vs LL/LH (structure)
    2. SMA20 vs SMA50 (tendance globale)
    3. Net move sur la fenetre (direction generale)

    Les 3 doivent etre alignes pour confidence 1.0.
    """
    if len(df) < lookback_bars:
        sub = df
    else:
        sub = df.iloc[-lookback_bars:]

    if len(sub) < 20:
        return TrendAnalysis("range", 0.0, None, None, "Pas assez de bougies")

    # === Signal 1 : swings HH/HL ===
    swings = detect_swings(sub, left=left, right=right)
    highs_arr = [s for s in swings if s.kind == "high"]
    lows_arr = [s for s in swings if s.kind == "low"]
    swing_signal = "range"
    if len(highs_arr) >= 2 and len(lows_arr) >= 2:
        last_h, prev_h = highs_arr[-1].price, highs_arr[-2].price
        last_l, prev_l = lows_arr[-1].price, lows_arr[-2].price
        if last_h > prev_h and last_l > prev_l:
            swing_signal = "bullish"
        elif last_h < prev_h and last_l < prev_l:
            swing_signal = "bearish"

    # === Signal 2 : SMA20 vs prix actuel ===
    sma_signal = "range"
    if len(sub) >= 20:
        sma20 = sub["close"].rolling(20).mean().iloc[-1]
        last_close = sub["close"].iloc[-1]
        diff_pct = (last_close - sma20) / sma20 * 100 if sma20 > 0 else 0
        if diff_pct > 0.1:
            sma_signal = "bullish"
        elif diff_pct < -0.1:
            sma_signal = "bearish"

    # === Signal 3 : net move sur la fenetre ===
    net_signal = "range"
    first_close = sub["close"].iloc[0]
    last_close = sub["close"].iloc[-1]
    net_pct = (last_close - first_close) / first_close * 100 if first_close > 0 else 0
    # Seuil dependant de la fenetre : > 0.3% net move pour confirmer
    if net_pct > 0.3:
        net_signal = "bullish"
    elif net_pct < -0.3:
        net_signal = "bearish"

    # === Combine : majorite des 3 signaux ===
    signals = [swing_signal, sma_signal, net_signal]
    bull_count = signals.count("bullish")
    bear_count = signals.count("bearish")

    last_high = highs_arr[-1].price if highs_arr else None
    last_low = lows_arr[-1].price if lows_arr else None

    if bull_count >= 2 and bear_count == 0:
        confidence = 1.0 if bull_count == 3 else 0.7
        return TrendAnalysis("bullish", confidence, last_high, last_low,
                             f"swing={swing_signal}, sma={sma_signal}, net={net_pct:+.2f}%")
    if bear_count >= 2 and bull_count == 0:
        confidence = 1.0 if bear_count == 3 else 0.7
        return TrendAnalysis("bearish", confidence, last_high, last_low,
                             f"swing={swing_signal}, sma={sma_signal}, net={net_pct:+.2f}%")

    return TrendAnalysis("range", 0.3, last_high, last_low,
                         f"signaux mixtes: swing={swing_signal}, sma={sma_signal}, net={net_pct:+.2f}%")


def trade_aligned_with_trend(trade_direction: TrendDirection, trend: TrendAnalysis,
                              accept_range: bool = True) -> bool:
    """Verifie si la direction du trade est alignee avec le trend."""
    if trend.direction == "range":
        return accept_range   # En range, on autorise (les retournements sont OK)
    return trade_direction == trend.direction
