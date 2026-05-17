"""Daily Bias — bible §9 (y2Fwp4T9sRM, M4F4YYDVmMI, IR0MNTY2-rs).

Methode Vizion :
Sur la bougie daily de la VEILLE, on regarde :
1. PDH / PDL pris ?
   - PDH pris + cloture haussiere => bias BULLISH (cible nouveau PDH)
   - PDL pris + cloture baissiere => bias BEARISH (cible nouveau PDL)
   - Les deux pris  => suivre le sens de cloture
   - Aucun         => NEUTRE (on attend / on ne trade pas)
2. Force de cloture : range body / range total
   - close pres du high (>= 70%) = cloture forte haussiere
   - close pres du low  (<= 30%) = cloture forte baissiere
   - sinon doji
3. OB Daily / FVG Daily / Breaker Daily : confluences ajoutees
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd


Bias = Literal["bullish", "bearish", "neutral"]


@dataclass(frozen=True)
class DailyBias:
    """Bias daily pour une date donnee, base sur la bougie daily de la veille."""
    target_date: pd.Timestamp     # date du jour pour lequel on calcule le bias
    bias: Bias                    # bullish | bearish | neutral
    # Donnees de la bougie daily de la veille
    prev_open: float
    prev_high: float
    prev_low: float
    prev_close: float
    prev_pdh_taken: bool          # le high de J-2 a-t-il ete pris ?
    prev_pdl_taken: bool          # le low de J-2 a-t-il ete pris ?
    close_position: float         # position de la close dans le range (0=low, 1=high)
    close_strength: Literal["strong_bull", "strong_bear", "doji"]
    # Cibles
    target_above: float           # prochain PDH = high de J-1
    target_below: float           # prochain PDL = low de J-1


def _close_strength(o: float, h: float, l: float, c: float) -> tuple[float, str]:
    """Position relative de la close (0=low, 1=high) + classification."""
    rng = h - l
    if rng == 0:
        return 0.5, "doji"
    pos = (c - l) / rng
    # Iter 36 : seuil plus strict 0.75/0.25 (avant : 0.7/0.3)
    if pos >= 0.75:
        return pos, "strong_bull"
    if pos <= 0.25:
        return pos, "strong_bear"
    return pos, "doji"


def compute_daily_bias(df_d1: pd.DataFrame, target_date: pd.Timestamp) -> DailyBias | None:
    """Calcule le bias daily pour `target_date` a partir de la bougie daily de la veille.

    Args:
        df_d1: bougies daily (index datetime UTC).
        target_date: date du jour pour lequel on cherche le bias.

    Returns:
        DailyBias ou None si pas assez de donnees.
    """
    # On veut la bougie daily JUSTE AVANT target_date
    prev = df_d1[df_d1.index < target_date]
    if len(prev) < 2:
        return None

    yesterday = prev.iloc[-1]
    day_before = prev.iloc[-2]

    pdh_taken = bool(yesterday["high"] > day_before["high"])
    pdl_taken = bool(yesterday["low"] < day_before["low"])
    pos, strength = _close_strength(
        yesterday["open"], yesterday["high"],
        yesterday["low"], yesterday["close"],
    )

    # Decision du bias
    bias: Bias
    if pdh_taken and not pdl_taken:
        bias = "bullish" if strength == "strong_bull" else "neutral"
    elif pdl_taken and not pdh_taken:
        bias = "bearish" if strength == "strong_bear" else "neutral"
    elif pdh_taken and pdl_taken:
        # Outside day : suivre le sens de cloture
        if strength == "strong_bull":
            bias = "bullish"
        elif strength == "strong_bear":
            bias = "bearish"
        else:
            bias = "neutral"
    else:
        # Aucun des deux pris : inside day -> neutre
        bias = "neutral"

    return DailyBias(
        target_date=target_date,
        bias=bias,
        prev_open=float(yesterday["open"]),
        prev_high=float(yesterday["high"]),
        prev_low=float(yesterday["low"]),
        prev_close=float(yesterday["close"]),
        prev_pdh_taken=pdh_taken,
        prev_pdl_taken=pdl_taken,
        close_position=float(pos),
        close_strength=strength,
        target_above=float(yesterday["high"]),
        target_below=float(yesterday["low"]),
    )


def build_d1_from_h1(df_h1: pd.DataFrame) -> pd.DataFrame:
    """Construit des bougies D1 a partir de bougies H1 (resample).

    Utile si on n'a pas de cache D1 mais qu'on a H1.
    """
    out = df_h1.resample("1D").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()
    return out


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent))
    from bot_v2.data_loader import load

    # Pas de D1 en cache => on resample depuis H1
    df_h1 = load("XAUUSD", "H1")
    df_d1 = build_d1_from_h1(df_h1)
    print(f"XAUUSD D1 (resample H1) : {len(df_d1)} bougies daily")
    print(df_d1.tail(5).to_string())

    # Bias pour aujourd'hui
    target = df_d1.index.max() + pd.Timedelta(days=1)
    bias = compute_daily_bias(df_d1, target)
    if bias:
        print(f"\n=== Bias pour {target.date()} ===")
        print(f"  Bias : {bias.bias.upper()}")
        print(f"  Veille : O={bias.prev_open:.3f} H={bias.prev_high:.3f} "
              f"L={bias.prev_low:.3f} C={bias.prev_close:.3f}")
        print(f"  PDH pris (vs J-2) : {bias.prev_pdh_taken}")
        print(f"  PDL pris (vs J-2) : {bias.prev_pdl_taken}")
        print(f"  Close strength : {bias.close_strength} ({bias.close_position*100:.0f}% du range)")
        print(f"  Cibles : above={bias.target_above:.3f} below={bias.target_below:.3f}")

    # Bias sur les 5 derniers jours pour verif
    print("\n=== Bias 5 derniers jours ===")
    for i in range(5, 0, -1):
        d = df_d1.index[-i]
        b = compute_daily_bias(df_d1, d)
        if b:
            print(f"  {d.date()} | bias={b.bias:8s} | close_strength={b.close_strength}")
