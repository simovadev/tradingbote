"""Power of Three (PO3) / AMD — bible §11.

Definitions Vizion :
- **PO3 (Power of Three)** : lecture du couple corps/meche d'une bougie HTF.
  - OLHC = bougie bullish (Open puis Low (manipulation) puis High (distribution) puis Close)
  - OHLC = bougie bearish (Open puis High (manipulation) puis Low (distribution) puis Close)

- **AMD** : Accumulation - Manipulation - Distribution.
  - **Accumulation** : range autour de l'open.
  - **Manipulation** : prise de liquidite (meche dans le sens inverse au sens final).
  - **Distribution** : le vrai mouvement (le corps).

Decision user 2026-05-15 : utilisable sur D1, H4, H1 ET M5 comme signal d'entree.

Detection :
On regarde une bougie HTF "courante" (en cours de formation) et on identifie
sa phase via les bougies LTF (sous-bougies) qui la composent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd


PO3Phase = Literal["accumulation", "manipulation", "distribution", "complete"]
PO3Sense = Literal["bullish", "bearish", "undetermined"]


@dataclass(frozen=True)
class PO3:
    """Analyse PO3 d'une bougie HTF."""
    htf_open: float
    htf_high: float
    htf_low: float
    htf_close: float
    sense: PO3Sense           # bullish (OLHC) | bearish (OHLC) | undetermined
    phase: PO3Phase           # accumulation | manipulation | distribution | complete
    # Position relative
    close_to_high: float      # (close - low) / (high - low), 0..1
    body_pct: float           # |close - open| / (high - low)
    upper_wick_pct: float     # (high - max(open,close)) / (high-low)
    lower_wick_pct: float     # (min(open,close) - low) / (high-low)


def analyze_po3(htf_bar: pd.Series) -> PO3:
    """Analyse une bougie HTF pour determiner son PO3.

    Logique :
    - Si bougie haussiere (close > open) :
        - Si meche basse importante (>30% range) = OLHC = bullish complete
        - Sinon : continuation ou en construction
    - Sinon : OHLC bearish
    - Phase :
        - body_pct < 20% & wicks faibles : accumulation
        - une wick > 40% sans body marque : manipulation
        - body > 50% : distribution
        - sinon : complete
    """
    o = float(htf_bar["open"])
    h = float(htf_bar["high"])
    l = float(htf_bar["low"])
    c = float(htf_bar["close"])
    rng = h - l

    if rng == 0:
        return PO3(o, h, l, c, "undetermined", "accumulation", 0.5, 0.0, 0.0, 0.0)

    body = abs(c - o)
    body_pct = body / rng
    upper_wick = (h - max(o, c)) / rng
    lower_wick = (min(o, c) - l) / rng
    close_to_high = (c - l) / rng

    # Sens
    if c > o:
        # Bougie haussiere : OLHC potential (manipulation = wick basse)
        sense: PO3Sense = "bullish" if lower_wick > 0.25 else "undetermined"
    elif c < o:
        sense = "bearish" if upper_wick > 0.25 else "undetermined"
    else:
        sense = "undetermined"

    # Phase
    if body_pct < 0.2 and upper_wick < 0.3 and lower_wick < 0.3:
        phase: PO3Phase = "accumulation"
    elif body_pct < 0.4 and (upper_wick > 0.4 or lower_wick > 0.4):
        phase = "manipulation"
    elif body_pct >= 0.5:
        phase = "distribution"
    else:
        phase = "complete"

    return PO3(o, h, l, c, sense, phase, close_to_high, body_pct, upper_wick, lower_wick)


def analyze_amd_sequence(df: pd.DataFrame, n: int = 5) -> str:
    """Regarde les N dernieres bougies et tente d'identifier un cycle AMD en cours.

    Retourne une description textuelle de la phase courante.
    """
    if len(df) < n:
        return "donnees insuffisantes"

    recent = df.iloc[-n:]
    # Range total
    h = float(recent["high"].max())
    l = float(recent["low"].min())
    rng = h - l
    if rng == 0:
        return "range degenere"

    # Position de chaque bougie : tete (high) basse (low) corps
    # On cherche : 1ere phase compresse, puis pic dans un sens, puis reverse
    closes = recent["close"].values
    opens = recent["open"].values
    highs = recent["high"].values
    lows = recent["low"].values

    # Phase 1 : accumulation (premieres bougies tassees)
    first = recent.iloc[:n // 2]
    first_rng = float(first["high"].max() - first["low"].min())

    # Phase 2 : derniere bougie
    last = recent.iloc[-1]

    # Heuristique simple : si first_rng < rng/2 et bougie finale = displacement -> distribution
    if first_rng < rng * 0.5 and abs(last["close"] - last["open"]) > rng * 0.3:
        if last["close"] > last["open"]:
            return "distribution_bullish"
        return "distribution_bearish"

    # Sinon : phase intermediaire
    if first_rng < rng * 0.4:
        return "manipulation_in_progress"
    return "accumulation"


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent))
    from bot_v2.data_loader import load
    from bot_v2.concepts.daily_bias import build_d1_from_h1

    # PO3 sur la derniere bougie D1
    df_h1 = load("XAUUSD", "H1")
    df_d1 = build_d1_from_h1(df_h1)
    print(f"XAUUSD D1 : {len(df_d1)} bougies daily\n")

    # PO3 de la derniere bougie daily
    last_d1 = df_d1.iloc[-1]
    p = analyze_po3(last_d1)
    print(f"=== PO3 derniere bougie D1 ({df_d1.index[-1].date()}) ===")
    print(f"  OHLC : O={p.htf_open:.3f} H={p.htf_high:.3f} L={p.htf_low:.3f} C={p.htf_close:.3f}")
    print(f"  Sense : {p.sense}")
    print(f"  Phase : {p.phase}")
    print(f"  Body  : {p.body_pct*100:.1f}%")
    print(f"  Upper wick : {p.upper_wick_pct*100:.1f}%")
    print(f"  Lower wick : {p.lower_wick_pct*100:.1f}%")
    print(f"  Close to high : {p.close_to_high*100:.1f}%")

    # PO3 sur les 5 dernieres bougies M5
    df_m5 = load("XAUUSD", "M5")
    print("\n=== PO3 des 5 dernieres bougies M5 ===")
    for i in range(-5, 0):
        bar = df_m5.iloc[i]
        p = analyze_po3(bar)
        print(f"  {df_m5.index[i]} | sense={p.sense:11s} phase={p.phase:14s} body={p.body_pct*100:.0f}%")

    # AMD sequence M5
    amd = analyze_amd_sequence(df_m5.iloc[-30:], n=5)
    print(f"\nAMD sequence 5 dernieres bougies M5 : {amd}")
