"""Liquidites externes : swing high/low, BSL/SSL, equal H/L, IQH/IQL.

Bible §4 — Liquidites externes.

Definitions Vizion :
- **Swing High** : un high precede et suivi de highs plus bas (pivot).
- **Swing Low** : un low precede et suivi de lows plus hauts.
- **BSL** (Buy Side Liquidity) : zone au-dessus d'un swing high = stops des shorts.
- **SSL** (Sell Side Liquidity) : zone sous un swing low = stops des longs.
- **Equal High/Low** : 2 highs (ou 2 lows) quasi au meme niveau = magnet a liquidite.
- **Sweep** : prise de liquidite = la meche depasse un swing PUIS la bougie clos en arriere.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd


SwingKind = Literal["high", "low"]


@dataclass(frozen=True)
class Swing:
    """Un swing high/low identifie."""
    kind: SwingKind                 # "high" | "low"
    index: int                      # position dans le DataFrame (iloc)
    timestamp: pd.Timestamp
    price: float                    # high si kind="high", low si "low"
    # Force = nombre de bougies de chaque cote qui confirment le swing
    strength: int
    # True si liquidite encore presente (pas encore prise)
    is_live: bool = True


@dataclass(frozen=True)
class Sweep:
    """Une prise de liquidite identifiee (le swing a ete sweep)."""
    swing: Swing                          # le swing qui a ete pris
    sweep_index: int                      # iloc de la bougie qui sweep
    sweep_timestamp: pd.Timestamp
    direction: Literal["bullish", "bearish"]  # bullish = sweep d'un swing LOW (faux down)


def find_swings(df: pd.DataFrame, strength: int = 2) -> list[Swing]:
    """Detecte les swing highs et lows dans le DataFrame.

    Un swing high d'ordre N = high entoure de N bougies de chaque cote avec un high inferieur.
    Strict (>), pas (>=), pour eviter les plateaux ambigus.

    Args:
        df: DataFrame indexe en datetime, colonnes high/low.
        strength: nombre de bougies de chaque cote (defaut 2 = "fractale 5 bougies").

    Returns:
        Liste des swings tries par index ascendant.
    """
    if "high" not in df.columns or "low" not in df.columns:
        raise ValueError("DataFrame doit contenir 'high' et 'low'")

    highs = df["high"].values
    lows = df["low"].values
    n = len(df)
    swings: list[Swing] = []

    for i in range(strength, n - strength):
        # Swing high : high[i] > all highs des `strength` bougies de chaque cote
        left_highs = highs[i - strength:i]
        right_highs = highs[i + 1:i + 1 + strength]
        if highs[i] > left_highs.max() and highs[i] > right_highs.max():
            swings.append(Swing(
                kind="high",
                index=i,
                timestamp=df.index[i],
                price=float(highs[i]),
                strength=strength,
            ))

        # Swing low : symetrique
        left_lows = lows[i - strength:i]
        right_lows = lows[i + 1:i + 1 + strength]
        if lows[i] < left_lows.min() and lows[i] < right_lows.min():
            swings.append(Swing(
                kind="low",
                index=i,
                timestamp=df.index[i],
                price=float(lows[i]),
                strength=strength,
            ))

    return sorted(swings, key=lambda s: s.index)


def find_sweeps(
    df: pd.DataFrame,
    swings: list[Swing],
    min_depth_atr: float = 0.0,
    atr_lookback: int = 14,
) -> list[Sweep]:
    """Pour chaque swing, cherche si une bougie posterieure l'a sweep.

    VERSION OPTIMISEE NUMPY : O(N) par swing au lieu de O(N) avec branchements Python.
    Gain ~20-50x sur gros datasets (M1 5 ans).

    Sweep d'un swing high = bougie dont le HIGH depasse le swing high
                            mais qui CLOS en-dessous (rejet).
    Sweep d'un swing low  = bougie dont le LOW casse le swing low
                            mais qui CLOS au-dessus (rejet).

    Le sweep est unidirectionnel : on cherche la PREMIERE bougie qui sweep,
    apres l'index du swing.
    """
    if not swings:
        return []
    import numpy as np

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    n = len(df)
    timestamps = df.index
    sweeps: list[Sweep] = []

    # Pre-calcule ATR (high-low rolling mean sur atr_lookback bougies)
    # pour le filtre min_depth_atr.
    if min_depth_atr > 0:
        hl_range = highs - lows
        atr_arr = pd.Series(hl_range).rolling(atr_lookback).mean().shift(1).values
    else:
        atr_arr = None

    for sw in swings:
        start_idx = sw.index + 1
        if start_idx >= n:
            continue

        if sw.kind == "high":
            # 1ere bougie ou high > swing.price
            # np.argmax retourne 0 si AUCUN True (donc on check after)
            mask_breach = highs[start_idx:] > sw.price
            if not mask_breach.any():
                continue
            j_rel = int(mask_breach.argmax())
            j = start_idx + j_rel
            # Verifie rejet (close < swing.price)
            if closes[j] < sw.price:
                if min_depth_atr > 0 and atr_arr is not None:
                    atr = atr_arr[j]
                    if atr > 0 and not np.isnan(atr):
                        depth = highs[j] - sw.price
                        if depth < min_depth_atr * atr:
                            continue
                sweeps.append(Sweep(
                    swing=sw,
                    sweep_index=j,
                    sweep_timestamp=timestamps[j],
                    direction="bearish",
                ))
        else:
            mask_breach = lows[start_idx:] < sw.price
            if not mask_breach.any():
                continue
            j_rel = int(mask_breach.argmax())
            j = start_idx + j_rel
            if closes[j] > sw.price:
                if min_depth_atr > 0 and atr_arr is not None:
                    atr = atr_arr[j]
                    if atr > 0 and not np.isnan(atr):
                        depth = sw.price - lows[j]
                        if depth < min_depth_atr * atr:
                            continue
                sweeps.append(Sweep(
                    swing=sw,
                    sweep_index=j,
                    sweep_timestamp=timestamps[j],
                    direction="bullish",
                ))

    return sweeps


def find_equal_levels(
    swings: list[Swing],
    tolerance_pct: float = 0.05,
) -> list[tuple[Swing, Swing]]:
    """Detecte les paires de swings au MEME niveau (equal H/L, magnet a liquidite).

    Args:
        swings: liste des swings.
        tolerance_pct: ecart toleré en % du prix pour considerer "equal" (defaut 0.05%).

    Returns:
        Liste de paires (swing1, swing2) du meme kind, niveaux quasi egaux.
    """
    pairs: list[tuple[Swing, Swing]] = []
    by_kind: dict[SwingKind, list[Swing]] = {"high": [], "low": []}
    for sw in swings:
        by_kind[sw.kind].append(sw)

    for kind, lst in by_kind.items():
        for i, a in enumerate(lst):
            for b in lst[i + 1:]:
                avg = (a.price + b.price) / 2
                if avg == 0:
                    continue
                diff_pct = abs(a.price - b.price) / avg * 100
                if diff_pct <= tolerance_pct:
                    pairs.append((a, b))
    return pairs


def live_swings(swings: list[Swing], sweeps: list[Sweep]) -> list[Swing]:
    """Filtre : ne garde que les swings qui n'ont PAS encore ete sweep."""
    swept_ids = {(sw.swing.kind, sw.swing.index) for sw in sweeps}
    return [s for s in swings if (s.kind, s.index) not in swept_ids]


if __name__ == "__main__":
    # Test : charge XAUUSD M5, detecte swings + sweeps
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent))
    from bot_v2.data_loader import load

    df = load("XAUUSD", "M5")
    print(f"XAUUSD M5 : {len(df)} bougies\n")

    # On prend une fenetre raisonnable (1 semaine) pour visualiser
    df_week = df.last("7D")
    print(f"Derniers 7 jours : {len(df_week)} bougies")

    swings = find_swings(df_week, strength=2)
    print(f"Swings detectes : {len(swings)}")
    print(f"  - Highs : {sum(1 for s in swings if s.kind == 'high')}")
    print(f"  - Lows  : {sum(1 for s in swings if s.kind == 'low')}")

    sweeps = find_sweeps(df_week, swings)
    print(f"\nSweeps detectes : {len(sweeps)}")
    print(f"  - Bullish (low sweep) : {sum(1 for s in sweeps if s.direction == 'bullish')}")
    print(f"  - Bearish (high sweep): {sum(1 for s in sweeps if s.direction == 'bearish')}")

    live = live_swings(swings, sweeps)
    print(f"\nSwings encore LIVE (non sweep) : {len(live)}")

    equals = find_equal_levels(swings, tolerance_pct=0.05)
    print(f"Equal levels (tolerance 0.05%) : {len(equals)}")

    # Affiche les 5 derniers sweeps
    print("\n=== Derniers sweeps ===")
    for s in sweeps[-5:]:
        print(f"  {s.sweep_timestamp} | {s.direction:8s} | swing {s.swing.kind} @ {s.swing.price:.3f}")
