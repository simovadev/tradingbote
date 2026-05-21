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
    s = strength

    # OPTIM V5.6 (2026-05-21) : vectorisation via sliding_window_view.
    # Avant : boucle Python 88k iterations x slices .max()/.min() pandas (~0.8s).
    # Maintenant : 4 passes numpy vectorisees (~0.05s).
    if n < 2 * s + 1:
        return []

    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view

    win_h = sliding_window_view(highs, 2 * s + 1)  # shape (n-2s, 2s+1)
    win_l = sliding_window_view(lows, 2 * s + 1)
    center_h = win_h[:, s]
    center_l = win_l[:, s]
    # Strict (>) pour high, strict (<) pour low — identique a l'original.
    left_max_h = win_h[:, :s].max(axis=1)
    right_max_h = win_h[:, s + 1:].max(axis=1)
    left_min_l = win_l[:, :s].min(axis=1)
    right_min_l = win_l[:, s + 1:].min(axis=1)
    is_high = (center_h > left_max_h) & (center_h > right_max_h)
    is_low = (center_l < left_min_l) & (center_l < right_min_l)
    idx_high = np.flatnonzero(is_high) + s  # decalage fenetre -> index global
    idx_low = np.flatnonzero(is_low) + s

    # ORDRE EXACT a reproduire (original) : a chaque i, on append high PUIS
    # low, puis sorted(key=index) stable. Un meme i peut etre les deux (rare).
    # On tag rank=0 pour high, rank=1 pour low -> sort par (index, rank) reproduit.
    timestamps = df.index
    tagged = [(int(i), 0, "high", float(highs[i])) for i in idx_high]
    tagged += [(int(i), 1, "low", float(lows[i])) for i in idx_low]
    tagged.sort(key=lambda t: (t[0], t[1]))
    return [
        Swing(kind=k, index=i, timestamp=timestamps[i], price=p, strength=s)
        for (i, _r, k, p) in tagged
    ]


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
    from bot_v2.concepts._fast import first_breach

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

    # OPTIM V5.6 (2026-05-21) : suffixe max/min pour rejet O(1) des swings
    # jamais sweepes. Avant : `arr[start:] > price` par swing materialisait
    # toute la queue (~88k floats) -> O(N^2) sur 23k swings. Maintenant :
    # check suffixe O(1) + first_breach galloping pour les sweepes (sweep
    # arrive tot apres swing -> 1-2 fenetres suffisent).
    suffix_max_high = np.maximum.accumulate(highs[::-1])[::-1]
    suffix_min_low = np.minimum.accumulate(lows[::-1])[::-1]

    for sw in swings:
        start_idx = sw.index + 1
        if start_idx >= n:
            continue

        if sw.kind == "high":
            # Rejet O(1) : si max futur <= swing.price, jamais sweepe (strict >)
            if suffix_max_high[start_idx] <= sw.price:
                continue
            j = first_breach(highs, start_idx, sw.price, np.greater)
            if j < 0:
                continue
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
            if suffix_min_low[start_idx] >= sw.price:
                continue
            j = first_breach(lows, start_idx, sw.price, np.less)
            if j < 0:
                continue
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
