"""Fair Value Gap (FVG), IFVG, Volume Imbalance — bible §3.

Definitions Vizion :
- **FVG bullish (Busy)** : sur 3 bougies consecutives [i-1, i, i+1] :
    low[i+1] > high[i-1]  => fenetre = (high[i-1], low[i+1])
  Le corps de la bougie centrale forme un "gap" non couvert par les meches
  des bougies adjacentes.
- **FVG bearish (CBI)** : symetrique :
    high[i+1] < low[i-1]  => fenetre = (high[i+1], low[i-1])

- **Rebalance** : un FVG est rebalance quand le price revient et comble la fenetre
  (au moins une fois traverser).

- **IFVG (Inverse FVG)** : un FVG TRAVERSE EN CLOTURE dans l'autre sens devient
  un IFVG (TrTBuk2Ttl4). Signature d'inversion.

- **Volume Imbalance** : gap entre cloture de bougie i et ouverture de bougie i+1
  (intra-bougie, pas weekend). A combler comme un FVG.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd


FVGDirection = Literal["bullish", "bearish"]


@dataclass(frozen=True)
class FVG:
    """Un Fair Value Gap (3 bougies)."""
    direction: FVGDirection      # "bullish" (Busy) | "bearish" (CBI)
    # La bougie CENTRALE est celle qui "casse" (l'index reference)
    center_index: int
    center_ts: pd.Timestamp
    # Bornes de la fenetre FVG
    top: float                   # bord haut de la fenetre
    bottom: float                # bord bas de la fenetre
    # Etat : rebalance ou non, inverse en IFVG ou non
    rebalanced: bool = False
    rebalance_index: int | None = None
    inversed: bool = False              # devenu un IFVG
    inverse_index: int | None = None


@dataclass(frozen=True)
class VolumeImbalance:
    """Gap entre close[i] et open[i+1]."""
    index: int                  # index de la bougie i (gap entre i et i+1)
    timestamp: pd.Timestamp
    direction: FVGDirection
    top: float
    bottom: float


def detect_fvg(df: pd.DataFrame) -> list[FVG]:
    """Detecte tous les FVG dans le DataFrame.

    FVG bullish (Busy) : low[i+1] > high[i-1]
    FVG bearish (CBI)  : high[i+1] < low[i-1]
    """
    if len(df) < 3:
        return []

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values

    fvgs: list[FVG] = []

    for i in range(1, len(df) - 1):
        # Bullish : low du future > high du passe
        if lows[i + 1] > highs[i - 1]:
            top = float(lows[i + 1])
            bottom = float(highs[i - 1])
            # Cherche si rebalance plus tard
            rebalanced = False
            rebalance_idx = None
            inversed = False
            inverse_idx = None
            for j in range(i + 2, len(df)):
                # Rebalance : low de la bougie j entre dans la fenetre
                if not rebalanced and lows[j] <= top:
                    rebalanced = True
                    rebalance_idx = j
                # Inverse : cloture sous le bottom (corps, pas meche)
                if rebalanced and closes[j] < bottom:
                    inversed = True
                    inverse_idx = j
                    break
            fvgs.append(FVG(
                direction="bullish",
                center_index=i,
                center_ts=df.index[i],
                top=top,
                bottom=bottom,
                rebalanced=rebalanced,
                rebalance_index=rebalance_idx,
                inversed=inversed,
                inverse_index=inverse_idx,
            ))

        # Bearish : high du future < low du passe
        if highs[i + 1] < lows[i - 1]:
            top = float(lows[i - 1])
            bottom = float(highs[i + 1])
            rebalanced = False
            rebalance_idx = None
            inversed = False
            inverse_idx = None
            for j in range(i + 2, len(df)):
                if not rebalanced and highs[j] >= bottom:
                    rebalanced = True
                    rebalance_idx = j
                if rebalanced and closes[j] > top:
                    inversed = True
                    inverse_idx = j
                    break
            fvgs.append(FVG(
                direction="bearish",
                center_index=i,
                center_ts=df.index[i],
                top=top,
                bottom=bottom,
                rebalanced=rebalanced,
                rebalance_index=rebalance_idx,
                inversed=inversed,
                inverse_index=inverse_idx,
            ))

    return fvgs


def detect_volume_imbalance(df: pd.DataFrame) -> list[VolumeImbalance]:
    """Detecte les gaps entre close[i] et open[i+1] (intra-bougie).

    Ignore les gaps weekend (>= 8h entre 2 bougies).
    """
    opens = df["open"].values
    closes = df["close"].values
    out: list[VolumeImbalance] = []

    for i in range(len(df) - 1):
        # Skip weekend gaps : si plus de 8h entre 2 bougies, on ignore
        delta_h = (df.index[i + 1] - df.index[i]).total_seconds() / 3600
        if delta_h > 8:
            continue

        c = closes[i]
        o = opens[i + 1]
        if o > c:
            out.append(VolumeImbalance(
                index=i,
                timestamp=df.index[i],
                direction="bullish",
                top=float(o),
                bottom=float(c),
            ))
        elif o < c:
            out.append(VolumeImbalance(
                index=i,
                timestamp=df.index[i],
                direction="bearish",
                top=float(c),
                bottom=float(o),
            ))

    return out


def live_fvgs(fvgs: list[FVG]) -> list[FVG]:
    """Retourne uniquement les FVG non encore rebalances (encore "ouverts")."""
    return [f for f in fvgs if not f.rebalanced]


def ifvgs(fvgs: list[FVG]) -> list[FVG]:
    """Retourne uniquement les FVG qui ont ete inverses (devenus IFVG)."""
    return [f for f in fvgs if f.inversed]


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent))
    from bot_v2.data_loader import load

    df = load("XAUUSD", "M5")
    mask = df.index >= (df.index.max() - pd.Timedelta(days=7))
    df_week = df[mask]
    print(f"XAUUSD M5 (7 derniers jours) : {len(df_week)} bougies\n")

    fvgs = detect_fvg(df_week)
    print(f"FVG detectes : {len(fvgs)}")
    print(f"  - Bullish (Busy) : {sum(1 for f in fvgs if f.direction == 'bullish')}")
    print(f"  - Bearish (CBI)  : {sum(1 for f in fvgs if f.direction == 'bearish')}")
    print(f"  - Rebalances    : {sum(1 for f in fvgs if f.rebalanced)}")
    print(f"  - Encore LIVE   : {sum(1 for f in fvgs if not f.rebalanced)}")
    print(f"  - Inverses (IFVG): {sum(1 for f in fvgs if f.inversed)}")

    vi = detect_volume_imbalance(df_week)
    print(f"\nVolume Imbalances : {len(vi)}")
    print(f"  - Bullish : {sum(1 for v in vi if v.direction == 'bullish')}")
    print(f"  - Bearish : {sum(1 for v in vi if v.direction == 'bearish')}")

    # Affiche les 5 derniers FVG live
    live = live_fvgs(fvgs)
    print(f"\n=== 5 derniers FVG encore LIVE ===")
    for f in live[-5:]:
        size = f.top - f.bottom
        print(f"  {f.center_ts} | {f.direction:8s} | top={f.top:.3f} bot={f.bottom:.3f} size={size:.3f}")

    print(f"\n=== 3 IFVG (FVG inverses) ===")
    for f in ifvgs(fvgs)[:3]:
        print(f"  {f.center_ts} | etait {f.direction} | inverse a {df_week.index[f.inverse_index]}")
