"""Discount / Premium / Equilibrium — bible §6.

Sur le dernier mouvement directionnel (swing low -> swing high pour un leg haussier,
ou inverse pour un leg baissier) :
- **Equilibrium** = 50% Fibo
- **Discount** = sous l'equilibrium (zone d'ACHAT)
- **Premium**  = au-dessus de l'equilibrium (zone de VENTE)

Regle Vizion (eS6adgOouqI, xUb364bilGQ) :
- OB long doit etre en zone DISCOUNT.
- OB short doit etre en zone PREMIUM.
- OB en zone OPPOSEE = invalide (filtre eliminatoire).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from bot_v2.concepts.liquidity import Swing, find_swings


Zone = Literal["discount", "premium", "equilibrium"]


@dataclass(frozen=True)
class FibRange:
    """Le range Fibo entre 2 swings consecutifs de signes opposes."""
    # Swing bas et swing haut (peu importe l'ordre temporel)
    low_price: float
    high_price: float
    low_ts: pd.Timestamp
    high_ts: pd.Timestamp
    # Direction du leg : "up" si low -> high (leg haussier), "down" sinon
    direction: Literal["up", "down"]

    @property
    def equilibrium(self) -> float:
        return (self.low_price + self.high_price) / 2

    @property
    def range_size(self) -> float:
        return self.high_price - self.low_price

    def zone_of(self, price: float) -> Zone:
        """Dans quelle zone se trouve un prix donne."""
        eq = self.equilibrium
        # Tolerance ±5% du range autour de l'equilibrium = "equilibrium"
        tol = self.range_size * 0.05
        if abs(price - eq) <= tol:
            return "equilibrium"
        return "discount" if price < eq else "premium"

    def fib_level(self, price: float) -> float:
        """Retourne le niveau Fibo (0 = low, 100 = high)."""
        if self.range_size == 0:
            return 50.0
        return (price - self.low_price) / self.range_size * 100


def latest_fib_range(df: pd.DataFrame, swing_strength: int = 2) -> FibRange | None:
    """Calcule le range Fibo du DERNIER LEG DIRECTIONNEL PROPRE (bible §6.2 + xUb364bilGQ).

    Correction 2026-05-15 : on ne prend plus les 2 derniers swings extremes
    (qui peuvent etre tres anciens et donner un range obsolete). On prend le DERNIER
    LEG = la paire de swings consecutifs de signes opposes la plus recente.

    Logique :
    - Trier les swings par index.
    - Prendre les 2 derniers swings consecutifs de SIGNES OPPOSES.
    - Le range Fibo = entre ces 2 swings. Direction = sens du plus recent.
    """
    swings = find_swings(df, strength=swing_strength)
    if len(swings) < 2:
        return None

    # On parcourt les swings de la fin vers le debut et on cherche la derniere paire
    # consecutive de signes opposes.
    last = swings[-1]
    prev = None
    for s in reversed(swings[:-1]):
        if s.kind != last.kind:
            prev = s
            break
    if prev is None:
        return None

    # last = swing le plus recent, prev = swing oppose precedent.
    # Le leg directionnel va de prev -> last.
    if last.kind == "high":
        # Leg haussier : prev (low) -> last (high)
        low_swing, high_swing, direction = prev, last, "up"
    else:
        # Leg baissier : prev (high) -> last (low)
        low_swing, high_swing, direction = last, prev, "down"

    return FibRange(
        low_price=low_swing.price,
        high_price=high_swing.price,
        low_ts=low_swing.timestamp,
        high_ts=high_swing.timestamp,
        direction=direction,
    )


def is_in_discount(price: float, fib: FibRange) -> bool:
    return fib.zone_of(price) == "discount"


def is_in_premium(price: float, fib: FibRange) -> bool:
    return fib.zone_of(price) == "premium"


def ob_zone_check(ob_price: float, direction: Literal["bullish", "bearish"], fib: FibRange) -> bool:
    """Verifie qu'un OB est dans la BONNE zone selon sa direction.

    - OB bullish (long) => doit etre en DISCOUNT.
    - OB bearish (short) => doit etre en PREMIUM.
    """
    zone = fib.zone_of(ob_price)
    if direction == "bullish":
        return zone in ("discount", "equilibrium")
    return zone in ("premium", "equilibrium")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent))
    from bot_v2.data_loader import load
    from bot_v2.concepts.order_block import detect_order_blocks

    df = load("XAUUSD", "M5")
    mask = df.index >= (df.index.max() - pd.Timedelta(days=7))
    df_week = df[mask]
    print(f"XAUUSD M5 (7 derniers jours) : {len(df_week)} bougies\n")

    # 1. Range Fibo sur le H1 (plus pertinent que M5 pour D/P)
    df_h1 = load("XAUUSD", "H1")
    mask_h1 = df_h1.index >= (df_h1.index.max() - pd.Timedelta(days=14))
    df_h1_recent = df_h1[mask_h1]
    fib = latest_fib_range(df_h1_recent, swing_strength=2)
    if fib:
        print("Dernier Fib Range (H1 14j) :")
        print(f"  Low @ {fib.low_ts}  : {fib.low_price:.3f}")
        print(f"  High @ {fib.high_ts} : {fib.high_price:.3f}")
        print(f"  Equilibrium : {fib.equilibrium:.3f}")
        print(f"  Direction : {fib.direction}")
        print(f"  Range size : {fib.range_size:.3f}")
        last_close = float(df_h1_recent["close"].iloc[-1])
        print(f"\n  Prix actuel : {last_close:.3f}")
        print(f"  Zone : {fib.zone_of(last_close)}")
        print(f"  Fib level : {fib.fib_level(last_close):.1f}%")

    # 2. Verifie les OB M5 dans le contexte Fibo M5
    fib_m5 = latest_fib_range(df_week, swing_strength=3)
    if fib_m5:
        obs = detect_order_blocks(df_week)
        print(f"\nOB detectes (M5) : {len(obs)}")
        n_in_zone = 0
        for ob in obs:
            ob_mid = (ob.ob_high + ob.ob_low) / 2
            if ob_zone_check(ob_mid, ob.direction, fib_m5):
                n_in_zone += 1
        print(f"OB en BONNE zone (Discount pour bullish, Premium pour bearish) : {n_in_zone}/{len(obs)}")
