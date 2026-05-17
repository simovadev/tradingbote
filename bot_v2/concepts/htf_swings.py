"""Swings HTF pertinents — pour TP et niveaux cles.

Probleme V1 : le TP etait souvent fixed_rr=2.0 car find_next_swing_target
cherchait des swings LTF (souvent trop proches ou deja pris).

Solution : utiliser les swings H1/H4/D1 qui sont les VRAIES cibles algo.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from bot_v2.concepts.liquidity import Swing, find_swings


@dataclass(frozen=True)
class HTFSwing:
    """Un swing detecte en HTF, traduit en niveau de prix utilisable en LTF."""
    swing: Swing
    tf_name: str               # "H1", "H4", "D1"
    is_live: bool              # pas encore pris en LTF
    # Compte combien de fois ce niveau a ete TESTE (mais pas pris)
    retest_count: int = 0


def collect_htf_swings(
    htf_dfs: dict[str, pd.DataFrame],
    swing_strength: int = 3,
) -> list[HTFSwing]:
    """Recolte tous les swings sur plusieurs TFs HTF.

    Args:
        htf_dfs: {"H1": df_h1, "H4": df_h4, "D1": df_d1, ...}
        swing_strength: plus eleve sur HTF = swings plus significatifs.
    """
    out: list[HTFSwing] = []
    for tf_name, df in htf_dfs.items():
        if df is None or len(df) < 10:
            continue
        swings = find_swings(df, strength=swing_strength)
        for s in swings:
            out.append(HTFSwing(swing=s, tf_name=tf_name, is_live=True))
    return out


def find_next_htf_target(
    htf_swings: list[HTFSwing],
    entry_price: float,
    direction: Literal["bullish", "bearish"],
    min_distance_pct: float = 0.0005,
    prefer_tf: list[str] | None = None,
) -> HTFSwing | None:
    """Trouve le prochain swing HTF pertinent comme TP.

    Prefere les swings du TF le plus haut disponible.
    """
    if prefer_tf is None:
        # Analyse 2026-05-15 : on prefere H1/H4 a D1 car les TP D1 sont trop loins
        # et le price reverse souvent avant. Bible §13.3 "prochain swing pertinent".
        prefer_tf = ["H1", "H4", "D1"]

    # Groupe par TF
    by_tf: dict[str, list[HTFSwing]] = {}
    for hs in htf_swings:
        by_tf.setdefault(hs.tf_name, []).append(hs)

    # Parcourt par ordre de preference TF
    for tf in prefer_tf:
        if tf not in by_tf:
            continue
        candidates = by_tf[tf]
        if direction == "bullish":
            # Swings high au-dessus de entry
            targets = [
                hs for hs in candidates
                if hs.swing.kind == "high"
                and hs.swing.price > entry_price * (1 + min_distance_pct)
                and hs.is_live
            ]
            if targets:
                # Le plus proche au-dessus
                return min(targets, key=lambda h: h.swing.price)
        else:
            targets = [
                hs for hs in candidates
                if hs.swing.kind == "low"
                and hs.swing.price < entry_price * (1 - min_distance_pct)
                and hs.is_live
            ]
            if targets:
                return max(targets, key=lambda h: h.swing.price)

    return None


def count_retests(
    df: pd.DataFrame,
    price_level: float,
    tolerance_pct: float = 0.001,
    after_index: int = 0,
) -> int:
    """Compte combien de fois le price a TOUCHE (sans casser) un niveau.

    Un "touch" = le high ou le low de bougie touche le niveau ± tolerance.
    On ne compte PAS les fois ou le price clos au-dela (= cassure).
    """
    if after_index >= len(df):
        return 0
    sub = df.iloc[after_index:]
    tol = abs(price_level) * tolerance_pct
    touches = 0
    last_touched = False
    for _, row in sub.iterrows():
        in_range = (row["low"] - tol) <= price_level <= (row["high"] + tol)
        if in_range:
            if not last_touched:
                touches += 1
                last_touched = True
        else:
            last_touched = False
    return touches


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent))
    from bot_v2.data_loader import load

    df_h1 = load("XAUUSD", "H1")
    df_h4 = load("XAUUSD", "H4")
    df_d1 = load("XAUUSD", "D1")
    htf_swings = collect_htf_swings({"H1": df_h1, "H4": df_h4, "D1": df_d1}, swing_strength=3)

    print(f"Swings HTF total : {len(htf_swings)}")
    from collections import Counter
    by_tf = Counter(hs.tf_name for hs in htf_swings)
    for tf, c in by_tf.most_common():
        print(f"  {tf} : {c}")

    last_price = float(df_h1["close"].iloc[-1])
    print(f"\nPrix actuel XAU : {last_price:.3f}")

    target_up = find_next_htf_target(htf_swings, last_price, "bullish")
    target_dn = find_next_htf_target(htf_swings, last_price, "bearish")
    if target_up:
        delta = target_up.swing.price - last_price
        print(f"Prochaine cible HAUT : {target_up.swing.price:.3f} (TF {target_up.tf_name}, +{delta:.3f})")
    if target_dn:
        delta = last_price - target_dn.swing.price
        print(f"Prochaine cible BAS  : {target_dn.swing.price:.3f} (TF {target_dn.tf_name}, -{delta:.3f})")
