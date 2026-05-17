"""SMT Divergence — bible §7 + decision user 2026-05-15.

Definition Vizion (0dzjqeTDXxU, uOv1znt2uAY) :
- Sur 2 actifs CORRELES : l'un prend une liquidite que l'autre ne prend pas.
- Signature algorithmique : "l'actif fort prend, l'actif faible non".

Paires (bot_v2/config.py SMT_PAIRS) :
- positive : bougent dans le meme sens (NAS/SPX, XAU/XAG, NAS/GER40, USOIL/UKOIL)
- inverse  : sens oppose (XAU/DXY)

Detection bullish SMT :
- Le primaire fait un swing LOW plus bas (prise du low)
- Le correle (positive) ne fait pas de low plus bas
- => SMT bullish : le marche divergent annonce une hausse

Detection bearish SMT :
- Primaire fait un swing HIGH plus haut (prise du high)
- Correle (positive) ne fait pas de high plus haut

Pour la correlation INVERSE :
- Bullish SMT : primaire fait low plus bas, correle ne fait PAS de high plus haut
- Bearish SMT : primaire fait high plus haut, correle ne fait PAS de low plus bas

Decision user 2026-05-15 : BONUS, pas filtre eliminatoire.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from bot_v2.concepts.liquidity import Swing, find_swings
from bot_v2.config import SMT_PAIRS


Direction = Literal["bullish", "bearish"]


@dataclass(frozen=True)
class SMTDivergence:
    """Une divergence SMT detectee."""
    primary: str                    # actif primaire (ex. "XAUUSD")
    correlated: str                 # actif correle (ex. "XAGUSD")
    correlation: Literal["positive", "inverse"]
    direction: Direction            # direction du setup attendu
    # Timestamps
    primary_extreme_ts: pd.Timestamp
    correlated_window_start: pd.Timestamp
    correlated_window_end: pd.Timestamp


def detect_smt(
    df_primary: pd.DataFrame,
    df_correlated: pd.DataFrame,
    correlation: Literal["positive", "inverse"],
    primary_name: str,
    correlated_name: str,
    swing_strength: int = 2,
    lookback_bars: int = 20,
) -> list[SMTDivergence]:
    """Detecte les divergences SMT entre 2 actifs.

    Logique :
    - Trouve les 2 derniers swing LOWS du primaire.
    - Si LL2 < LL1 (low plus bas), regarde si le correlated a fait la meme chose
      sur la meme fenetre temporelle.
    - Si non => SMT bullish.
    - Symetrique pour HIGHs (SMT bearish).
    """
    out: list[SMTDivergence] = []

    swings_p = find_swings(df_primary, strength=swing_strength)
    if len(swings_p) < 2:
        return out

    lows_p = [s for s in swings_p if s.kind == "low"][-3:]   # 3 derniers lows
    highs_p = [s for s in swings_p if s.kind == "high"][-3:]

    # === Bullish SMT (primaire fait un low plus bas)
    if len(lows_p) >= 2:
        l1, l2 = lows_p[-2], lows_p[-1]
        if l2.price < l1.price:
            # Fenetre temporelle correspondante
            t_start = l1.timestamp
            t_end = l2.timestamp
            df_c_window = df_correlated[(df_correlated.index >= t_start) & (df_correlated.index <= t_end)]
            if len(df_c_window) >= 2:
                # Pour positive : on regarde si le correle a fait un low plus bas
                # Pour inverse  : on regarde si le correle a fait un high plus haut
                if correlation == "positive":
                    c_low_at_l1 = float(df_correlated[df_correlated.index <= t_start].iloc[-5:]["low"].min())
                    c_low_at_l2 = float(df_c_window["low"].min())
                    if c_low_at_l2 >= c_low_at_l1:
                        out.append(SMTDivergence(
                            primary=primary_name,
                            correlated=correlated_name,
                            correlation=correlation,
                            direction="bullish",
                            primary_extreme_ts=l2.timestamp,
                            correlated_window_start=t_start,
                            correlated_window_end=t_end,
                        ))
                else:  # inverse
                    c_high_at_l1 = float(df_correlated[df_correlated.index <= t_start].iloc[-5:]["high"].max())
                    c_high_at_l2 = float(df_c_window["high"].max())
                    if c_high_at_l2 <= c_high_at_l1:
                        out.append(SMTDivergence(
                            primary=primary_name,
                            correlated=correlated_name,
                            correlation=correlation,
                            direction="bullish",
                            primary_extreme_ts=l2.timestamp,
                            correlated_window_start=t_start,
                            correlated_window_end=t_end,
                        ))

    # === Bearish SMT (primaire fait un high plus haut)
    if len(highs_p) >= 2:
        h1, h2 = highs_p[-2], highs_p[-1]
        if h2.price > h1.price:
            t_start = h1.timestamp
            t_end = h2.timestamp
            df_c_window = df_correlated[(df_correlated.index >= t_start) & (df_correlated.index <= t_end)]
            if len(df_c_window) >= 2:
                if correlation == "positive":
                    c_high_at_h1 = float(df_correlated[df_correlated.index <= t_start].iloc[-5:]["high"].max())
                    c_high_at_h2 = float(df_c_window["high"].max())
                    if c_high_at_h2 <= c_high_at_h1:
                        out.append(SMTDivergence(
                            primary=primary_name,
                            correlated=correlated_name,
                            correlation=correlation,
                            direction="bearish",
                            primary_extreme_ts=h2.timestamp,
                            correlated_window_start=t_start,
                            correlated_window_end=t_end,
                        ))
                else:  # inverse
                    c_low_at_h1 = float(df_correlated[df_correlated.index <= t_start].iloc[-5:]["low"].min())
                    c_low_at_h2 = float(df_c_window["low"].min())
                    if c_low_at_h2 >= c_low_at_h1:
                        out.append(SMTDivergence(
                            primary=primary_name,
                            correlated=correlated_name,
                            correlation=correlation,
                            direction="bearish",
                            primary_extreme_ts=h2.timestamp,
                            correlated_window_start=t_start,
                            correlated_window_end=t_end,
                        ))

    return out


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent))
    from bot_v2.data_loader import load

    primary = "XAUUSD"
    print(f"=== SMT pour {primary} ===\n")

    df_p = load(primary, "M5")
    mask = df_p.index >= (df_p.index.max() - pd.Timedelta(days=3))
    df_p = df_p[mask]

    for correlated, corr_type in SMT_PAIRS.get(primary, []):
        try:
            df_c = load(correlated, "M5")
        except Exception:
            # Fallback M1 pour XAG/SPX (pas de M5 en cache)
            try:
                df_c = load(correlated, "M1")
            except Exception:
                print(f"  {correlated} : pas de cache, skip")
                continue
        mask_c = df_c.index >= (df_c.index.max() - pd.Timedelta(days=3))
        df_c = df_c[mask_c]

        smts = detect_smt(df_p, df_c, corr_type, primary, correlated)
        print(f"\n  {primary} vs {correlated} ({corr_type}) : {len(smts)} divergences")
        for s in smts:
            print(f"    {s.direction:8s} | extreme @ {s.primary_extreme_ts}")
