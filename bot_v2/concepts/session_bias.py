"""Session bias intraday — bias different selon London / NY (bible §17, XQ6A6w3brwU).

Concept Vizion : le bias daily est global, mais chaque SESSION peut faire son
propre mouvement. Exemples :
- "London Reversal" (XQ6A6w3brwU) : London fait un sens, NY inverse au cash open.
- "NY Manipulation" : NY ouvre en faisant le mauvais sens, puis distribue.
- "NY Reversal" : tendance NY contre la matinee.

Methode :
1. On regarde la session en cours (London = 02-05 NY, NY AM = 9-11 NY).
2. On regarde le sens du mouvement DEJA fait dans la session.
3. Si l'OB tente d'aller DANS le sens deja fait = continuation (bonus).
4. Si l'OB tente d'INVERSER ce qui a ete fait = reversal play (bonus encore plus
   pres de la cloture de session).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from bot_v2.concepts.killzones import KILLZONES, NY_TZ, killzone_at, to_ny_time


Direction = Literal["bullish", "bearish"]


@dataclass(frozen=True)
class SessionContext:
    """Contexte de la session en cours pour un timestamp donne."""
    session_name: str | None        # "London", "NY_AM", "NY_PM", "NY_Lunch", None hors KZ
    session_direction: Direction | None  # bullish/bearish/None selon mvt deja fait
    minutes_into_session: int       # minutes ecoulees dans la session
    range_so_far_pct: float         # range deja fait / range moyen 5j (pour intensite)


def _session_bounds_ny(session_name: str) -> tuple[int, int, int, int] | None:
    """Retourne (start_h, start_m, end_h, end_m) en heure NY pour une session."""
    for kz in KILLZONES:
        if kz.name == session_name:
            return (kz.start_hour, kz.start_minute, kz.end_hour, kz.end_minute)
    return None


def get_session_context(
    df: pd.DataFrame,
    ts: pd.Timestamp,
) -> SessionContext:
    """Recolte le contexte de session pour un timestamp."""
    session_name = killzone_at(ts)
    if session_name is None:
        return SessionContext(None, None, 0, 0.0)

    bounds = _session_bounds_ny(session_name)
    if bounds is None:
        return SessionContext(session_name, None, 0, 0.0)

    # Calcule le debut de la session en NY
    ny_t = to_ny_time(ts)
    session_start_ny = ny_t.replace(hour=bounds[0], minute=bounds[1], second=0, microsecond=0)
    # Si on a traverse minuit (Asia), on ajuste
    if session_start_ny > ny_t:
        session_start_ny = session_start_ny - pd.Timedelta(days=1)

    minutes = int((ny_t - session_start_ny).total_seconds() / 60)

    # Bougies de la session jusqu'a maintenant
    session_start_utc = session_start_ny.astimezone(__import__("zoneinfo").ZoneInfo("UTC"))
    session_bars = df[(df.index >= pd.Timestamp(session_start_utc)) & (df.index <= ts)]
    if len(session_bars) < 1:
        return SessionContext(session_name, None, minutes, 0.0)

    # Direction de la session
    open_p = float(session_bars.iloc[0]["open"])
    curr_p = float(session_bars.iloc[-1]["close"])
    if curr_p > open_p:
        direction: Direction = "bullish"
    elif curr_p < open_p:
        direction = "bearish"
    else:
        direction = None

    # Range so far vs moyenne 5 jours
    range_so_far = float(session_bars["high"].max() - session_bars["low"].min())
    # Range moyen sur 5 sessions equivalentes
    avg_range = 0.0
    try:
        last_5d_idx = df.index.max() - pd.Timedelta(days=5)
        df_5d = df[df.index >= last_5d_idx]
        # Approximation : moyenne high-low sur la fenetre 5j
        if len(df_5d) > 0:
            avg_range = float((df_5d["high"].rolling(60).max() - df_5d["low"].rolling(60).min()).mean())
    except Exception:
        pass

    range_pct = (range_so_far / avg_range * 100) if avg_range > 0 else 0.0

    return SessionContext(session_name, direction, minutes, range_pct)


def session_play_score(
    ctx: SessionContext,
    ob_direction: Direction,
) -> int:
    """Score 0-15 selon la coherence de l'OB avec la session.

    - OB dans le SENS de la session, dans la 1ere moitie -> continuation (+10).
    - OB CONTRE le sens, dans la 2eme moitie -> reversal play (+15).
    - OB sans session active -> +0.
    """
    if ctx.session_name is None or ctx.session_direction is None:
        return 0

    bounds = _session_bounds_ny(ctx.session_name)
    if bounds is None:
        return 0

    # Duree session en minutes
    session_minutes = (bounds[2] - bounds[0]) * 60 + (bounds[3] - bounds[1])
    if session_minutes <= 0:
        session_minutes = 60  # fallback
    half = session_minutes / 2

    if ob_direction == ctx.session_direction:
        # Continuation
        return 10 if ctx.minutes_into_session < half else 5
    else:
        # Reversal play : plus pertinent en 2eme moitie de session
        return 15 if ctx.minutes_into_session > half else 5


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent))
    from bot_v2.data_loader import load

    df = load("XAUUSD", "M5")
    # Test sur derniere bougie
    last_ts = df.index[-1]
    ctx = get_session_context(df, last_ts)
    print(f"Timestamp : {last_ts}")
    print(f"  Session     : {ctx.session_name}")
    print(f"  Direction   : {ctx.session_direction}")
    print(f"  Minutes in  : {ctx.minutes_into_session}")
    print(f"  Range so far: {ctx.range_so_far_pct:.0f}%")

    # Score pour OB long et OB short
    print(f"\n  Score OB long  : {session_play_score(ctx, 'bullish')}/15")
    print(f"  Score OB short : {session_play_score(ctx, 'bearish')}/15")
