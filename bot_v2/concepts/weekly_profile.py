"""Weekly Profile — bible §10 (0pjj5rM1M18, rdrCs58XZfA).

3 profils weekly Vizion :
1. **Classic Expansion** (le plus frequent) :
   - Lundi : hesitation / fausse direction
   - Mardi : reversal + prise du PDL/PDH lundi -> debut expansion
   - Mardi-Mercredi-Jeudi : expansion (= phase tradable)
   - Vendredi : profit-taking / TGIF
2. **Midweek Reversal** :
   - Lundi-Mardi : expansion initiale
   - Mercredi : retournement
   - Jeudi-Vendredi : expansion dans l'autre sens
3. **Consolidation Reversal** :
   - Lundi-Mercredi : range (pas tradable)
   - Jeudi : breakout
   - Vendredi : suit le breakout

Decision user 2026-05-15 : on trade tous les jours (lundi a vendredi inclus).
Le profil sert a anticiper le sens, pas a filtrer les jours.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd


WeeklyProfile = Literal[
    "classic_expansion_up",
    "classic_expansion_down",
    "midweek_reversal_up",      # commence baissier, finit haussier
    "midweek_reversal_down",
    "consolidation_reversal_up",
    "consolidation_reversal_down",
    "undetermined",
]


@dataclass(frozen=True)
class WeeklyAnalysis:
    """Analyse weekly de la semaine en cours ou passee."""
    week_start: pd.Timestamp        # lundi de la semaine
    monday_open: float
    weekly_high: float
    weekly_low: float
    current_close: float            # derniere cloture disponible
    profile: WeeklyProfile
    # Position relative dans la semaine
    is_above_monday_open: bool
    high_day: int                   # jour de la semaine ou WH a ete fait (0=lun, 4=ven)
    low_day: int                    # jour de la semaine ou WL a ete fait


def get_week_data(df_d1: pd.DataFrame, ref_ts: pd.Timestamp) -> pd.DataFrame | None:
    """Retourne les bougies daily de la semaine contenant `ref_ts` (lundi a vendredi)."""
    # Lundi de la semaine de ref_ts
    monday = ref_ts.normalize() - pd.Timedelta(days=ref_ts.weekday())
    friday = monday + pd.Timedelta(days=4, hours=23, minutes=59)
    week_df = df_d1[(df_d1.index >= monday) & (df_d1.index <= friday)]
    if len(week_df) == 0:
        return None
    return week_df


def classify_weekly_profile(week_df: pd.DataFrame) -> WeeklyProfile:
    """Classifie la semaine en cours.

    Heuristique simple :
    - Si <= 2 jours : undetermined (lundi seul, ou lundi+mardi).
    - Si la cloture vendredi > open lundi de plus de la moitie du range hebdo :
        classic_expansion_up.
    - Inverse pour _down.
    - Si le WH ET le WL sont faits le meme jour milieu de semaine (mer ou jeu) :
        midweek_reversal.
    - Si le range des 3 premiers jours < 30% du range total : consolidation_reversal.
    """
    if len(week_df) < 3:
        return "undetermined"

    mon_open = float(week_df.iloc[0]["open"])
    cur_close = float(week_df.iloc[-1]["close"])
    wh = float(week_df["high"].max())
    wl = float(week_df["low"].min())
    rng = wh - wl
    if rng == 0:
        return "undetermined"

    # Quel jour les extremes ?
    high_day = int(week_df["high"].idxmax().weekday())
    low_day = int(week_df["low"].idxmin().weekday())

    # Classic expansion : cloture > open de plus de 50% du range
    delta = cur_close - mon_open
    if abs(delta) > rng * 0.4:
        if delta > 0:
            return "classic_expansion_up"
        return "classic_expansion_down"

    # Midweek reversal : extremes faits en milieu de semaine
    if (high_day in (2, 3) or low_day in (2, 3)) and abs(delta) > rng * 0.2:
        return "midweek_reversal_up" if delta > 0 else "midweek_reversal_down"

    # Consolidation reversal : ouverture range etroit, breakout fin de semaine
    if len(week_df) >= 4:
        first3_rng = float(week_df.iloc[:3]["high"].max() - week_df.iloc[:3]["low"].min())
        if first3_rng < rng * 0.5:
            return "consolidation_reversal_up" if delta > 0 else "consolidation_reversal_down"

    return "undetermined"


def analyze_week(df_d1: pd.DataFrame, ref_ts: pd.Timestamp | None = None) -> WeeklyAnalysis | None:
    """Analyse la semaine de `ref_ts` (defaut : derniere date du DataFrame)."""
    if ref_ts is None:
        ref_ts = df_d1.index.max()
    week_df = get_week_data(df_d1, ref_ts)
    if week_df is None or len(week_df) == 0:
        return None

    monday_open = float(week_df.iloc[0]["open"])
    wh = float(week_df["high"].max())
    wl = float(week_df["low"].min())
    cur_close = float(week_df.iloc[-1]["close"])
    profile = classify_weekly_profile(week_df)

    return WeeklyAnalysis(
        week_start=week_df.index[0].normalize(),
        monday_open=monday_open,
        weekly_high=wh,
        weekly_low=wl,
        current_close=cur_close,
        profile=profile,
        is_above_monday_open=cur_close > monday_open,
        high_day=int(week_df["high"].idxmax().weekday()),
        low_day=int(week_df["low"].idxmin().weekday()),
    )


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent))
    from bot_v2.data_loader import load
    from bot_v2.concepts.daily_bias import build_d1_from_h1

    df_h1 = load("XAUUSD", "H1")
    df_d1 = build_d1_from_h1(df_h1)
    print(f"XAUUSD D1 : {len(df_d1)} bougies")

    # Analyse semaine en cours
    wa = analyze_week(df_d1)
    if wa:
        print(f"\n=== Semaine du {wa.week_start.date()} ===")
        print(f"  Monday open : {wa.monday_open:.3f}")
        print(f"  Weekly High : {wa.weekly_high:.3f} (jour {wa.high_day})")
        print(f"  Weekly Low  : {wa.weekly_low:.3f} (jour {wa.low_day})")
        print(f"  Cur close   : {wa.current_close:.3f}")
        print(f"  Profile     : {wa.profile}")
        print(f"  Au-dessus open lundi ? {wa.is_above_monday_open}")

    # Analyse des 6 dernieres semaines
    print("\n=== Profiles des 6 dernieres semaines ===")
    mondays = pd.date_range(end=df_d1.index.max(), periods=6, freq="W-MON", tz="UTC")
    for m in mondays:
        wa = analyze_week(df_d1, m + pd.Timedelta(days=4))
        if wa:
            print(f"  {wa.week_start.date()} | {wa.profile:30s} | "
                  f"open={wa.monday_open:.2f} close={wa.current_close:.2f}")
