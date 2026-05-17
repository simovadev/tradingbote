"""Filtres temporels avances : day-of-week, news blackout, session timing fin."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Literal

import pandas as pd

DayQuality = Literal["best", "good", "neutral", "avoid"]


# Day-of-week (ICT classique) :
# Mardi-Jeudi = vrais jours smart money
# Lundi = position-taking, manipulation
# Vendredi = volatile, prise de profit
# Weekend = ferme (pas de bougies normalement)
DAY_QUALITY: dict[int, DayQuality] = {
    0: "neutral",   # Lundi
    1: "best",      # Mardi
    2: "best",      # Mercredi
    3: "best",      # Jeudi
    4: "neutral",   # Vendredi
    5: "avoid",     # Samedi
    6: "avoid",     # Dimanche
}

DAY_WEIGHT: dict[DayQuality, int] = {
    "best": 5,
    "good": 3,
    "neutral": 0,
    "avoid": -10,
}

DAY_LABEL = {0: "Lundi", 1: "Mardi", 2: "Mercredi", 3: "Jeudi", 4: "Vendredi", 5: "Samedi", 6: "Dimanche"}


@dataclass
class DayAnalysis:
    weekday: int
    quality: DayQuality
    weight: int
    label: str


def analyze_day(ts: pd.Timestamp) -> DayAnalysis:
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    wd = ts.weekday()
    q = DAY_QUALITY.get(wd, "neutral")
    return DayAnalysis(
        weekday=wd, quality=q, weight=DAY_WEIGHT[q],
        label=f"{DAY_LABEL.get(wd, '?')} ({q})",
    )


# ============ News blackout ============
# Heures connues a haut impact (sans API news, on hardcode les classiques)
# - NFP : 1er vendredi du mois @ 12:30 UTC (8:30 ET)
# - FOMC : ~8 fois/an, mercredi @ 18:00 UTC (14:00 ET) + press conf 18:30
# - CPI US : ~12 UTC mensuel
# Comme on n'a pas le calendar live, on flag les fenetres a risque :

HIGH_IMPACT_WINDOWS = [
    # NFP (1er vendredi du mois)
    {"name": "NFP", "weekday": 4, "first_of_month": True,
     "start": time(12, 25), "end": time(13, 0)},
    # FOMC (on flag mercredi 17:55 - 19:00 UTC, on filtrera trop large pour rester safe)
    {"name": "FOMC potentiel", "weekday": 2,
     "start": time(17, 55), "end": time(19, 0), "rare": True},
    # CPI US (jour incertain, on flag matin US tous les jours du 10 au 16)
    {"name": "CPI window", "day_range": (10, 16),
     "start": time(12, 25), "end": time(13, 0)},
]


@dataclass
class NewsCheck:
    is_blackout: bool
    risk_window: str | None
    detail: str


def check_news_blackout(ts: pd.Timestamp) -> NewsCheck:
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    t = ts.time()
    wd = ts.weekday()
    dom = ts.day

    # NFP : 1er vendredi du mois = jour 1 a 7 et vendredi
    if wd == 4 and 1 <= dom <= 7 and time(12, 25) <= t <= time(13, 0):
        return NewsCheck(True, "NFP", f"NFP probable @ {t.strftime('%H:%M')}")

    # FOMC : on est conservateur, on flag mercredi 17:55-19:00 mais rare
    # (on laisse passer car beaucoup de faux positifs sinon)
    # Skip pour eviter trop de blackouts

    # CPI : jour 10-16, 12:25-13:00 UTC
    if 10 <= dom <= 16 and time(12, 25) <= t <= time(13, 0) and wd in (1, 2, 3):
        return NewsCheck(True, "CPI window", f"CPI possible @ {t.strftime('%H:%M')}")

    return NewsCheck(False, None, "")
