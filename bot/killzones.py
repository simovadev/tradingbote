"""Killzones ICT — sessions de trading a haute liquidite (calibration user).

REGLE DURE :
- Le bot ne trade QUE pendant une killzone active.
- L'OB n'est valide QUE si sa bougie source a ete formee DANS une killzone.

Heures en UTC (calibrees sur la lecture du user):
- Asia        : 02:00 - 05:00 UTC
- London      : 07:00 - 12:00 UTC
- New York    : 14:00 - 17:00 UTC
- London Close: 17:00 - 20:00 UTC (Power Hour, retournements frequents)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Literal

import pandas as pd

KillzoneName = Literal["asia", "london", "ny", "london_close", "none"]


@dataclass(frozen=True)
class Killzone:
    name: KillzoneName
    label: str
    start: time
    end: time
    weight: int


KILLZONES: list[Killzone] = [
    Killzone("asia",         "Asia",         time(2, 0),  time(5, 0),  weight=10),
    Killzone("london",       "London",       time(7, 0),  time(12, 0), weight=18),
    Killzone("ny",           "NY",           time(14, 0), time(17, 0), weight=18),
    Killzone("london_close", "London Close", time(17, 0), time(20, 0), weight=14),
]


def current_killzone(ts: pd.Timestamp) -> Killzone | None:
    """Retourne la killzone active a un timestamp UTC donne, ou None."""
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")

    t = ts.time()
    # On prend la 1ere qui matche, dans l'ordre = priorite (overlap d'abord serait mieux)
    matches = [kz for kz in KILLZONES if kz.start <= t < kz.end]
    if not matches:
        return None
    # Priorise la plus "forte" si plusieurs matchent (overlap > NY AM > London...)
    return max(matches, key=lambda kz: kz.weight)


def is_in_killzone(ts: pd.Timestamp) -> bool:
    return current_killzone(ts) is not None
