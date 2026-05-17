"""Power of Three (PO3) + Judas Swing — concepts ICT classiques.

PO3 = Accumulation + Manipulation + Distribution
1. Accumulation : Asia range = consolidation, smart money construit ses positions
2. Manipulation : Debut London ou NY, sweep dans LA DIRECTION OPPOSEE au vrai move
   (= Judas Swing : faux mouvement qui piege les retail traders)
3. Distribution : vrai move dans la direction opposee au Judas

Judas Swing typique :
- 07:00-08:30 UTC : London open, premier mouvement souvent un Judas
- 12:00-13:30 UTC : NY open, idem

Si on detecte un sweep du Asia range puis un retournement immediat -> c'est le Judas.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Literal

import pandas as pd

from bot.detectors.sessions import DailyLevels

JudasType = Literal["asia_high_sweep", "asia_low_sweep", "none"]


@dataclass
class JudasAnalysis:
    detected: bool
    judas_type: JudasType
    judas_time: pd.Timestamp | None
    detail: str


def detect_judas(
    df_m1: pd.DataFrame,
    levels: DailyLevels,
    until_time: pd.Timestamp,
) -> JudasAnalysis:
    """Detecte si un Judas Swing a eu lieu avant `until_time`.

    Judas = mouvement qui :
    1. Sort des bornes du Asia range (sweep high ou low)
    2. Pendant l'ouverture London (07:00-09:00 UTC) ou NY (12:00-13:30 UTC)
    3. Suivi d'un retournement (le prix revient DANS le range Asia ou plus loin
       dans la direction opposee).
    """
    if levels.asia_high is None or levels.asia_low is None:
        return JudasAnalysis(False, "none", None, "Pas de Asia range")

    if until_time.tz is None:
        until_time = until_time.tz_localize("UTC")

    # Fenetres London open et NY open
    day = until_time.normalize()
    judas_windows = [
        (day + pd.Timedelta(hours=7),  day + pd.Timedelta(hours=9)),
        (day + pd.Timedelta(hours=12), day + pd.Timedelta(hours=13, minutes=30)),
    ]

    for wstart, wend in judas_windows:
        if wstart >= until_time:
            continue
        wend_eff = min(wend, until_time)
        window = df_m1.loc[(df_m1.index >= wstart) & (df_m1.index <= wend_eff)]
        if window.empty:
            continue

        # Sweep du Asia high (Judas baissier = on monte puis on retombe)
        if (window["high"] > levels.asia_high).any():
            sweep_time = window[window["high"] > levels.asia_high].index[0]
            # Retournement : on est redescendu sous le asia_high avant until_time ?
            after_sweep = df_m1.loc[(df_m1.index > sweep_time) & (df_m1.index <= until_time)]
            if not after_sweep.empty and (after_sweep["close"] < levels.asia_high).any():
                return JudasAnalysis(
                    True, "asia_high_sweep", sweep_time,
                    f"Judas haut @ {sweep_time.strftime('%H:%M')} ({levels.asia_high:.2f})",
                )

        # Sweep du Asia low (Judas haussier)
        if (window["low"] < levels.asia_low).any():
            sweep_time = window[window["low"] < levels.asia_low].index[0]
            after_sweep = df_m1.loc[(df_m1.index > sweep_time) & (df_m1.index <= until_time)]
            if not after_sweep.empty and (after_sweep["close"] > levels.asia_low).any():
                return JudasAnalysis(
                    True, "asia_low_sweep", sweep_time,
                    f"Judas bas @ {sweep_time.strftime('%H:%M')} ({levels.asia_low:.2f})",
                )

    return JudasAnalysis(False, "none", None, "Pas de Judas detecte")
