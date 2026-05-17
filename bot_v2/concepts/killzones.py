"""Killzones Vizion — heure NEW YORK avec gestion DST (bible §8).

Decision user 2026-05-15 :
- Pas de switch 8h30/9h30 selon news. NY AM = 9h30 NY fixe pour indices.
- Killzones standard : Asia, London, NY AM, NY Lunch, NY PM.

Reference bible :
- §8 Killzones (heure NY)
- §8.1 : Asia 20-00h NY, London 02-05h NY, NY AM 9h-11h NY (indices ouvrent 9h30),
         NY Lunch 12-13h NY, NY PM 14-16h NY
- §8.5 : Hors killzone = trade dechet (sauf if HTF tres clair)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from zoneinfo import ZoneInfo

import pandas as pd


NY_TZ = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class Killzone:
    """Une killzone definie en heure NY."""
    name: str
    start_hour: int       # heure debut en NY
    start_minute: int
    end_hour: int         # heure fin en NY (exclusive)
    end_minute: int
    # Si True, traverse minuit (ex. Asia 20h->00h)
    crosses_midnight: bool = False


# Bible §8 + decision user 2026-05-15 + ajustements analyse 2026-05-15 :
# - Plages elargies de 30min a 1h car les rejets "hors KZ" sur trades qui auraient WIN
#   etaient pile aux limites (08:40 NY, 13:20 NY, etc).
# - Ajout London Close (16-20h NY) qui n'etait pas couvert.
KILLZONES: list[Killzone] = [
    # Asia : 19h NY -> 01h NY (etendu, traverse minuit)
    Killzone("Asia",     19,  0,  1,  0, crosses_midnight=True),
    # London : 02h-06h NY (etendu, etait 02-05)
    Killzone("London",    2,  0,  6,  0),
    # NY AM : 08:00-12h NY (iter 38 : etendu 8h30 -> 8h00 pour gagner pre-market US).
    Killzone("NY_AM",     8,  0, 12,  0),
    # NY Lunch : 12h-13h30 NY
    Killzone("NY_Lunch", 12,  0, 13, 30),
    # NY PM : 13h30-16h NY (vers la cloture cash, etendu)
    Killzone("NY_PM",    13, 30, 16,  0),
    # London Close : 16h-20h NY (bible §8 mentionne 17h-20h, on prend 16h pour overlap)
    Killzone("London_Close", 16, 0, 20, 0),
]


def to_ny_time(ts: pd.Timestamp | datetime) -> datetime:
    """Convertit un timestamp (UTC ou naive considere UTC) en heure NY (DST auto)."""
    if isinstance(ts, pd.Timestamp):
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        return ts.tz_convert(NY_TZ).to_pydatetime()
    # datetime
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=ZoneInfo("UTC"))
    return ts.astimezone(NY_TZ)


def _in_window(t: time, start: time, end: time, crosses_midnight: bool) -> bool:
    if crosses_midnight:
        # Ex. start=20:00 end=00:00 -> on est dedans si t >= start OR t < end
        return t >= start or t < end
    return start <= t < end


def killzone_at(ts: pd.Timestamp | datetime) -> str | None:
    """Retourne le NOM de la killzone active a ce timestamp, ou None si aucune.

    Le timestamp peut etre UTC ; on le convertit en heure NY (DST gere).
    """
    ny = to_ny_time(ts)
    t = ny.time()
    for kz in KILLZONES:
        start = time(kz.start_hour, kz.start_minute)
        end = time(kz.end_hour, kz.end_minute)
        if _in_window(t, start, end, kz.crosses_midnight):
            return kz.name
    return None


def is_in_killzone(ts: pd.Timestamp | datetime) -> bool:
    return killzone_at(ts) is not None


def annotate_killzones(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute une colonne 'killzone' au DataFrame (nom ou None).

    Le DataFrame doit avoir un index datetime (UTC ou naive considere UTC).
    """
    if df.index.tz is None:
        idx_utc = df.index.tz_localize("UTC")
    else:
        idx_utc = df.index.tz_convert("UTC")
    # Conversion en NY pour tout l'index
    ny_idx = idx_utc.tz_convert(NY_TZ)
    times = ny_idx.time

    names: list[str | None] = []
    for t in times:
        match = None
        for kz in KILLZONES:
            start = time(kz.start_hour, kz.start_minute)
            end = time(kz.end_hour, kz.end_minute)
            if _in_window(t, start, end, kz.crosses_midnight):
                match = kz.name
                break
        names.append(match)

    out = df.copy()
    out["killzone"] = names
    return out


if __name__ == "__main__":
    # Test rapide : on prend XAUUSD M5 et on annote
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent))
    from bot_v2.data_loader import load

    df = load("XAUUSD", "M5")
    df = annotate_killzones(df)
    print(f"Total bougies : {len(df)}")
    print(f"En killzone   : {df['killzone'].notna().sum()}")
    print(f"Hors KZ       : {df['killzone'].isna().sum()}")
    print("\nRepartition par killzone :")
    print(df["killzone"].value_counts(dropna=False).to_string())
    print("\nExemples (5 dernieres bougies) :")
    print(df.tail(5)[["close", "killzone"]].to_string())
