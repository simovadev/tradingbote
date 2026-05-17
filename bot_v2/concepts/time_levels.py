"""Niveaux Time+Price — bible V2 §6 (video 05).

Distinguer 2 niveaux Time+Price intraday souvent confondus :

- **Daily Open NY** = 18h NY (= 00h Paris hiver) : ouverture officielle bougie daily.
  Sert de reference PO3 daily, separe les sessions daily.

- **Open Midnight NY** = 00h NY (= 06h Paris hiver) : pivot intraday.
  Niveau de retracement minimal attendu en debut de journee EU.
  Cassure en dessous = biais affaibli pour un long.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from zoneinfo import ZoneInfo

import pandas as pd

from bot_v2.concepts.killzones import NY_TZ, to_ny_time


@dataclass(frozen=True)
class TimeLevel:
    """Un niveau Time+Price reference (Daily Open ou Open Midnight)."""
    name: str                    # "DailyOpenNY" | "OpenMidnightNY"
    timestamp: pd.Timestamp      # timestamp UTC du tick d'ouverture
    price: float                 # prix d'ouverture (open de la bougie correspondante)


def _find_open_at_ny_hour(
    df: pd.DataFrame,
    target_ts: pd.Timestamp,
    ny_hour: int,
    ny_minute: int = 0,
) -> TimeLevel | None:
    """Trouve la derniere bougie qui correspond a `ny_hour:ny_minute` NY avant target_ts.

    Retourne le `open` de cette bougie comme niveau de reference.
    """
    if df.index.tz is None:
        idx_utc = df.index.tz_localize("UTC")
    else:
        idx_utc = df.index.tz_convert("UTC")

    # Filtre les bougies <= target_ts
    if hasattr(target_ts, "tz") and target_ts.tz is None:
        target_ts_utc = target_ts.tz_localize("UTC")
    else:
        target_ts_utc = target_ts

    # Convertit index en NY pour matcher l'heure
    ny_idx = idx_utc.tz_convert(NY_TZ)
    target_time = time(ny_hour, ny_minute)

    # Cherche la bougie la plus recente <= target_ts et dont l'heure NY est >= target_time
    # On veut la bougie qui contient la transition (ex. pour 00h NY, on prend la bougie
    # qui commence a 00h NY ou la 1ere apres).
    mask = (idx_utc <= target_ts_utc)
    if not mask.any():
        return None
    df_before = df[mask]
    ny_before = ny_idx[mask]

    # On cherche la derniere bougie dont l'heure NY est EXACTEMENT target_time (ou la plus proche apres)
    for i in range(len(df_before) - 1, -1, -1):
        t = ny_before[i].time()
        if t == target_time:
            return TimeLevel(
                name="ny_open",
                timestamp=df_before.index[i],
                price=float(df_before.iloc[i]["open"]),
            )

    return None


def get_daily_open_ny(df_ltf: pd.DataFrame, target_ts: pd.Timestamp) -> TimeLevel | None:
    """Daily Open NY = 18h NY (ouverture officielle bougie daily ICT)."""
    lvl = _find_open_at_ny_hour(df_ltf, target_ts, 18, 0)
    if lvl is None:
        return None
    return TimeLevel(name="DailyOpenNY", timestamp=lvl.timestamp, price=lvl.price)


def get_open_midnight_ny(df_ltf: pd.DataFrame, target_ts: pd.Timestamp) -> TimeLevel | None:
    """Open Midnight NY = 00h NY (pivot intraday Vizion)."""
    lvl = _find_open_at_ny_hour(df_ltf, target_ts, 0, 0)
    if lvl is None:
        return None
    return TimeLevel(name="OpenMidnightNY", timestamp=lvl.timestamp, price=lvl.price)


def check_above_open_midnight(
    df_ltf: pd.DataFrame,
    ob_price: float,
    target_ts: pd.Timestamp,
    direction: str,
) -> tuple[bool, float | None]:
    """Verifie le rapport de l'OB par rapport a Open Midnight NY.

    Bible V2 §6 : pour un trade BULLISH, le prix ne devrait PAS etre descendu
    en dessous d'Open Midnight NY (sinon biais affaibli).

    Returns:
        (is_ok, open_midnight_price)
    """
    omn = get_open_midnight_ny(df_ltf, target_ts)
    if omn is None:
        return True, None  # pas de donnees -> on laisse passer

    if direction == "bullish":
        return ob_price >= omn.price, omn.price
    else:
        return ob_price <= omn.price, omn.price


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent))
    from bot_v2.data_loader import load

    df = load("NAS100", "M5")
    last_ts = df.index[-1]
    daily = get_daily_open_ny(df, last_ts)
    midnight = get_open_midnight_ny(df, last_ts)
    last_price = float(df["close"].iloc[-1])

    print(f"NAS100 derniere bougie : {last_ts} @ {last_price:.2f}")
    if daily:
        print(f"  Daily Open NY (18h NY)  : {daily.timestamp} @ {daily.price:.2f}")
    if midnight:
        print(f"  Open Midnight NY (00h)  : {midnight.timestamp} @ {midnight.price:.2f}")
        delta = last_price - midnight.price
        print(f"  Price vs Open Midnight  : {delta:+.2f}")
