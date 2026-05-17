"""Fetch des bougies pour bot_v2 — utilise la config bot_v2 (incl. Forex).

Usage :
    python -m bot_v2.fetch_data                              # tous les actifs primaires, 30j
    python -m bot_v2.fetch_data --instrument EURUSD          # un seul actif
    python -m bot_v2.fetch_data --instrument EURUSD --tf M1  # un seul TF
    python -m bot_v2.fetch_data --days 90                    # 3 mois
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

import pandas as pd

from bot_v2.config import ALL_TF, DATA_DIR, INSTRUMENTS
from bot_v2.data_loader import cache_path


def _get_duka_instrument(instrument: str):
    import dukascopy_python.instruments as di
    code = INSTRUMENTS[instrument]["duka"]
    return getattr(di, code)


def fetch_dukascopy(instrument: str, tf: str, start: datetime, end: datetime) -> pd.DataFrame:
    import dukascopy_python

    interval_map = {
        "M1":  dukascopy_python.INTERVAL_MIN_1,
        "M5":  dukascopy_python.INTERVAL_MIN_5,
        "M15": dukascopy_python.INTERVAL_MIN_15,
        "H1":  dukascopy_python.INTERVAL_HOUR_1,
        "H4":  dukascopy_python.INTERVAL_HOUR_4,
        "D1":  dukascopy_python.INTERVAL_DAY_1,
    }
    if tf not in interval_map:
        raise ValueError(f"TF inconnu : {tf} (attendu : {list(interval_map.keys())})")

    df = dukascopy_python.fetch(
        _get_duka_instrument(instrument),
        interval_map[tf],
        dukascopy_python.OFFER_SIDE_BID,
        start,
        end,
    )

    if df is None or df.empty:
        raise RuntimeError(f"Dukascopy vide pour {instrument} {tf} {start} -> {end}")

    df = df.rename(columns=str.lower)
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()


def fetch_and_cache(instrument: str, tf: str, days: int) -> None:
    end = datetime.now(tz=timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=days)

    path = cache_path(instrument, tf)
    print(f"  [{instrument} / {tf}] Fetch {start.date()} -> {end.date()}...", end=" ", flush=True)
    try:
        df = fetch_dukascopy(instrument, tf, start, end)
    except Exception as e:
        print(f"ERREUR: {e}")
        return
    print(f"{len(df)} bougies -> {path.name}")
    df.to_parquet(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--instrument", type=str, default=None)
    ap.add_argument("--tf", type=str, default=None)
    ap.add_argument("--primary-only", action="store_true",
                    help="Ne fetche que les actifs primaires (skip smt_only)")
    args = ap.parse_args()

    if args.instrument:
        instruments = [args.instrument]
    elif args.primary_only:
        instruments = [k for k, v in INSTRUMENTS.items() if v["role"] == "primary"]
    else:
        instruments = list(INSTRUMENTS.keys())

    tfs = [args.tf] if args.tf else ALL_TF

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for inst in instruments:
        if inst not in INSTRUMENTS:
            print(f"INSTRUMENT INCONNU : {inst}")
            continue
        print(f"\n=== {inst} ({INSTRUMENTS[inst]['label']}) ===")
        for tf in tfs:
            fetch_and_cache(inst, tf, args.days)

    print(f"\nOK. Cache dans {DATA_DIR}")


if __name__ == "__main__":
    main()
