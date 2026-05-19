"""Build dataset ML XAUUSD M1 sur 8 ans Admiral (2018-12 -> 2026-05).

Cache Dukascopy a ete swappe vers Admiral via admiral_resample_and_swap.py.
Couvre Covid 2020 + crypto crash 2022 + post-Covid -> regimes varies.

Output : data/ml_dataset_XAUUSD_admiral_8ans.parquet
"""
from __future__ import annotations

import sys
sys.path.insert(0, 'c:/Users/Shadow/TradingBot')

from pathlib import Path

import pandas as pd

from bot_v2.data_loader import load
from bot_v2.ml_dataset import build_dataset


def main():
    instruments = ["XAUUSD"]
    df = load("XAUUSD", "M1")
    earliest_end = df.index[-1]
    earliest_start = df.index[0]

    print(f"Plage data Admiral : {earliest_start.date()} -> {earliest_end.date()}")
    print(f"Total bougies M1   : {len(df):,}")
    print(f"Annees             : {(earliest_end - earliest_start).days / 365.25:.2f}")
    print(f"Actif              : XAUUSD (Admiral 8 ans)")
    print()

    output_path = Path("c:/Users/Shadow/TradingBot/data/ml_dataset_XAUUSD_admiral_8ans.parquet")
    build_dataset(
        earliest_start,
        earliest_end,
        instruments,
        output_path=output_path,
        chunk_months=3,
        ltf="M1",
    )


if __name__ == "__main__":
    main()
