"""Build dataset ML XAUUSD V3.5 - Admiral 8 ans, filtres relaches.

V3.5 changes vs V3 (2026-05-19) :
- 5 filtres eliminatoires passes en BONUS/MALUS :
  * Session sans direction : -8 (au lieu de REJET)
  * Parent OB HTF absent : -15 (au lieu de REJET)
  * Grand-parent OB absent : -5 (au lieu de REJET)
  * Mauvaise zone P/D Fibo : -10 (au lieu de REJET)
  * FVG sync absent : -5 (au lieu de REJET)
- +5 nouvelles features pour que le ML apprenne :
  has_FVG_sync, has_parent_ob, has_grandparent_ob, has_good_zone, has_session_direction

Objectif : 2-4x plus de setups, le ML apprend les vraies conditions gagnantes.

Output : data/ml_dataset_XAUUSD_admiral_8ans_V3_5.parquet
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

    print(f"=== ML DATASET XAUUSD V3.5 (filtres relaches) ===")
    print(f"Plage data : {earliest_start.date()} -> {earliest_end.date()}")
    print(f"Bougies M1 : {len(df):,}")
    print(f"Annees     : {(earliest_end - earliest_start).days / 365.25:.2f}")
    print()

    output_path = Path("c:/Users/Shadow/TradingBot/data/ml_dataset_XAUUSD_admiral_8ans_V3_5.parquet")
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
