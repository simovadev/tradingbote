"""Build dataset ML XAUUSD V3 - Admiral 8 ans, features nettoyees + enrichies.

V3 changes (2026-05-19) :
- Fix bug multi_liquidity_sweep (setup_quality.py:131-140)
- Suppression 7 features constantes (multi_liq_sweep, kz_none, has_sync_fvg,
  has_grandparent, has_phase_reversal, is_mss_setup, has_mss_confirmation)
- Ajout features temps : hour_of_day, day_of_week, minutes_into_killzone
- Ajout features volatilite : atr_at_setup, atr_ratio_100
- Ajout features distance : dist_to_pdh_pct, dist_to_pdl_pct, dist_to_d1_open_pct

Output : data/ml_dataset_XAUUSD_admiral_8ans_V3.parquet
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

    print(f"=== ML DATASET XAUUSD V3 (Admiral 8 ans, features enrichies) ===")
    print(f"Plage data : {earliest_start.date()} -> {earliest_end.date()}")
    print(f"Bougies M1 : {len(df):,}")
    print(f"Annees     : {(earliest_end - earliest_start).days / 365.25:.2f}")
    print()

    output_path = Path("c:/Users/Shadow/TradingBot/data/ml_dataset_XAUUSD_admiral_8ans_V3.parquet")
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
