"""Build dataset ML USDCAD M1 - Phase 4 (user 2026-05-18)."""
from __future__ import annotations

import sys
sys.path.insert(0, 'c:/Users/Shadow/TradingBot')

import pandas as pd

from bot_v2.data_loader import load
from bot_v2.ml_dataset import build_dataset


def main():
    instruments = ["USDCAD"]
    df = load("USDCAD", "M1")
    earliest_end = df.index[-1]
    latest_start = earliest_end - pd.Timedelta(days=365 * 4)
    print(f"Plage utilisee : {latest_start.date()} -> {earliest_end.date()} (4 ans)")
    print(f"Actif : USDCAD (Phase 4 - TF M1)")
    build_dataset(latest_start, earliest_end, instruments, chunk_months=3, ltf="M1")


if __name__ == "__main__":
    main()
