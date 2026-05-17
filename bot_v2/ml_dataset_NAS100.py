"""Build dataset ML NAS100 M1 — Phase 2, meme methodologie que les autres actifs (4 ans).

Decision user 2026-05-17 : rebuild avec fix SL (sweep -> validation).
"""
from __future__ import annotations

import sys
sys.path.insert(0, 'c:/Users/Shadow/TradingBot')

import pandas as pd

from bot_v2.data_loader import load
from bot_v2.ml_dataset import build_dataset


def main():
    instruments = ["NAS100"]
    df = load("NAS100", "M1")
    earliest_end = df.index[-1]
    latest_start = earliest_end - pd.Timedelta(days=365 * 4)  # 4 ans (passe de 2 a 4 ans, user 2026-05-17)
    print(f"Plage utilisee : {latest_start.date()} -> {earliest_end.date()} (4 ans)")
    print(f"Actif : NAS100 (Phase 2 - TF M1)")
    build_dataset(latest_start, earliest_end, instruments, chunk_months=3, ltf="M1")


if __name__ == "__main__":
    main()
