"""Build dataset ML GER40 M1 — Phase 3, meme methodologie que NAS100 Phase 2.

Decision user 2026-05-17 : GER40 (DAX 40) ajoute au portefeuille apres XAUUSD et NAS100.
"""
from __future__ import annotations

import sys
sys.path.insert(0, 'c:/Users/Shadow/TradingBot')

import pandas as pd

from bot_v2.data_loader import load
from bot_v2.ml_dataset import build_dataset


def main():
    instruments = ["GER40"]
    df = load("GER40", "M1")
    earliest_end = df.index[-1]
    latest_start = earliest_end - pd.Timedelta(days=365 * 4)  # 4 ans (user 2026-05-17)
    print(f"Plage utilisee : {latest_start.date()} -> {earliest_end.date()} (4 ans)")
    print(f"Actif : GER40 (Phase 3 - TF M1)")
    build_dataset(latest_start, earliest_end, instruments, chunk_months=3, ltf="M1")


if __name__ == "__main__":
    main()
