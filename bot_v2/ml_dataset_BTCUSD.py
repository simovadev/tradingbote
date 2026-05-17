"""Build dataset ML BTCUSD M1 — Phase 5, meme methodologie que GER40/NAS100 (4 ans).

Decision user 2026-05-17 : BTC apres abandon USOUSD. Crypto 24/7, sera interessant
car comportement different (pas de killzones US strictes, volatilite haute).
"""
from __future__ import annotations

import sys
sys.path.insert(0, 'c:/Users/Shadow/TradingBot')

import pandas as pd

from bot_v2.data_loader import load
from bot_v2.ml_dataset import build_dataset


def main():
    instruments = ["BTCUSD"]
    df = load("BTCUSD", "M1")
    earliest_end = df.index[-1]
    latest_start = earliest_end - pd.Timedelta(days=365 * 4)  # 4 ans
    print(f"Plage utilisee : {latest_start.date()} -> {earliest_end.date()} (4 ans)")
    print(f"Actif : BTCUSD (Phase 5 - TF M1)")
    build_dataset(latest_start, earliest_end, instruments, chunk_months=3, ltf="M1")


if __name__ == "__main__":
    main()
