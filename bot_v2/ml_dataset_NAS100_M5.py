"""Build dataset ML NAS100 M5 — meme methodologie que NAS100 M1 v7.

Decision user 2026-05-16 : completer Phase 2 NAS100 avec un modele M5 specifique
(au lieu d'utiliser le ml_model_M5 generique entraine sur XAUUSD).
"""
from __future__ import annotations

import sys
sys.path.insert(0, 'c:/Users/Shadow/TradingBot')

import pandas as pd

from bot_v2.data_loader import load
from bot_v2.ml_dataset import build_dataset


def main():
    instruments = ["NAS100"]
    df = load("NAS100", "M5")
    earliest_end = df.index[-1]
    latest_start = earliest_end - pd.Timedelta(days=365 * 2)  # 2 ans
    print(f"Plage utilisee : {latest_start.date()} -> {earliest_end.date()} (2 ans)")
    print(f"Actif : NAS100 (Phase 2 - TF M5)")
    # chunk_months=6 sur M5 (12x moins de bougies que M1, on peut plus large)
    build_dataset(latest_start, earliest_end, instruments, chunk_months=6, ltf="M5")


if __name__ == "__main__":
    main()
