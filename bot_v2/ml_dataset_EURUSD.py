"""Build dataset ML EURUSD M1 — Phase 6 (forex), meme methodologie que les autres actifs.

Decision user 2026-05-17 : test forex apres validation Phase 1-5 (XAU/NAS/GER/BTC).
EURUSD etait desactive en 2026-05-15 (WR catastrophique en backtest), mais avec :
- Fix SL (cause majeure des fausses pertes corrigee)
- OB+MSS confirmes (plus selectif)
- Daily bias en penalite (pas rejet)
... il y a une vraie chance que ca marche cette fois.
"""
from __future__ import annotations

import sys
sys.path.insert(0, 'c:/Users/Shadow/TradingBot')

import pandas as pd

from bot_v2.data_loader import load
from bot_v2.ml_dataset import build_dataset


def main():
    instruments = ["EURUSD"]
    df = load("EURUSD", "M1")
    earliest_end = df.index[-1]
    latest_start = earliest_end - pd.Timedelta(days=365 * 4)  # 4 ans
    print(f"Plage utilisee : {latest_start.date()} -> {earliest_end.date()} (4 ans)")
    print(f"Actif : EURUSD (Phase 6 - TF M1)")
    build_dataset(latest_start, earliest_end, instruments, chunk_months=3, ltf="M1")


if __name__ == "__main__":
    main()
