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

import os
import sys

# Auto-detect Windows vs Linux pour le chemin racine
ROOT = "c:/Users/Shadow/TradingBot" if sys.platform == "win32" else "/workspace/TradingBot"
sys.path.insert(0, ROOT)

from pathlib import Path

import pandas as pd

from bot_v2.data_loader import load
from bot_v2.ml_dataset import build_dataset


def main():
    instruments = ["XAUUSD"]
    df = load("XAUUSD", "M1")
    earliest_end = df.index[-1]
    earliest_start = df.index[0]

    # Chunks 1 mois si CPU >= 64 cores (ULTRA mode : 93 chunks parallel)
    # Sinon chunks 3 mois (31 chunks)
    cpu_count = os.cpu_count() or 4
    chunk_months = 1 if cpu_count >= 64 else 3

    print(f"=== ML DATASET XAUUSD V3.5 (filtres relaches) ===")
    print(f"Plage data : {earliest_start.date()} -> {earliest_end.date()}")
    print(f"Bougies M1 : {len(df):,}")
    print(f"Annees     : {(earliest_end - earliest_start).days / 365.25:.2f}")
    print(f"Cores      : {cpu_count} -> chunks de {chunk_months} mois")
    print()

    output_path = Path(f"{ROOT}/data/ml_dataset_XAUUSD_admiral_8ans_V3_5.parquet")
    build_dataset(
        earliest_start,
        earliest_end,
        instruments,
        output_path=output_path,
        chunk_months=chunk_months,
        ltf="M1",
    )


if __name__ == "__main__":
    main()
