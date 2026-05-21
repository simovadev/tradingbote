"""Build dataset ML V7 sur donnees Vantage 8 ans (data_vantage/).

V7 vs V5/V6 :
- Meme code/features/filtres que V5
- SOURCE DE DONNEES : data_vantage/ (Vantage broker, 8 ans 2018-2026)
- Necessaire car le live tourne sur Vantage : prix divergent ~19 USD/bougie XAUUSD
  vs Admiral, le ML V5 voit des patterns differents en live -> rejette tout.
- Split train/VAL/OOS pour validation rigoureuse :
    Training : 2018-03 -> 2025-05 (~7 ans)
    Val      : 2025-05 -> 2025-11 (~6 mois)
    OOS      : 2025-11 -> 2026-05 (~6 mois, JAMAIS vu en training)

Usage :
    python -m bot_v2.build_v7_vantage_dataset XAUUSD
    python -m bot_v2.build_v7_vantage_dataset --all
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# IMPORTANT : forcer le data loader a pointer vers data_vantage/ AVANT
# d'importer ml_dataset (qui utilise data_loader.load au module top-level).
os.environ["BUILD_DATA_DIR"] = "data_vantage"

ROOT = "c:/Users/Shadow/TradingBot" if sys.platform == "win32" else "/workspace/TradingBot"
sys.path.insert(0, ROOT)

import pandas as pd

from bot_v2.data_loader import load
from bot_v2.ml_dataset import build_dataset


ALL_ASSETS = [
    "XAUUSD", "NAS100", "GER40", "BTCUSD",
    "EURUSD", "GBPUSD", "AUDUSD", "USDJPY",
    "SP500", "DJ30", "UK100", "FRA40",
    "USDCAD", "USDCHF",
]

# V7 (2026-05-21) : data Vantage 8 ans (2018-03 -> 2026-05).
# Le split TRAIN/VAL/OOS se fait dans train_v7_vantage.py :
#   TRAIN 7 ans / VAL 6 mois / OOS 6 mois
TRAIN_START = pd.Timestamp("2018-03-01", tz="UTC")
TRAIN_END = pd.Timestamp("2026-05-21", tz="UTC")


def build_one(asset: str):
    df = load(asset, "M1")
    earliest_start = df.index[0]
    earliest_end = df.index[-1]
    print(f"\n=== ML DATASET {asset} V7 (Vantage 8 ans training) ===", flush=True)
    print(f"Source data    : data_vantage/{asset}_M1.parquet", flush=True)
    print(f"Plage data dispo : {earliest_start.date()} -> {earliest_end.date()}", flush=True)
    print(f"Bougies M1 dispo : {len(df):,}", flush=True)

    # Determine la fenetre training a utiliser
    train_start = max(TRAIN_START, earliest_start)
    train_end = min(TRAIN_END, earliest_end)
    print(f"Fenetre training : {train_start.date()} -> {train_end.date()}", flush=True)
    n_days_training = (train_end - train_start).days
    print(f"Jours training : {n_days_training} jours (~{n_days_training/30:.1f} mois)", flush=True)

    cpu_count = os.cpu_count() or 4
    # Sur Vast.ai EPYC 96 cores : chunks de ~14 jours pour saturer
    # Sur PC local 4 cores : chunks de ~1 mois suffisent (5 chunks total)
    if cpu_count >= 128:
        chunk_months = 0.5
    elif cpu_count >= 64:
        chunk_months = 1
    else:
        chunk_months = 1.5
    print(f"Cores : {cpu_count} -> chunks de {chunk_months} mois", flush=True)

    output_path = Path(f"{ROOT}/data/ml_dataset_{asset}_vantage_v7.parquet")
    df_result = build_dataset(
        train_start,
        train_end,
        [asset],
        output_path=output_path,
        chunk_months=chunk_months,
        ltf="M1",
        version_suffix="_V7_VANTAGE",  # dossier separe des chunks V5/V6
    )

    print(f"\n>>> RECAP {asset} V7 : {len(df_result):,} candidats totaux", flush=True)
    if len(df_result) > 0 and "outcome" in df_result.columns:
        closed = df_result[df_result["outcome"].isin(["WIN", "LOSS"])]
        if len(closed) > 0:
            wr = (closed["outcome"] == "WIN").mean() * 100
            print(f">>> RECAP {asset} V7 : WR brut {wr:.1f}% sur {len(closed)} fermes",
                  flush=True)
    print("=" * 60, flush=True)


def main():
    p = argparse.ArgumentParser(description="Build V7 ML dataset sur donnees Vantage 8 ans")
    p.add_argument("asset", nargs="?", help="Asset (XAUUSD) ou --all")
    p.add_argument("--all", action="store_true", help="Build tous les 14 actifs")
    args = p.parse_args()

    print(f"BUILD_DATA_DIR forced to : {os.environ.get('BUILD_DATA_DIR')}", flush=True)

    if args.all or args.asset == "--all":
        for a in ALL_ASSETS:
            try:
                build_one(a)
            except Exception as e:
                print(f"!! FAIL {a} : {e}", flush=True)
                import traceback
                traceback.print_exc()
                continue
    elif args.asset:
        if args.asset not in ALL_ASSETS:
            print(f"Actif inconnu : {args.asset}. Choix : {ALL_ASSETS}")
            sys.exit(1)
        build_one(args.asset)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
