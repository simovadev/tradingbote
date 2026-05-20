"""Rebuild V5 sur 3 mois (decembre 2025 - fevrier 2026) pour 1 actif.

But : avoir un modele V5 entraine sur la MEME fenetre temporelle que ce que le live voit.
Si rebuild 3 mois + entrainement -> test sur 1 jour en fev = beaucoup de trades a 0.75
alors le bug actuel = juste fenetre live trop courte vs train.

Usage:
    python rebuild_v5_3mois.py XAUUSD
"""
import os
import sys
import time
import argparse
from pathlib import Path

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from bot_v2.data_loader import load
from bot_v2.ml_dataset import build_dataset

# Periode rebuild : 3 mois (decembre 2025 - fevrier 2026)
START_TS = pd.Timestamp("2025-12-01", tz="UTC")
END_TS = pd.Timestamp("2026-02-28", tz="UTC")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("asset")
    args = p.parse_args()

    asset = args.asset
    print(f"\n=== REBUILD V5 3 MOIS : {asset} ===", flush=True)
    print(f"Periode : {START_TS.date()} -> {END_TS.date()}", flush=True)

    # Verifie data dispo
    df = load(asset, "M1")
    print(f"Data M1 dispo : {df.index[0].date()} -> {df.index[-1].date()} ({len(df):,} bougies)", flush=True)

    if df.index[0] > START_TS:
        print(f"!! Data commence apres {START_TS.date()}, ajuste START_TS", flush=True)
        return
    if df.index[-1] < END_TS:
        print(f"!! Data finit avant {END_TS.date()}, ajuste END_TS", flush=True)
        return

    output_path = Path(f"{ROOT}/data/ml_dataset_{asset}_admiral_3mois_V5.parquet")

    t0 = time.time()
    cpu_count = os.cpu_count() or 4
    # 3 mois = 1 chunk OK
    df_result = build_dataset(
        START_TS, END_TS,
        [asset],
        output_path=output_path,
        chunk_months=1,  # chunks 1 mois -> 3 chunks parallele
        ltf="M1",
        version_suffix="_V5_3mois",  # dossier partial separe
    )
    print(f"\nDuree : {time.time()-t0:.0f}s", flush=True)

    if df_result is not None and len(df_result) > 0:
        print(f"\n>>> RECAP {asset} V5 3 mois : {len(df_result):,} candidats", flush=True)
        if "outcome" in df_result.columns:
            closed = df_result[df_result["outcome"].isin(["WIN", "LOSS"])]
            wr = (closed["outcome"] == "WIN").mean() * 100 if len(closed) > 0 else 0
            print(f">>> WR brut : {wr:.1f}% sur {len(closed):,} fermes", flush=True)
            print(f">>> Output : {output_path.name}", flush=True)


if __name__ == "__main__":
    main()
