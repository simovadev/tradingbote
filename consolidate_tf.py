"""Consolide les partials M5 et M15 en datasets finaux.

Comme on a un bug dans build_dataset (chunk_id sans LTF), les partials sont
sur disque mais le dataset final n'a pas ete genere. On consolide a la main.
"""
import sys
sys.path.insert(0, 'c:/Users/Shadow/TradingBot')

import pandas as pd
from pathlib import Path


def consolidate(tf: str):
    partial_dir = Path(f"c:/Users/Shadow/TradingBot/data/ml_partial_{tf}")
    if not partial_dir.exists():
        print(f"Pas de dossier {partial_dir}")
        return

    parts = sorted(partial_dir.glob("*.parquet"))
    print(f"\n=== {tf} : {len(parts)} partials trouves ===")

    all_rows = []
    for p in parts:
        df = pd.read_parquet(p)
        n = len(df)
        all_rows.extend(df.to_dict("records"))
        print(f"  {p.name} : {n} rows")

    if not all_rows:
        print("Aucune row a sauver.")
        return

    df_all = pd.DataFrame(all_rows)
    print(f"\nTotal : {len(df_all)} rows")
    print(f"Distribution outcomes :")
    print(df_all["outcome"].value_counts().to_string())

    closed = df_all[df_all["outcome"].isin(["WIN", "LOSS"])]
    if len(closed) > 0:
        wr = (closed["outcome"] == "WIN").mean() * 100
        print(f"WR brut : {wr:.1f}% sur {len(closed)} trades fermes")

    out_path = Path(f"c:/Users/Shadow/TradingBot/data/ml_dataset_{tf}.parquet")
    df_all.to_parquet(out_path)
    print(f"Sauve : {out_path}")


if __name__ == "__main__":
    consolidate("M5")
    consolidate("M15")
