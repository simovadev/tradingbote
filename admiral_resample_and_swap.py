"""Resample Admiral M1 vers tous les TF, et remplace le cache Dukascopy.

Etapes :
1. Charge data_admiral/<asset>_M1.parquet
2. Resample vers M5, M15, M30, H1, H4, D1 (OHLCV propre)
3. Sauvegarde dans data_admiral/<asset>_<TF>.parquet
4. Backup data/cache/<asset>_<TF>.parquet -> data/cache_backup_dukascopy/
5. Remplace les caches par les versions Admiral

Usage : python admiral_resample_and_swap.py XAUUSD XAGUSD DXY
"""
import sys
sys.path.insert(0, 'c:/Users/Shadow/TradingBot')

import shutil
from pathlib import Path

import pandas as pd


ADMIRAL_DIR = Path("c:/Users/Shadow/TradingBot/data_admiral")
CACHE_DIR = Path("c:/Users/Shadow/TradingBot/data/cache")
BACKUP_DIR = Path("c:/Users/Shadow/TradingBot/data/cache_backup_dukascopy")
BACKUP_DIR.mkdir(exist_ok=True)

TF_RULES = {
    "M5":  "5min",
    "M15": "15min",
    "M30": "30min",
    "H1":  "1h",
    "H4":  "4h",
    "D1":  "1D",
}


def resample_m1(df_m1: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample M1 OHLCV vers un TF superieur."""
    agg = {
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }
    df = df_m1.resample(rule, label="left", closed="left").agg(agg)
    df = df.dropna(subset=["open"])  # supprime les periodes sans bougie (weekend)
    return df


def process_asset(asset: str):
    print(f"\n=== {asset} ===")
    m1_path = ADMIRAL_DIR / f"{asset}_M1.parquet"
    if not m1_path.exists():
        print(f"  SKIP : {m1_path} introuvable")
        return

    df_m1 = pd.read_parquet(m1_path)
    print(f"  M1 : {len(df_m1):>10,} bougies | {df_m1.index[0]} -> {df_m1.index[-1]}")

    # 1. Resample tous TF
    for tf, rule in TF_RULES.items():
        df_tf = resample_m1(df_m1, rule)
        out_path = ADMIRAL_DIR / f"{asset}_{tf}.parquet"
        df_tf.to_parquet(out_path)
        print(f"  {tf:3s}: {len(df_tf):>10,} bougies -> {out_path.name}")

    # 2. Backup et remplacer caches Dukascopy
    for tf in ["M1"] + list(TF_RULES.keys()):
        cache_path = CACHE_DIR / f"{asset}_{tf}.parquet"
        admiral_path = ADMIRAL_DIR / f"{asset}_{tf}.parquet"

        # Backup existant si pas deja fait
        if cache_path.exists():
            backup_path = BACKUP_DIR / f"{asset}_{tf}.parquet"
            if not backup_path.exists():
                shutil.copy2(cache_path, backup_path)
                print(f"  Backup Duka {tf} -> {backup_path.name}")

        # Copie Admiral vers cache
        shutil.copy2(admiral_path, cache_path)


def main():
    assets = sys.argv[1:] if len(sys.argv) > 1 else ["XAUUSD", "XAGUSD", "DXY"]
    print(f"Assets : {assets}")
    print(f"Admiral dir : {ADMIRAL_DIR}")
    print(f"Cache dir   : {CACHE_DIR}")
    print(f"Backup dir  : {BACKUP_DIR}")

    for a in assets:
        process_asset(a)

    print(f"\n=== TERMINE ===")
    print(f"Cache Dukascopy sauvegarde dans : {BACKUP_DIR}")
    print(f"Cache actuel pointe maintenant vers Admiral pour : {assets}")


if __name__ == "__main__":
    main()
