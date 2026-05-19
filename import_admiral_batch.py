"""Import batch de tous les CSV Admiral dans data_admiral/ + resample TF + swap cache.

Reuse import_admiral_csv + admiral_resample_and_swap en batch pour 12 actifs.
"""
import sys
sys.path.insert(0, 'c:/Users/Shadow/TradingBot')

import shutil
import time
from pathlib import Path

import pandas as pd


DOWNLOADS = Path("C:/Users/Shadow/Downloads/Nouveau dossier")
ADMIRAL_DIR = Path("c:/Users/Shadow/TradingBot/data_admiral")
CACHE_DIR = Path("c:/Users/Shadow/TradingBot/data/cache")
BACKUP_DIR = Path("c:/Users/Shadow/TradingBot/data/cache_backup_dukascopy")

ADMIRAL_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
BACKUP_DIR.mkdir(exist_ok=True)


# Mapping nom CSV Admiral -> notre nom interne
MAPPING = {
    "AUDUSD": "AUDUSD",
    "BTCUSD": "BTCUSD",
    "EURUSD": "EURUSD",
    "GBPUSD": "GBPUSD",
    "USDCAD": "USDCAD",
    "USDCHF": "USDCHF",
    "USDJPY": "USDJPY",
    "GERMANY40": "GER40",
    "US100": "NAS100",
    "[CAC40]": "FRA40",
    "[DJI30]": "DJ30",
    "[FTSE100]": "UK100",
    "[SP500]": "SP500",
}


TF_RULES = {
    "M5":  "5min",
    "M15": "15min",
    "M30": "30min",
    "H1":  "1h",
    "H4":  "4h",
    "D1":  "1D",
}


def find_csv(admiral_name: str) -> Path | None:
    """Trouve le CSV correspondant a un nom Admiral dans Downloads.
    Note : iterdir au lieu de glob car les crochets [] sont des wildcards en glob.
    """
    for f in DOWNLOADS.iterdir():
        if f.is_file() and f.name.startswith(f"{admiral_name}_M1_") and f.suffix == ".csv":
            return f
    return None


def import_csv(csv_path: Path) -> pd.DataFrame:
    """Lit un CSV Admiral et retourne un DataFrame OHLCV UTC."""
    df = pd.read_csv(csv_path, sep="\t")
    df.columns = [c.strip("<>").lower() for c in df.columns]
    df["dt"] = pd.to_datetime(
        df["date"] + " " + df["time"],
        format="%Y.%m.%d %H:%M:%S",
        utc=True,
    )
    df = df.set_index("dt")
    df = df[["open", "high", "low", "close", "tickvol"]].rename(columns={"tickvol": "volume"})
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df


def resample_m1(df_m1: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample M1 OHLCV vers TF superieur."""
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    df = df_m1.resample(rule, label="left", closed="left").agg(agg)
    return df.dropna(subset=["open"])


def process(admiral_name: str, our_name: str) -> bool:
    csv_path = find_csv(admiral_name)
    if csv_path is None:
        print(f"  [{our_name:8s}] CSV introuvable pour {admiral_name}")
        return False

    t0 = time.time()
    size_mb = csv_path.stat().st_size / 1024 / 1024
    print(f"\n=== {our_name} (depuis {admiral_name}, {size_mb:.0f} MB) ===")

    # 1. Load CSV
    print(f"  [1/4] Loading CSV...")
    df_m1 = import_csv(csv_path)
    print(f"        {len(df_m1):,} bougies | {df_m1.index[0]} -> {df_m1.index[-1]}")

    # 2. Save M1 to data_admiral
    print(f"  [2/4] Save M1 to data_admiral...")
    m1_path = ADMIRAL_DIR / f"{our_name}_M1.parquet"
    df_m1.to_parquet(m1_path)

    # 3. Resample to all TFs
    print(f"  [3/4] Resample to all TFs...")
    for tf, rule in TF_RULES.items():
        df_tf = resample_m1(df_m1, rule)
        out_path = ADMIRAL_DIR / f"{our_name}_{tf}.parquet"
        df_tf.to_parquet(out_path)

    # 4. Backup old cache + copy admiral -> cache
    print(f"  [4/4] Swap cache (backup Duka first)...")
    for tf in ["M1"] + list(TF_RULES.keys()):
        cache_path = CACHE_DIR / f"{our_name}_{tf}.parquet"
        admiral_path = ADMIRAL_DIR / f"{our_name}_{tf}.parquet"
        if cache_path.exists():
            backup_path = BACKUP_DIR / f"{our_name}_{tf}.parquet"
            if not backup_path.exists():
                shutil.copy2(cache_path, backup_path)
        shutil.copy2(admiral_path, cache_path)

    elapsed = time.time() - t0
    print(f"  OK : {our_name} traite en {elapsed:.0f}s")
    return True


def main():
    print(f"=== IMPORT BATCH ADMIRAL ({len(MAPPING)} actifs) ===")
    print(f"Source : {DOWNLOADS}")
    print(f"Output : {ADMIRAL_DIR} + cache swap")

    ok = 0
    fail = 0
    t_start = time.time()
    for admiral_name, our_name in MAPPING.items():
        try:
            if process(admiral_name, our_name):
                ok += 1
            else:
                fail += 1
        except Exception as e:
            print(f"  ERREUR {our_name}: {e}")
            fail += 1

    elapsed = time.time() - t_start
    print(f"\n=== TERMINE en {elapsed:.0f}s ===")
    print(f"  OK    : {ok}/{len(MAPPING)}")
    print(f"  Echec : {fail}/{len(MAPPING)}")


if __name__ == "__main__":
    main()
