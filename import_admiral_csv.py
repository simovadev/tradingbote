"""Importe les CSV Admiral MT5 (export 8 ans) en parquet propre.

Format CSV Admiral : tab-separated
    <DATE>  <TIME>  <OPEN>  <HIGH>  <LOW>  <CLOSE>  <TICKVOL>  <VOL>  <SPREAD>

Output : data_admiral/<actif>_M1.parquet (index UTC, OHLCV)

Usage :
    python import_admiral_csv.py <chemin_csv> <nom_actif>
    python import_admiral_csv.py "C:/Users/Shadow/Downloads/GOLD_M1_201812030100_202605191158.csv" XAUUSD
"""
import sys
from pathlib import Path

import pandas as pd


OUTPUT_DIR = Path("c:/Users/Shadow/TradingBot/data_admiral")
OUTPUT_DIR.mkdir(exist_ok=True)


def import_csv(csv_path: Path, asset: str) -> pd.DataFrame:
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


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    asset = sys.argv[2].upper()

    if not csv_path.exists():
        print(f"ERREUR : {csv_path} introuvable")
        sys.exit(1)

    size_mb = csv_path.stat().st_size / 1024 / 1024
    print(f"=== IMPORT ADMIRAL {asset} ===")
    print(f"Source : {csv_path.name} ({size_mb:.1f} MB)")
    print("Loading...")

    df = import_csv(csv_path, asset)

    first = df.index[0]
    last = df.index[-1]
    days = (last - first).days
    years = days / 365.25

    out_path = OUTPUT_DIR / f"{asset}_M1.parquet"
    df.to_parquet(out_path)
    out_mb = out_path.stat().st_size / 1024 / 1024

    print(f"\nBougies   : {len(df):>10,}")
    print(f"Periode   : {first} -> {last}")
    print(f"Duree     : {days} jours ({years:.2f} ans)")
    print(f"Output    : {out_path} ({out_mb:.1f} MB)")

    # Stats qualite
    print(f"\nQuality check :")
    print(f"  NaN open/close : {df['open'].isna().sum()} / {df['close'].isna().sum()}")
    print(f"  Volume==0      : {(df['volume'] == 0).sum()}")

    # Detection gros gaps (>1h en weekday)
    gaps = df.index.to_series().diff()
    big_gaps = gaps[gaps > pd.Timedelta(hours=4)]
    print(f"  Gaps > 4h      : {len(big_gaps)} (normal pour weekends + holidays)")


if __name__ == "__main__":
    main()
