"""v22_fetch_recent.py - Fetch les dernieres bougies M5/H1/D1 depuis MT5.

Met a jour data_vantage/ avec les bougies manquantes pour les 6 actifs.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timezone

DATA = ROOT / "data_vantage"

SYMBOL_MAP = {
    "EURUSD": "EURUSD+",
    "XAUUSD": "XAUUSD+",
    "NAS100": "NAS100",
    "GER40":  "GER40",
    "BTCUSD": "BTCUSD",
    "CL-OIL": "USOUSD",
}

TF_MAP = {
    "M5": mt5.TIMEFRAME_M5,
    "H1": mt5.TIMEFRAME_H1,
    "D1": mt5.TIMEFRAME_D1,
}


def fetch_and_merge(our_name, mt5_name, tf_str, mt5_tf):
    path = DATA / f"{our_name}_{tf_str}.parquet"
    df_old = pd.read_parquet(path)
    last_ts = df_old.index[-1]
    # Fetch a partir de last_ts (en utc)
    # On prend large pour etre sur d'avoir tout
    n = 50000   # max ~6 mois en M5
    rates = mt5.copy_rates_from(mt5_name, mt5_tf, datetime.now(timezone.utc), n)
    if rates is None or len(rates) == 0:
        print(f"  {our_name} {tf_str} : pas de rates")
        return
    df_new = pd.DataFrame(rates)
    df_new["time"] = pd.to_datetime(df_new["time"], unit="s", utc=True)
    df_new = df_new.set_index("time")
    df_new = df_new.rename(columns={"tick_volume": "volume"})
    # On filtre uniquement les bougies plus recentes que last_ts
    df_new = df_new[df_new.index > last_ts]
    if len(df_new) == 0:
        print(f"  {our_name} {tf_str} : deja a jour (last={last_ts})")
        return
    # Concat et sauvegarde
    cols = ["open", "high", "low", "close", "volume"]
    df_new = df_new[cols] if all(c in df_new.columns for c in cols) else df_new
    df_combined = pd.concat([df_old, df_new])
    df_combined = df_combined[~df_combined.index.duplicated(keep="last")]
    df_combined = df_combined.sort_index()
    df_combined.to_parquet(path)
    print(f"  {our_name} {tf_str} : +{len(df_new)} bougies, last = {df_combined.index[-1]}")


def main():
    if not mt5.initialize():
        print(f"MT5 fail : {mt5.last_error()}"); return
    info = mt5.account_info()
    print(f"MT5 OK : {info.login} {info.server}\n")

    for our_name, mt5_name in SYMBOL_MAP.items():
        print(f"=== {our_name} ({mt5_name}) ===")
        for tf_str, mt5_tf in TF_MAP.items():
            try:
                fetch_and_merge(our_name, mt5_name, tf_str, mt5_tf)
            except Exception as e:
                print(f"  ERR {our_name} {tf_str} : {e}")
        print()

    mt5.shutdown()


if __name__ == "__main__":
    main()
