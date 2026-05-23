"""Exporte les ticks MT5 (bid/ask) d'une journee pour les 14 actifs.

Sortie : data_ticks/<ASSET>_ticks_<YYYYMMDD>.parquet
Colonnes : time_ns (epoch ns UTC reel), bid, ask

Ces parquets seront uploades sur Vast pour le backtest tick par tick
(Vast n'a pas MT5, donc on lui donne les ticks pre-exportes).

Usage :
    python export_ticks.py --date 2026-05-19
    python export_ticks.py --date 2026-05-19 --assets XAUUSD EURUSD
"""
from __future__ import annotations
import argparse
import datetime
import os
import sys
from pathlib import Path

sys.path.insert(0, "c:/Users/Shadow/TradingBot")

import numpy as np
import pandas as pd
import MetaTrader5 as mt5

from bot_v2.mt5_executor import MT5Executor, to_broker_symbol

LIVE_ASSETS = [
    "XAUUSD", "NAS100", "GER40", "BTCUSD",
    "EURUSD", "GBPUSD", "AUDUSD", "USDJPY",
    "SP500", "DJ30", "UK100", "FRA40",
    "USDCAD", "USDCHF",
]

OUT_DIR = Path("c:/Users/Shadow/TradingBot/data_ticks")
OUT_DIR.mkdir(exist_ok=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--date", required=True, help="YYYY-MM-DD (jour a exporter)")
    p.add_argument("--assets", nargs="+", default=LIVE_ASSETS)
    p.add_argument("--margin_h", type=int, default=2,
                   help="Marge en heures avant/apres pour fills tardifs (default 2)")
    args = p.parse_args()

    exe = MT5Executor()
    if not exe.initialize():
        print("MT5 INIT FAIL - ouvre MT5 et connecte-toi")
        sys.exit(1)
    off = exe.broker_utc_offset_sec
    print(f"MT5 OK, offset={off}s")

    day = pd.Timestamp(args.date, tz="UTC")
    # Marge : -margin_h avant 00h, +margin_h apres 23h59
    start = (day - pd.Timedelta(hours=args.margin_h)).to_pydatetime()
    end = (day + pd.Timedelta(hours=24 + args.margin_h)).to_pydatetime()

    print(f"Export ticks {args.date} (avec marge {args.margin_h}h) pour {len(args.assets)} actifs\n")

    total_ticks = 0
    for asset in args.assets:
        broker = to_broker_symbol(asset)
        raw = mt5.copy_ticks_range(
            broker,
            start + datetime.timedelta(seconds=off),
            end + datetime.timedelta(seconds=off),
            mt5.COPY_TICKS_ALL,
        )
        if raw is None or len(raw) == 0:
            print(f"  {asset:8s} : AUCUN TICK ({mt5.last_error()})")
            continue
        tdf = pd.DataFrame(raw)
        # epoch ns en UTC reel (compense offset)
        time_ns = (tdf["time_msc"].values.astype(np.int64) * 1_000_000) - (off * 1_000_000_000)
        out_df = pd.DataFrame({
            "time_ns": time_ns,
            "bid": tdf["bid"].values.astype(np.float64),
            "ask": tdf["ask"].values.astype(np.float64),
        })
        date_str = args.date.replace("-", "")
        out_path = OUT_DIR / f"{asset}_ticks_{date_str}.parquet"
        out_df.to_parquet(out_path, index=False, compression="snappy")
        first = pd.Timestamp(time_ns[0], tz="UTC")
        last = pd.Timestamp(time_ns[-1], tz="UTC")
        sz = out_path.stat().st_size / 1024 / 1024
        print(f"  {asset:8s} : {len(out_df):>9,} ticks  {first.strftime('%m-%d %H:%M')} -> {last.strftime('%m-%d %H:%M')}  ({sz:.1f} MB)")
        total_ticks += len(out_df)

    exe.shutdown()
    print(f"\nTotal : {total_ticks:,} ticks exportes dans {OUT_DIR}")
    print(f"Taille totale : {sum(f.stat().st_size for f in OUT_DIR.glob('*.parquet'))/1024/1024:.1f} MB")


if __name__ == "__main__":
    main()
