"""Telecharge l'historique M1 + tous TF de Vantage MT5 pour les 14 actifs.

PREREQUIS :
- MT5 desktop ouvert et connecte au compte Vantage
- Pour chaque actif : avoir scroll le chart M1 jusqu'en 2022 (utilise scroll_mt5.py)
  -> Sans ce scroll prealable, Vantage retourne 100k bougies max (~3.5 mois)

Output :
- data_vantage/<actif>_M1.parquet
- data_vantage/<actif>_M5.parquet
- ... pour tous les TF

Usage :
    python download_vantage_history.py XAUUSD          # un seul actif
    python download_vantage_history.py --all           # tous les actifs
    python download_vantage_history.py XAUUSD EURUSD   # plusieurs
"""
from __future__ import annotations

import sys
sys.path.insert(0, 'c:/Users/Shadow/TradingBot')

import argparse
import time
from pathlib import Path

import pandas as pd
import MetaTrader5 as mt5

from bot_v2.mt5_executor import MT5Executor, to_broker_symbol


# 14 actifs live (JP225 ecarte 2026-05-19)
ALL_ASSETS = [
    "XAUUSD", "NAS100", "GER40", "BTCUSD", "EURUSD", "GBPUSD", "AUDUSD", "USDJPY",
    "SP500", "DJ30", "UK100", "FRA40", "USDCAD", "USDCHF",
]

TF_MAP = {
    "M1":  mt5.TIMEFRAME_M1,
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1":  mt5.TIMEFRAME_H1,
    "H4":  mt5.TIMEFRAME_H4,
    "D1":  mt5.TIMEFRAME_D1,
}

OUTPUT_DIR = Path("c:/Users/Shadow/TradingBot/data_vantage")
OUTPUT_DIR.mkdir(exist_ok=True)

CHUNK_SIZE = 10000  # MT5 limite a ~10k bougies par requete
MAX_ITERATIONS = 1000  # 10k * 1000 = 10M bougies max


def download_tf(broker_sym: str, tf_name: str, tf_id: int) -> pd.DataFrame | None:
    """Download tous les chunks pour un TF donne."""
    all_rates = []
    pos = 0
    total = 0

    for i in range(MAX_ITERATIONS):
        rates = mt5.copy_rates_from_pos(broker_sym, tf_id, pos, CHUNK_SIZE)
        if rates is None or len(rates) == 0:
            err = mt5.last_error()
            if err[0] != 1:  # 1 = success / pas d'erreur
                print(f"    Stop a pos={pos}, err={err}")
            break

        df_chunk = pd.DataFrame(rates)
        all_rates.append(df_chunk)
        total += len(rates)

        if (i + 1) % 20 == 0:
            df_chunk['time_dt'] = pd.to_datetime(df_chunk['time'], unit='s', utc=True)
            print(f"    Chunk {i+1} : total={total}, plus vieille={df_chunk['time_dt'].min()}")

        pos += CHUNK_SIZE
        if len(rates) < CHUNK_SIZE:
            break

    if not all_rates:
        return None

    df = pd.concat(all_rates).drop_duplicates(subset='time').sort_values('time')
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df = df.set_index('time')
    df = df[['open', 'high', 'low', 'close', 'tick_volume']].rename(columns={'tick_volume': 'volume'})
    return df


def process_asset(asset: str, tfs: list[str]):
    broker_sym = to_broker_symbol(asset)
    print(f"\n=== {asset} (broker: {broker_sym}) ===")

    if not mt5.symbol_select(broker_sym, True):
        print(f"  ERREUR : impossible de selectionner {broker_sym}")
        return

    for tf in tfs:
        if tf not in TF_MAP:
            continue
        tf_id = TF_MAP[tf]
        print(f"\n  --- TF {tf} ---")
        t0 = time.time()
        df = download_tf(broker_sym, tf, tf_id)
        if df is None or len(df) == 0:
            print(f"    Aucune data pour {asset} {tf}")
            continue

        elapsed = time.time() - t0
        first = df.index[0]
        last = df.index[-1]
        duree = (last - first).days
        annees = duree / 365

        output_path = OUTPUT_DIR / f"{asset}_{tf}.parquet"
        df.to_parquet(output_path)
        print(f"    {len(df):>8} bougies | {first.date()} -> {last.date()} | {annees:.1f} ans | {elapsed:.0f}s")
        print(f"    Sauve : {output_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("assets", nargs="*", help="Actifs a telecharger (default: tous)")
    ap.add_argument("--all", action="store_true", help="Telecharge tous les actifs")
    ap.add_argument("--tf", default="all", help="TF (M1, M5, ..., D1 ou 'all')")
    args = ap.parse_args()

    if args.all or not args.assets:
        assets = ALL_ASSETS
    else:
        assets = args.assets

    if args.tf == "all":
        tfs = list(TF_MAP.keys())
    else:
        tfs = [args.tf]

    print(f"Actifs   : {assets}")
    print(f"TF       : {tfs}")
    print(f"Output   : {OUTPUT_DIR}")
    print()

    m = MT5Executor()
    if not m.initialize():
        print("ERREUR : impossible d'init MT5")
        return

    print(f"Connecte au compte {m.account_info.login} | balance={m.get_balance():.2f}€")

    try:
        for asset in assets:
            process_asset(asset, tfs)
    finally:
        m.shutdown()

    print(f"\n=== TERMINE - Cache dans {OUTPUT_DIR} ===")


if __name__ == "__main__":
    main()
