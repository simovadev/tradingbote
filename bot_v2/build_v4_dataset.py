"""Build dataset ML V4 pour 14 actifs - displacement DYNAMIQUE selon volatilite.

V4 changes vs V3.5 (2026-05-21) :
- min_displacement_atr s'adapte automatiquement a la volatilite (vol_ratio ATR14/ATR100)
- Capture les setups en marche chaotique (qui etaient rejetes en V3.5)
- Le ML apprend les patterns en TOUTES conditions, pas juste calme

Usage:
    python -m bot_v2.build_v4_dataset XAUUSD
    python -m bot_v2.build_v4_dataset --all
"""
from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path

ROOT = "c:/Users/Shadow/TradingBot" if sys.platform == "win32" else "/workspace/TradingBot"
sys.path.insert(0, ROOT)

from bot_v2.data_loader import load
from bot_v2.ml_dataset import build_dataset


ALL_ASSETS = [
    "XAUUSD", "NAS100", "GER40", "BTCUSD",
    "EURUSD", "GBPUSD", "AUDUSD", "USDJPY",
    "SP500", "DJ30", "UK100", "FRA40",
    "USDCAD", "USDCHF",
]


def build_one(asset: str):
    df = load(asset, "M1")
    earliest_end = df.index[-1]
    earliest_start = df.index[0]

    cpu_count = os.cpu_count() or 4
    chunk_months = 1 if cpu_count >= 64 else 3

    print(f"\n=== ML DATASET {asset} V4 (displacement dynamique) ===", flush=True)
    print(f"Plage data : {earliest_start.date()} -> {earliest_end.date()}", flush=True)
    print(f"Bougies M1 : {len(df):,}", flush=True)
    print(f"Annees     : {(earliest_end - earliest_start).days / 365.25:.2f}", flush=True)
    print(f"Cores      : {cpu_count} -> chunks de {chunk_months} mois", flush=True)

    output_path = Path(f"{ROOT}/data/ml_dataset_{asset}_admiral_8ans_V4.parquet")
    df_result = build_dataset(
        earliest_start,
        earliest_end,
        [asset],
        output_path=output_path,
        chunk_months=chunk_months,
        ltf="M1",
    )

    print(f"\n>>> RECAP {asset} V4 : {len(df_result):,} candidats totaux", flush=True)
    if len(df_result) > 0 and "outcome" in df_result.columns:
        closed = df_result[df_result["outcome"].isin(["WIN", "LOSS"])]
        if len(closed) > 0:
            wr = (closed["outcome"] == "WIN").mean() * 100
            print(f">>> RECAP {asset} V4 : WR brut {wr:.1f}% sur {len(closed)} fermes", flush=True)
    print("=" * 60, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("asset", nargs="?", help="Asset name (XAUUSD) ou --all")
    p.add_argument("--all", action="store_true", help="Build tous les 14 actifs")
    args = p.parse_args()

    if args.all or args.asset == "--all":
        for a in ALL_ASSETS:
            try:
                build_one(a)
            except Exception as e:
                print(f"!! FAIL {a} : {e}")
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
