"""Build dataset ML V10 RR1.5 sur donnees Vantage 8 ans (data_vantage/).

V10 RR1.5 vs V8 :
- MEME data Vantage 8 ans (2018-03 -> 2026-05)
- FIX DATA LEAKAGES multiples :
  * count_ob_retests : capper a +15 bougies (etait : infini -> top7 feature leakee)
  * has_mss_nearby : forme avant validation (etait : ±10 bougies abs)
  * compute_ob_strength critere 3 FVG : avant validation (etait : ±3)
  * associated_pdr_count : <= validation_index (etait : +5)
  * SMT window : <= validation_ts (etait : +5min)
  * breaker pipeline : <= validation_index (etait : ±20)
  * FVG sync pipeline : <= validation_index (etait : ±1)

Impact attendu : WR backtest plus bas (60-68% au lieu de 77%) MAIS le live
pourra reproduire ces probas (plus de divergence training/live).

Usage :
    python -m bot_v2.build_v10rr15_vantage_dataset XAUUSD
    python -m bot_v2.build_v10rr15_vantage_dataset --all
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

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

TRAIN_START = pd.Timestamp("2018-03-01", tz="UTC")
TRAIN_END = pd.Timestamp("2026-05-22", tz="UTC")


def build_one(asset: str):
    df = load(asset, "M1")
    earliest_start = df.index[0]
    earliest_end = df.index[-1]
    print(f"\n=== ML DATASET {asset} V10 RR1.5 (Vantage 8 ans, RR=1.5 test) ===", flush=True)
    print(f"Source data      : data_vantage/{asset}_M1.parquet", flush=True)
    print(f"Plage data dispo : {earliest_start.date()} -> {earliest_end.date()}", flush=True)
    print(f"Bougies M1 dispo : {len(df):,}", flush=True)

    train_start = max(TRAIN_START, earliest_start)
    train_end = min(TRAIN_END, earliest_end)
    print(f"Fenetre training : {train_start.date()} -> {train_end.date()}", flush=True)
    n_days_training = (train_end - train_start).days
    print(f"Jours training : {n_days_training} jours (~{n_days_training/30:.1f} mois)", flush=True)

    cpu_count = os.cpu_count() or 4
    if cpu_count >= 256:
        chunk_months = 0.25
    elif cpu_count >= 128:
        chunk_months = 0.5
    elif cpu_count >= 64:
        chunk_months = 1
    else:
        chunk_months = 1.5
    print(f"Cores : {cpu_count} -> chunks de {chunk_months} mois", flush=True)

    output_path = Path(f"{ROOT}/data/ml_dataset_{asset}_vantage_v10rr15.parquet")
    df_result = build_dataset(
        train_start,
        train_end,
        [asset],
        output_path=output_path,
        chunk_months=chunk_months,
        ltf="M1",
        version_suffix="_V10RR15_VANTAGE",
    )

    print(f"\n>>> RECAP {asset} V10 RR1.5 : {len(df_result):,} candidats totaux", flush=True)
    if len(df_result) > 0 and "outcome" in df_result.columns:
        closed = df_result[df_result["outcome"].isin(["WIN", "LOSS"])]
        if len(closed) > 0:
            wr = (closed["outcome"] == "WIN").mean() * 100
            print(f">>> RECAP {asset} V10 RR1.5 : WR brut {wr:.1f}% sur {len(closed)} fermes",
                  flush=True)
    print("=" * 60, flush=True)


def main():
    p = argparse.ArgumentParser(description="Build V10 RR1.5 ML dataset (RR=1.5 test)")
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
