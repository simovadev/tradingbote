"""Build dataset ML V12 sur donnees Vantage 8 ans (data_vantage/).

V12 vs V11 :
- V11 utilisait des snapshots K=[3,7,15] -> le ML voyait 3 a 15 bougies
  POST-validation pour decider. Identifie le 22/05 comme un DATA LEAKAGE :
  le ML apprenait a noter un OB sur le mouvement post-OB. En live, ca le
  faisait entrer trop tard (apres que le mouvement etait deja consomme).
- V12 = "PUR AMONT" : 1 seule evaluation par OB, au moment EXACT de la
  validation (cut_idx = vi + 1). Le cache (swings/fvgs/breakers/structure/mss)
  est filtre strictement <= vi. Le ML apprend a noter l'OB UNIQUEMENT sur
  son contexte amont (sweep, structure, deplacement avant l'OB, etc.).
- Pas de feature snapshot_k.
- Tout le reste (data 8 ans Vantage, simulate_trade, hyperparams) inchange.

Logique ICT/SMC respectee :
- Un OB se trade DES sa validation, pas 3-15 min apres
- La note depend du SETUP qui a cree l'OB, pas du mouvement post-OB
- Resultat attendu : le bot trade le DEBUT du mouvement, pas la fin

Usage :
    python -m bot_v2.build_v12_dataset XAUUSD
    python -m bot_v2.build_v12_dataset --all
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["BUILD_DATA_DIR"] = "data_vantage"
os.environ["BUILD_V12_MODE"] = "1"  # active mode amont strict dans ml_dataset.py

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
    print(f"\n=== ML DATASET {asset} V12 (Vantage 8 ans, PUR AMONT, zero leakage) ===", flush=True)
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
    print(f"BUILD_V12_MODE = {os.environ.get('BUILD_V12_MODE')} (1 = mode amont strict)", flush=True)

    output_path = Path(f"{ROOT}/data/ml_dataset_{asset}_vantage_v12.parquet")
    df_result = build_dataset(
        train_start,
        train_end,
        [asset],
        output_path=output_path,
        chunk_months=chunk_months,
        ltf="M1",
        version_suffix="_V12_VANTAGE",
    )

    print(f"\n>>> RECAP {asset} V12 : {len(df_result):,} candidats totaux", flush=True)
    if len(df_result) > 0 and "outcome" in df_result.columns:
        closed = df_result[df_result["outcome"].isin(["WIN", "LOSS"])]
        if len(closed) > 0:
            wr = (closed["outcome"] == "WIN").mean() * 100
            print(f">>> RECAP {asset} V12 : WR brut {wr:.1f}% sur {len(closed)} fermes",
                  flush=True)
    print("=" * 60, flush=True)


def main():
    p = argparse.ArgumentParser(
        description="Build V12 ML dataset (PUR AMONT - zero leakage, 1 ligne par OB)"
    )
    p.add_argument("asset", nargs="?", help="Asset (XAUUSD) ou --all")
    p.add_argument("--all", action="store_true", help="Build tous les 14 actifs")
    args = p.parse_args()

    print(f"BUILD_DATA_DIR forced to : {os.environ.get('BUILD_DATA_DIR')}", flush=True)
    print(f"BUILD_V12_MODE forced to : {os.environ.get('BUILD_V12_MODE')}", flush=True)

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
