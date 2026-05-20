"""Build dataset ML V5 pour 14 actifs - filtres caches retires + features volatilite.

V5 changes vs V4 (2026-05-20) :
- Vire confirm_ob_with_mss (window=10) -> le ML decide via feature has_mss_nearby
- max_bars_after_sweep 10->30 (capture OB qui prennent plus de temps a valider)
- max_scan_bars 500->1440 (backtest scan 24h au lieu 8h)
- Phase manipulation = malus -8 (au lieu REJET) -> trade les retournements violents
- MSS min_displacement_atr 0.5->0.3 (MSS plus sensible en volatilite moyenne)
- 4 nouvelles features ML : phase_reversal, phase_manipulation, vol_ratio_setup, has_mss_nearby
- Garde rejet phase accumulation (marche vraiment mort, bible Vizion §11.1)

Objectif : 3-4x plus de candidats, le ML decide en tous contextes de volatilite.

Usage:
    python -m bot_v2.build_v5_dataset XAUUSD
    python -m bot_v2.build_v5_dataset --all
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
    # V5.1 (2026-05-21) : chunks adaptatifs au nombre de cores.
    # Sur 192 threads, on veut ~192 chunks pour saturer la machine.
    # 7.5 ans = ~90 mois -> chunks ~14j = 192 chunks (ou 1 mois = 91 chunks)
    if cpu_count >= 128:
        chunk_months = 0.5   # ~14j -> ~192 chunks (sature 192 threads)
    elif cpu_count >= 64:
        chunk_months = 1     # ~91 chunks
    else:
        chunk_months = 3     # ~31 chunks

    print(f"\n=== ML DATASET {asset} V5 (cleanup filtres + features volatilite) ===", flush=True)
    print(f"Plage data : {earliest_start.date()} -> {earliest_end.date()}", flush=True)
    print(f"Bougies M1 : {len(df):,}", flush=True)
    print(f"Annees     : {(earliest_end - earliest_start).days / 365.25:.2f}", flush=True)
    print(f"Cores      : {cpu_count} -> chunks de {chunk_months} mois", flush=True)

    output_path = Path(f"{ROOT}/data/ml_dataset_{asset}_admiral_8ans_V5.parquet")
    df_result = build_dataset(
        earliest_start,
        earliest_end,
        [asset],
        output_path=output_path,
        chunk_months=chunk_months,
        ltf="M1",
        version_suffix="_V5",  # ml_partial_M1_V5/ -> separe des chunks V4
    )

    print(f"\n>>> RECAP {asset} V5 : {len(df_result):,} candidats totaux", flush=True)
    if len(df_result) > 0 and "outcome" in df_result.columns:
        closed = df_result[df_result["outcome"].isin(["WIN", "LOSS"])]
        if len(closed) > 0:
            wr = (closed["outcome"] == "WIN").mean() * 100
            print(f">>> RECAP {asset} V5 : WR brut {wr:.1f}% sur {len(closed)} fermes", flush=True)
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
