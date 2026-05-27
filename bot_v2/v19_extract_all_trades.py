"""V19 EXTRACT ALL TRADES (sans biais de selection).

L'erreur V19 v1 : on filtrait les "champions ICT parfaits" et "clear losses"
en utilisant des features (daily_bias, FVG, OB strength). Resultat : le ML voyait
ces features et trouvait un raccourci trivial (AUC=100%).

V19 fix : on prend TOUS les WIN et TOUS les LOSS. Le ML doit apprendre la VRAIE
correlation features -> outcome sans biais.

Output : all_trades_v19.parquet
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = "c:/Users/Shadow/TradingBot" if sys.platform == "win32" else "/workspace/TradingBot"
sys.path.insert(0, ROOT)

ALL_ASSETS = [
    "BTCUSD", "ETHUSD", "USDMXN", "XAUUSD", "USDJPY", "USDZAR", "USDCAD", "NZDUSD",
    "GBPUSD", "CL-OIL", "EURUSD", "AUDUSD", "USDCHF", "NAS100", "XAGUSD",
    "DJ30", "GAS-C", "HK50", "GER40", "FRA40", "UK100", "Nikkei225",
    "BVSPX", "SP500", "Coffee-C", "Cocoa-C", "Wheat-C", "Sugar-C",
]


def main():
    print("=== V19 EXTRACT ALL TRADES (sans biais selection) ===\n")
    all_dfs = []
    stats = []
    for asset in ALL_ASSETS:
        paths = [
            Path(f"{ROOT}/data/ml_dataset_{asset}_vantage_v18_7.parquet"),
            Path(f"{ROOT}/data/ml_dataset_{asset}_vantage_v18_4.parquet"),
            Path(f"{ROOT}/data/ml_dataset_{asset}_vantage_v18_3.parquet"),
        ]
        p = next((x for x in paths if x.exists()), None)
        if p is None:
            print(f"  {asset:<11} : no data")
            continue
        df = pd.read_parquet(p)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        # Garde TOUT (WIN + LOSS)
        df = df[df["outcome"].isin(["WIN", "LOSS"])].copy()
        df["asset_name"] = asset
        df["asset_id"] = ALL_ASSETS.index(asset)
        n_win = int((df["outcome"] == "WIN").sum())
        n_loss = int((df["outcome"] == "LOSS").sum())
        wr = n_win / (n_win + n_loss) * 100 if (n_win + n_loss) > 0 else 0
        print(f"  {asset:<11} : {len(df):>6} trades | WIN={n_win:>5} LOSS={n_loss:>5} (WR brut {wr:.1f}%)")
        stats.append({"asset": asset, "n_total": len(df), "n_win": n_win, "n_loss": n_loss, "wr_brut": wr})
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values("ts").reset_index(drop=True)

    n_win = int((combined["outcome"] == "WIN").sum())
    n_loss = int((combined["outcome"] == "LOSS").sum())
    print(f"\n=== TOTAL ===")
    print(f"  Total trades : {len(combined):,}")
    print(f"  WIN  : {n_win:,}")
    print(f"  LOSS : {n_loss:,}")
    print(f"  WR brut : {n_win/(n_win+n_loss)*100:.1f}%")
    print(f"  Periode : {combined['ts'].min()} -> {combined['ts'].max()}")

    out = Path(f"{ROOT}/data/all_trades_v19.parquet")
    combined.to_parquet(out)
    print(f"\nSaved : {out.name} ({out.stat().st_size/1e6:.1f} MB)")

    Path(f"{ROOT}/all_trades_v19_stats.json").write_text(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    main()
