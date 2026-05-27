"""V20 EXTRACT CHAMPIONS : filtre WIN + clear LOSS sur 78 actifs (28 V18.7 + 50 V20).

Difference vs v19_extract_champions.py :
- Lit 28 datasets V18.7 + 50 datasets V20 (au lieu de 28)
- N'utilise PAS asset_id (V20 = sans embedding)
- Garde asset_name pour stats/debug uniquement

Output : data/champions_v20.parquet
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path("/workspace/TradingBot") if sys.platform == "linux" else Path("c:/Users/Shadow/TradingBot")
sys.path.insert(0, str(ROOT))

# Charge selection 78 actifs
sel = json.loads((ROOT / "vantage_70_selection.json").read_text())
ALL_78 = [a["v19_name"] for a in sel]

V19_28 = {"BTCUSD","ETHUSD","USDMXN","XAUUSD","USDJPY","USDZAR","USDCAD","NZDUSD",
          "GBPUSD","CL-OIL","EURUSD","AUDUSD","USDCHF","NAS100","XAGUSD","DJ30",
          "GAS-C","HK50","GER40","FRA40","UK100","Nikkei225","BVSPX","SP500",
          "Coffee-C","Cocoa-C","Wheat-C","Sugar-C"}


def is_champion_win(row) -> bool:
    """Le setup ICT est PARFAIT et a gagne."""
    try:
        return (
            row["outcome"] == "WIN"
            and row["rr"] >= 1.8
            and row.get("bars_to_exit", 0) >= 5
            and row.get("bars_to_exit", 0) <= 500
            and row["daily_bias_aligned"] == 1
            and row.get("has_FVG_sync", 0) == 1
            and row.get("ob_strength", 0) >= 0.4
            and row.get("sweep_strength", 0) >= 0.4
            and row["ob_group_size"] >= 2
            and row.get("kz_london_close", 0) == 0
        )
    except Exception:
        return False


def is_clear_loss(row) -> bool:
    """Le setup ICT est MAL QUALIFIE et a perdu = exemple de 'ne pas trader'."""
    try:
        if row["outcome"] != "LOSS":
            return False
        warnings = 0
        if row["daily_bias_aligned"] == 0: warnings += 1
        if row.get("has_FVG_sync", 0) == 0: warnings += 1
        if row.get("ob_strength", 0) < 0.3: warnings += 1
        if row.get("sweep_strength", 0) < 0.3: warnings += 1
        if row.get("kz_london_close", 0) == 1: warnings += 1
        return warnings >= 2
    except Exception:
        return False


def extract_one_asset(asset: str) -> pd.DataFrame | None:
    """V20 FIX : garde TOUS les WIN/LOSS, sans filtre champion/clear_loss.

    Le filtre champion/clear_loss faisait que :
    - WIN = setups parfaits (8% de tous les trades)
    - LOSS = setups catastrophiques (80% des trades)
    -> separation triviale, AUC=1 (data leakage par construction).

    On garde VRAI WIN/LOSS brut + on ajoute is_champion/is_clear_loss en
    metadonnees pour stats, mais le label = outcome reel.
    """
    if asset in V19_28:
        paths = [
            ROOT / "data" / f"ml_dataset_{asset}_vantage_v18_7.parquet",
            ROOT / "data" / f"ml_dataset_{asset}_vantage_v18_4.parquet",
            ROOT / "data" / f"ml_dataset_{asset}_vantage_v18_3.parquet",
        ]
        version = "V18.7"
    else:
        paths = [ROOT / "data" / f"ml_dataset_{asset}_vantage_v20.parquet"]
        version = "V20"

    p = next((x for x in paths if x.exists()), None)
    if p is None:
        print(f"  {asset:12} [{version}] : DATASET INTROUVABLE")
        return None

    df = pd.read_parquet(p)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    # Garde TOUS les WIN/LOSS (pas de filtre)
    df = df[df["outcome"].isin(["WIN", "LOSS"])].copy()

    # Stats pour info (champion / clear_loss) - PAS utilise comme label
    df["is_champion"] = df.apply(is_champion_win, axis=1)
    df["is_clear_loss"] = df.apply(is_clear_loss, axis=1)

    df["asset_name"] = asset
    # Label V20 = VRAI WIN/LOSS (pas is_champion)
    df["label_v20"] = (df["outcome"] == "WIN").astype(int)

    return df


def main():
    print(f"=== V20 EXTRACT CHAMPIONS : {len(ALL_78)} actifs ===\n")
    print(f"  V18.7 datasets (28 V19 originaux) : {len(V19_28)}")
    print(f"  V20 datasets (50 nouveaux)        : {len(ALL_78) - len(V19_28)}")
    print()

    all_dfs = []
    for asset in ALL_78:
        df = extract_one_asset(asset)
        if df is None:
            continue
        n_champ = df["is_champion"].sum()
        n_loss = df["is_clear_loss"].sum()
        print(f"  {asset:12} : {len(df):>6} (champion={n_champ:>5}, loss={n_loss:>5})")
        all_dfs.append(df)

    print()
    if not all_dfs:
        print("AUCUN actif extrait. Abort.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    out_path = ROOT / "data" / "champions_v20.parquet"
    combined.to_parquet(out_path)
    print(f"=== TOTAL : {len(combined):,} trades sauvegardes ===")
    print(f"  Champions WIN : {combined['is_champion'].sum():,}")
    print(f"  Clear LOSS    : {combined['is_clear_loss'].sum():,}")
    print(f"  Output : {out_path}")
    print()

    # Stats par categorie
    print("Distribution par actif (top 15) :")
    counts = combined.groupby("asset_name").size().sort_values(ascending=False)
    for asset, n in counts.head(15).items():
        n_champ = ((combined["asset_name"] == asset) & combined["is_champion"]).sum()
        n_loss = ((combined["asset_name"] == asset) & combined["is_clear_loss"]).sum()
        print(f"  {asset:12} : {n:>6} (champ={n_champ:>5}, loss={n_loss:>5})")

    # Save stats
    stats = {
        "total_trades": int(len(combined)),
        "n_champions": int(combined["is_champion"].sum()),
        "n_clear_losses": int(combined["is_clear_loss"].sum()),
        "n_assets": int(combined["asset_name"].nunique()),
        "trades_per_asset": counts.to_dict(),
    }
    (ROOT / "champions_v20_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"\nStats : champions_v20_stats.json")


if __name__ == "__main__":
    main()
