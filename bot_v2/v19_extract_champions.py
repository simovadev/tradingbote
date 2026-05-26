"""V19 ETAPE 1 : Extract champions WIN + clear LOSS pour entrainer le DL.

Filtre les ~691k trades pour ne garder que :
- CHAMPIONS WIN : setups ICT parfaits qui ont gagne (~30-50k)
- CLEAR LOSS   : setups mal qualifies qui ont perdu (~30-50k)
- AMBIGU       : jete (le bruit qui faisait overfit le ML)

L'idee : donner au DL UNIQUEMENT des exemples nets pour qu'il apprenne
les VRAIS patterns ICT, pas les setups borderline.

Output : champions_v19.parquet (tous actifs concatenes avec asset_id)
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


def is_champion_win(row) -> bool:
    """Le setup ICT est PARFAIT et a gagne."""
    try:
        return (
            row["outcome"] == "WIN"
            and row["rr"] >= 1.8
            and row.get("bars_to_exit", 0) >= 5      # pas WIN en 1 bougie (chance pure)
            and row.get("bars_to_exit", 0) <= 500     # pas trainé pendant 8h+
            and row["daily_bias_aligned"] == 1
            and row.get("has_FVG_sync", 0) == 1
            and row.get("ob_strength", 0) >= 0.4
            and row.get("sweep_strength", 0) >= 0.4
            and row["ob_group_size"] >= 2
            and row.get("kz_london_close", 0) == 0   # exclure London Close
        )
    except Exception:
        return False


def is_clear_loss(row) -> bool:
    """Le setup ICT est MAL QUALIFIE et a perdu = exemple de "ne pas trader"."""
    try:
        if row["outcome"] != "LOSS":
            return False
        # Multiple warning signs (au moins 2)
        warnings = 0
        if row["daily_bias_aligned"] == 0:
            warnings += 1
        if row.get("has_FVG_sync", 0) == 0:
            warnings += 1
        if row.get("ob_strength", 0) < 0.3:
            warnings += 1
        if row.get("sweep_strength", 0) < 0.3:
            warnings += 1
        if row.get("kz_london_close", 0) == 1:
            warnings += 1
        return warnings >= 2
    except Exception:
        return False


def extract_one_asset(asset: str) -> pd.DataFrame | None:
    """Charge dataset V18.7/V18.4/V18.3, retourne champions + losses."""
    paths = [
        Path(f"{ROOT}/data/ml_dataset_{asset}_vantage_v18_7.parquet"),
        Path(f"{ROOT}/data/ml_dataset_{asset}_vantage_v18_4.parquet"),
        Path(f"{ROOT}/data/ml_dataset_{asset}_vantage_v18_3.parquet"),
        # Fallback : backup local
        Path(f"{ROOT}/VAST_BACKUP_V18_3/datasets_v18_3/ml_dataset_{asset}_vantage_v18_3.parquet"),
    ]
    p = next((x for x in paths if x.exists()), None)
    if p is None:
        return None

    df = pd.read_parquet(p)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    # Filtre WIN/LOSS only
    df = df[df["outcome"].isin(["WIN", "LOSS"])].copy()

    # Mark champions / clear losses
    df["is_champion"] = df.apply(is_champion_win, axis=1)
    df["is_clear_loss"] = df.apply(is_clear_loss, axis=1)

    # Keep only champions OR clear losses
    keep = df["is_champion"] | df["is_clear_loss"]
    df_filtered = df[keep].copy()

    # Add asset_id for embedding
    df_filtered["asset_name"] = asset
    df_filtered["asset_id"] = ALL_ASSETS.index(asset)

    # Label V19 : 1 = champion (à imiter), 0 = clear loss (à éviter)
    df_filtered["label_v19"] = df_filtered["is_champion"].astype(int)

    return df_filtered


def main():
    print(f"=== V19 EXTRACT CHAMPIONS : {len(ALL_ASSETS)} actifs ===\n")

    all_dfs = []
    stats = []
    for asset in ALL_ASSETS:
        df = extract_one_asset(asset)
        if df is None:
            print(f"  {asset:<11} : pas de dataset")
            continue
        n_total = len(df) + 0  # already filtered
        n_champ = int(df["is_champion"].sum())
        n_loss = int(df["is_clear_loss"].sum())
        # Pour stats
        full_path = Path(f"{ROOT}/data/ml_dataset_{asset}_vantage_v18_7.parquet")
        if full_path.exists():
            n_total_orig = len(pd.read_parquet(full_path, columns=["outcome"]))
        else:
            n_total_orig = 0
        pct_champ = n_champ / n_total_orig * 100 if n_total_orig > 0 else 0
        print(f"  {asset:<11} : {n_total_orig:>6} total | {n_champ:>5} champ ({pct_champ:.1f}%) | {n_loss:>5} clear_loss")
        stats.append({
            "asset": asset, "n_total": n_total_orig,
            "n_champions": n_champ, "n_clear_loss": n_loss,
            "pct_champions": pct_champ,
        })
        all_dfs.append(df)

    # Concat all
    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values("ts").reset_index(drop=True)

    print(f"\n=== TOTAL POOL V19 ===")
    print(f"  Champions WIN : {combined['is_champion'].sum():,}")
    print(f"  Clear LOSS    : {combined['is_clear_loss'].sum():,}")
    print(f"  TOTAL pool    : {len(combined):,}")
    print(f"  Ratio WIN/LOSS : {combined['label_v19'].mean():.2%}")
    print(f"  Periode : {combined['ts'].min()} -> {combined['ts'].max()}")

    # Distribution par actif
    print(f"\n=== Distribution par actif (champions only) ===")
    for asset in ALL_ASSETS:
        n = ((combined["asset_name"] == asset) & combined["is_champion"]).sum()
        n_loss = ((combined["asset_name"] == asset) & combined["is_clear_loss"]).sum()
        bar_c = "#" * int(n / 200)
        bar_l = "." * int(n_loss / 200)
        print(f"  {asset:<11} : {bar_c}{bar_l} (champ={n}, loss={n_loss})")

    # Sauvegarde
    out_path = Path(f"{ROOT}/data/champions_v19.parquet")
    combined.to_parquet(out_path)
    print(f"\nSaved : {out_path.name} ({out_path.stat().st_size / 1e6:.1f} MB)")

    stats_path = Path(f"{ROOT}/champions_v19_stats.json")
    stats_path.write_text(json.dumps(stats, indent=2, default=str))
    print(f"Stats : {stats_path.name}")


if __name__ == "__main__":
    main()
