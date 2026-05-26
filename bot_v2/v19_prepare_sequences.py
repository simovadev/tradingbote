"""V19 ETAPE 2 : Pour chaque trade champion/clear_loss, extraire
les sequences OHLCV brutes M1/M15/H1 avant le trade.

Output : v19_sequences.npz contenant :
    m1_seqs:    (N, 240, 5) - OHLCV M1 normalise par retours %
    m15_seqs:   (N, 60, 5)
    h1_seqs:    (N, 30, 5)
    ict_feats:  (N, 50) - 50 features ICT classiques
    asset_ids:  (N,) - int 0..27
    labels:     (N,) - 0/1
    pnl_R:      (N,) - R-multiple (regression possible)
    timestamps: (N,) - pour split temporel

Normalisation OHLCV (anti-drift):
    - Pour chaque seq, on prend les % retours par rapport au prix de cloture du
      DERNIER bar de la seq (ancrage local).
    - high/low/open en % du close de la bougie
    - volume normalise par rolling mean
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = "c:/Users/Shadow/TradingBot" if sys.platform == "win32" else "/workspace/TradingBot"
sys.path.insert(0, ROOT)


M1_LEN = 240    # 4h en M1
M15_LEN = 60    # 15h en M15
H1_LEN = 30     # 30h en H1


def find_data_dir() -> Path:
    """Trouve le dossier data_vantage."""
    for cand in [Path(f"{ROOT}/data_vantage"),
                  Path(f"{ROOT}/VAST_BACKUP_V18_3/data_vantage")]:
        if cand.exists():
            return cand
    raise FileNotFoundError("data_vantage introuvable")


def load_ohlcv(asset: str, tf: str, data_dir: Path) -> pd.DataFrame | None:
    """Charge OHLCV pour asset/tf."""
    p = data_dir / f"{asset}_{tf}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df = df.set_index("ts")
    else:
        df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()


def normalize_seq(ohlcv: np.ndarray) -> np.ndarray:
    """Normalise une sequence OHLCV.

    Input:  (T, 5) = open/high/low/close/volume
    Output: (T, 5) normalise :
        - open/high/low en % du close de la bougie
        - close en % du close du dernier bar de la seq (ancrage local)
        - volume normalise par mean (ratio)
    """
    arr = ohlcv.astype(np.float32).copy()
    closes = arr[:, 3]

    if closes[-1] == 0 or np.any(np.isnan(closes)):
        return None

    # Open/high/low/close relatifs au close du DERNIER bar de la seq
    ref = closes[-1]
    out = np.zeros_like(arr)
    out[:, 0] = (arr[:, 0] / ref) - 1.0    # open / ref - 1
    out[:, 1] = (arr[:, 1] / ref) - 1.0    # high / ref - 1
    out[:, 2] = (arr[:, 2] / ref) - 1.0    # low / ref - 1
    out[:, 3] = (closes / ref) - 1.0       # close / ref - 1
    # Volume : ratio par rapport a la mean de la seq
    vol_mean = arr[:, 4].mean()
    out[:, 4] = arr[:, 4] / vol_mean if vol_mean > 0 else 0.0

    # Clip pour eviter outliers
    out = np.clip(out, -0.20, 0.20)  # max 20% retour vs ref
    return out


def extract_seq_for_trade(ts: pd.Timestamp, ohlcv_df: pd.DataFrame, n_bars: int) -> np.ndarray | None:
    """Extrait n_bars dernieres bougies STRICTEMENT avant ts."""
    if ohlcv_df is None or ohlcv_df.empty:
        return None
    # Coupe au dernier ts < trade_ts (strictly before)
    past = ohlcv_df[ohlcv_df.index < ts]
    if len(past) < n_bars:
        return None
    chunk = past.iloc[-n_bars:][["open", "high", "low", "close", "volume"]].values
    return normalize_seq(chunk)


# 50 features ICT (sous-set des features V18.x sans les drift)
ICT_FEATURES = [
    # Setup quality
    "ob_strength", "sweep_strength", "retest_count", "is_unicorn",
    # Trade
    "rr", "tp_source_htf", "tp_source_capped",
    # Daily bias
    "daily_bias_aligned", "daily_bias_neutral",
    # Killzones
    "kz_london", "kz_ny_am", "kz_ny_pm", "kz_asia", "kz_ny_lunch", "kz_london_close",
    # Confluences
    "has_smt", "has_feu_vert", "has_breaker_kz", "has_mss_fvg",
    "has_po3_dist", "has_phase_expansion", "has_open_midnight_respect",
    "has_FVG_sync", "has_parent_ob", "has_grandparent_ob",
    "has_good_zone", "has_session_direction",
    # OB structure
    "ob_group_size", "bars_sweep_to_validation", "bars_group_to_validation",
    "is_bullish",
    # Time cyclic
    "hour_sin", "hour_cos", "day_of_week", "minutes_into_killzone",
    # Volume relatif
    "volume_relatif", "vol_ratio_setup",
    # Phases
    "phase_reversal", "phase_manipulation", "has_mss_nearby",
    # PO3
    "po3_body_pct", "po3_upper_wick", "po3_lower_wick",
    "po3_aligned", "po3_htf2_aligned",
    "displacement_ratio", "fib_level",
    # Momentum
    "mom_60", "mom_240", "body_ratio_recent", "mom_aligned",
    # ATR ratio (stationnaire)
    "atr_ratio_100",
]


def prepare_one_asset(asset: str, asset_id: int, data_dir: Path,
                       champions_df: pd.DataFrame, verbose: bool = True):
    """Process un actif : load OHLCV, extract sequences pour chaque trade."""
    df_asset = champions_df[champions_df["asset_name"] == asset].copy()
    if len(df_asset) == 0:
        return None

    if verbose:
        print(f"  {asset:<11} : {len(df_asset)} trades to process", flush=True)

    # Charge OHLCV M1, M15, H1
    df_m1 = load_ohlcv(asset, "M1", data_dir)
    df_m15 = load_ohlcv(asset, "M15", data_dir)
    df_h1 = load_ohlcv(asset, "H1", data_dir)

    if df_m1 is None or df_m15 is None or df_h1 is None:
        if verbose:
            print(f"  {asset:<11} : OHLCV manquant, skip", flush=True)
        return None

    m1_seqs, m15_seqs, h1_seqs = [], [], []
    ict_feats, asset_ids, labels, pnl_Rs, timestamps = [], [], [], [], []

    n_kept = 0
    for _, row in df_asset.iterrows():
        ts = row["ts"]
        m1 = extract_seq_for_trade(ts, df_m1, M1_LEN)
        if m1 is None:
            continue
        m15 = extract_seq_for_trade(ts, df_m15, M15_LEN)
        if m15 is None:
            continue
        h1 = extract_seq_for_trade(ts, df_h1, H1_LEN)
        if h1 is None:
            continue

        # ICT features (50)
        ict = []
        for f in ICT_FEATURES:
            if f in row.index:
                v = row[f]
                if isinstance(v, bool):
                    v = int(v)
                ict.append(float(v))
            else:
                ict.append(0.0)
        ict_arr = np.array(ict, dtype=np.float32)

        # pnl_R : +rr pour WIN, -1 pour LOSS
        if row["outcome"] == "WIN":
            pnl_R = float(row.get("rr", 2.0))
        else:
            pnl_R = -1.0

        m1_seqs.append(m1)
        m15_seqs.append(m15)
        h1_seqs.append(h1)
        ict_feats.append(ict_arr)
        asset_ids.append(asset_id)
        labels.append(int(row["label_v19"]))
        pnl_Rs.append(pnl_R)
        timestamps.append(ts)
        n_kept += 1

    if verbose:
        print(f"  {asset:<11} : kept {n_kept}/{len(df_asset)} (rest had insufficient history)", flush=True)

    return {
        "m1_seqs": np.array(m1_seqs, dtype=np.float32) if m1_seqs else None,
        "m15_seqs": np.array(m15_seqs, dtype=np.float32) if m15_seqs else None,
        "h1_seqs": np.array(h1_seqs, dtype=np.float32) if h1_seqs else None,
        "ict_feats": np.array(ict_feats, dtype=np.float32) if ict_feats else None,
        "asset_ids": np.array(asset_ids, dtype=np.int32) if asset_ids else None,
        "labels": np.array(labels, dtype=np.int32) if labels else None,
        "pnl_Rs": np.array(pnl_Rs, dtype=np.float32) if pnl_Rs else None,
        "timestamps": pd.to_datetime(timestamps) if timestamps else None,
    }


def main():
    print("=== V19 PREPARE SEQUENCES ===\n")

    # Charge champions
    champions_path = Path(f"{ROOT}/data/champions_v19.parquet")
    if not champions_path.exists():
        print(f"!! Missing {champions_path}. Run v19_extract_champions first.")
        return

    champions = pd.read_parquet(champions_path)
    print(f"Champions loaded : {len(champions)} samples")
    print(f"  Champions WIN : {champions['is_champion'].sum()}")
    print(f"  Clear LOSS    : {champions['is_clear_loss'].sum()}")

    data_dir = find_data_dir()
    print(f"OHLCV dir : {data_dir}\n")

    ALL_ASSETS = sorted(champions["asset_name"].unique().tolist())
    print(f"Processing {len(ALL_ASSETS)} assets...\n")

    all_m1, all_m15, all_h1 = [], [], []
    all_ict, all_aid, all_lbl, all_pnl, all_ts = [], [], [], [], []
    for asset in ALL_ASSETS:
        asset_id = sorted(champions["asset_name"].unique().tolist()).index(asset)
        res = prepare_one_asset(asset, asset_id, data_dir, champions)
        if res is None or res["m1_seqs"] is None:
            continue
        all_m1.append(res["m1_seqs"])
        all_m15.append(res["m15_seqs"])
        all_h1.append(res["h1_seqs"])
        all_ict.append(res["ict_feats"])
        all_aid.append(res["asset_ids"])
        all_lbl.append(res["labels"])
        all_pnl.append(res["pnl_Rs"])
        all_ts.append(res["timestamps"])

    m1 = np.concatenate(all_m1)
    m15 = np.concatenate(all_m15)
    h1 = np.concatenate(all_h1)
    ict = np.concatenate(all_ict)
    aid = np.concatenate(all_aid)
    lbl = np.concatenate(all_lbl)
    pnl = np.concatenate(all_pnl)
    ts_arr = np.concatenate([t.values.astype("datetime64[ns]") for t in all_ts])

    print(f"\n=== TOTAL SEQUENCES ===")
    print(f"  m1_seqs   shape : {m1.shape}")
    print(f"  m15_seqs  shape : {m15.shape}")
    print(f"  h1_seqs   shape : {h1.shape}")
    print(f"  ict_feats shape : {ict.shape}")
    print(f"  asset_ids shape : {aid.shape}")
    print(f"  labels    shape : {lbl.shape}")
    print(f"  pnl_Rs    shape : {pnl.shape}")

    # Sauvegarde compact
    out_path = Path(f"{ROOT}/data/v19_sequences.npz")
    np.savez_compressed(out_path,
                         m1_seqs=m1, m15_seqs=m15, h1_seqs=h1,
                         ict_feats=ict, asset_ids=aid, labels=lbl,
                         pnl_Rs=pnl, timestamps=ts_arr)
    size_mb = out_path.stat().st_size / 1e6
    print(f"\nSaved : {out_path.name} ({size_mb:.1f} MB)")

    # Distribution
    print(f"\n=== Distribution labels ===")
    print(f"  Champions (1) : {int((lbl == 1).sum())}")
    print(f"  Clear LOSS (0): {int((lbl == 0).sum())}")
    print(f"  Ratio WIN : {lbl.mean():.2%}")


if __name__ == "__main__":
    main()
