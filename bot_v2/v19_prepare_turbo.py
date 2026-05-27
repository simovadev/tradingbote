"""V19 ETAPE 2 TURBO : 28 workers (1 par actif), extraction vectorisee numpy.

Vs smart (100 workers, chunks 200):
- I/O reduit : 1 chargement OHLCV par actif (vs 1 par chunk)
- Vectorisation numpy au lieu de pandas .iloc dans boucle
- Sauvegarde immediate par actif (resume safe)

Gain attendu : x5-10 vs smart, x50 vs sequentiel
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = "c:/Users/Shadow/TradingBot" if sys.platform == "win32" else "/workspace/TradingBot"
sys.path.insert(0, ROOT)

M1_LEN = 240
M15_LEN = 60
H1_LEN = 30

ICT_FEATURES = [
    # NOTE V19 FIX : on garde TOUTES les features car maintenant le label = vrai WIN/LOSS
    # (pas is_champion). Donc daily_bias_aligned, has_FVG_sync etc. sont des VRAIES
    # features predictives du WIN/LOSS, pas des fuites de label.
    "ob_strength", "sweep_strength", "retest_count", "is_unicorn",
    "rr", "tp_source_htf", "tp_source_capped",
    "daily_bias_aligned", "daily_bias_neutral",
    "kz_london", "kz_ny_am", "kz_ny_pm", "kz_asia", "kz_ny_lunch", "kz_london_close",
    "has_smt", "has_feu_vert", "has_breaker_kz", "has_mss_fvg",
    "has_po3_dist", "has_phase_expansion", "has_open_midnight_respect",
    "has_FVG_sync", "has_parent_ob", "has_grandparent_ob",
    "has_good_zone", "has_session_direction",
    "ob_group_size", "bars_sweep_to_validation", "bars_group_to_validation",
    "is_bullish",
    "hour_sin", "hour_cos", "day_of_week", "minutes_into_killzone",
    "volume_relatif", "vol_ratio_setup",
    "phase_reversal", "phase_manipulation", "has_mss_nearby",
    "po3_body_pct", "po3_upper_wick", "po3_lower_wick",
    "po3_aligned", "po3_htf2_aligned",
    "displacement_ratio", "fib_level",
    "mom_60", "mom_240", "body_ratio_recent", "mom_aligned",
    "atr_ratio_100",
]


def load_ohlcv_fast(asset: str, tf: str, data_dir: Path):
    """Charge OHLCV en numpy direct (plus rapide que pandas)."""
    p = data_dir / f"{asset}_{tf}.parquet"
    if not p.exists():
        return None, None
    df = pd.read_parquet(p)
    if "ts" in df.columns:
        ts = pd.to_datetime(df["ts"], utc=True).values
        df = df.set_index("ts")
    else:
        ts = pd.to_datetime(df.index, utc=True).values
    df.index = ts
    df = df.sort_index()
    # Index sorted = on peut faire searchsorted rapide
    arr = df[["open", "high", "low", "close", "volume"]].values.astype(np.float32)
    return arr, df.index.values.astype("datetime64[ns]")


def extract_seqs_vectorized(trade_ts_arr, ohlcv_arr, ohlcv_ts_arr, n_bars):
    """Vectorise l'extraction : pour chaque trade_ts, prend les n_bars precedents."""
    # Pour chaque trade_ts, trouve l'index du dernier OHLCV STRICTEMENT avant
    idx_array = np.searchsorted(ohlcv_ts_arr, trade_ts_arr, side="left")  # left = STRICT before
    # Pour les trades dont l'historique est insuffisant, marquer comme NaN
    valid_mask = idx_array >= n_bars
    return idx_array, valid_mask


def normalize_chunk(ohlcv_chunk):
    """Normalise un chunk OHLCV (T, 5) : open/high/low/close en % du dernier close, vol en ratio."""
    closes = ohlcv_chunk[:, 3]
    ref = closes[-1]
    if ref == 0 or np.isnan(ref):
        return None
    out = np.empty_like(ohlcv_chunk)
    out[:, 0] = (ohlcv_chunk[:, 0] / ref) - 1.0
    out[:, 1] = (ohlcv_chunk[:, 1] / ref) - 1.0
    out[:, 2] = (ohlcv_chunk[:, 2] / ref) - 1.0
    out[:, 3] = (closes / ref) - 1.0
    vol_mean = ohlcv_chunk[:, 4].mean()
    out[:, 4] = ohlcv_chunk[:, 4] / vol_mean if vol_mean > 0 else 0.0
    return np.clip(out, -0.20, 0.20)


def process_asset(asset, asset_id, data_dir_str, champions_path_str):
    """Charge OHLCV 1x puis extrait toutes les seqs vectorise."""
    t0 = time.time()
    try:
        data_dir = Path(data_dir_str)
        champions_df = pd.read_parquet(champions_path_str)
        df_a = champions_df[champions_df["asset_name"] == asset].copy()
        if len(df_a) == 0:
            return {"status": "no_trades", "asset": asset}

        # Load OHLCV 1 seule fois
        m1_arr, m1_ts = load_ohlcv_fast(asset, "M1", data_dir)
        m15_arr, m15_ts = load_ohlcv_fast(asset, "M15", data_dir)
        h1_arr, h1_ts = load_ohlcv_fast(asset, "H1", data_dir)
        if m1_arr is None or m15_arr is None or h1_arr is None:
            return {"status": "no_ohlcv", "asset": asset}

        # Trade timestamps
        trade_ts = pd.to_datetime(df_a["ts"]).values.astype("datetime64[ns]")
        # Convertit ts en datetime64[ns] des deux cotes pour searchsorted
        m1_ts = np.asarray(m1_ts, dtype="datetime64[ns]")
        m15_ts = np.asarray(m15_ts, dtype="datetime64[ns]")
        h1_ts = np.asarray(h1_ts, dtype="datetime64[ns]")

        m1_idx, m1_valid = extract_seqs_vectorized(trade_ts, m1_arr, m1_ts, M1_LEN)
        m15_idx, m15_valid = extract_seqs_vectorized(trade_ts, m15_arr, m15_ts, M15_LEN)
        h1_idx, h1_valid = extract_seqs_vectorized(trade_ts, h1_arr, h1_ts, H1_LEN)

        valid = m1_valid & m15_valid & h1_valid
        df_valid = df_a.iloc[valid].copy()
        m1_idx = m1_idx[valid]
        m15_idx = m15_idx[valid]
        h1_idx = h1_idx[valid]
        ts_valid = trade_ts[valid]

        n = len(df_valid)
        if n == 0:
            return {"status": "no_valid", "asset": asset}

        m1_seqs = np.empty((n, M1_LEN, 5), dtype=np.float32)
        m15_seqs = np.empty((n, M15_LEN, 5), dtype=np.float32)
        h1_seqs = np.empty((n, H1_LEN, 5), dtype=np.float32)
        keep = np.ones(n, dtype=bool)
        for i in range(n):
            r1 = normalize_chunk(m1_arr[m1_idx[i]-M1_LEN:m1_idx[i]])
            r15 = normalize_chunk(m15_arr[m15_idx[i]-M15_LEN:m15_idx[i]])
            rh1 = normalize_chunk(h1_arr[h1_idx[i]-H1_LEN:h1_idx[i]])
            if r1 is None or r15 is None or rh1 is None:
                keep[i] = False
                continue
            m1_seqs[i] = r1
            m15_seqs[i] = r15
            h1_seqs[i] = rh1

        m1_seqs = m1_seqs[keep]
        m15_seqs = m15_seqs[keep]
        h1_seqs = h1_seqs[keep]
        df_valid = df_valid.iloc[keep]
        ts_valid = ts_valid[keep]

        # ICT features
        ict_feats = np.zeros((len(df_valid), len(ICT_FEATURES)), dtype=np.float32)
        for j, f in enumerate(ICT_FEATURES):
            if f in df_valid.columns:
                vals = df_valid[f].values
                if vals.dtype == bool:
                    vals = vals.astype(np.float32)
                ict_feats[:, j] = vals.astype(np.float32)

        # PnL R
        pnl_R = np.where(df_valid["outcome"].values == "WIN",
                          df_valid["rr"].values.astype(np.float32),
                          -1.0).astype(np.float32)
        # FIX DATA LEAKAGE : label = vrai outcome (WIN/LOSS), pas la classification champion/clear_loss
        # (label_v19 = is_champion = derive de features que le DL voit aussi -> overfit trivial)
        labels = (df_valid["outcome"].values == "WIN").astype(np.int32)
        asset_ids = np.full(len(df_valid), asset_id, dtype=np.int32)

        # Sauvegarde immediate
        out_path = Path(f"{ROOT}/data/v19_chunks/{asset}.npz")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_path,
                  m1_seqs=m1_seqs, m15_seqs=m15_seqs, h1_seqs=h1_seqs,
                  ict_feats=ict_feats, asset_ids=asset_ids,
                  labels=labels, pnl_Rs=pnl_R, timestamps=ts_valid)

        elapsed = time.time() - t0
        return {"status": "ok", "asset": asset, "n_seqs": len(df_valid),
                "elapsed": elapsed}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "exception", "asset": asset, "error": str(e)[:300]}


def main():
    print("=== V19 PREPARE TURBO : 28 workers, vectorise numpy ===\n")

    # V19 FIX : utilise all_trades_v19 (sans biais selection) au lieu de champions_v19
    all_trades_path = Path(f"{ROOT}/data/all_trades_v19.parquet")
    if not all_trades_path.exists():
        all_trades_path = Path(f"{ROOT}/data/champions_v19.parquet")
    champions_path = all_trades_path
    champions = pd.read_parquet(champions_path)
    print(f"Trades : {len(champions):,} (from {champions_path.name})")

    # Trouve data_vantage
    data_dir = None
    for cand in [Path(f"{ROOT}/data_vantage"),
                  Path(f"{ROOT}/VAST_BACKUP_V18_3/data_vantage")]:
        if cand.exists():
            data_dir = cand
            break
    if data_dir is None:
        print("!! data_vantage introuvable")
        return
    print(f"OHLCV dir : {data_dir}\n")

    ALL_ASSETS = sorted(champions["asset_name"].unique().tolist())
    chunks_dir = Path(f"{ROOT}/data/v19_chunks")
    chunks_dir.mkdir(parents=True, exist_ok=True)

    # Skip actifs deja faits
    todo = []
    for asset in ALL_ASSETS:
        out = chunks_dir / f"{asset}.npz"
        if out.exists():
            print(f"  {asset:<11} : deja fait (skip)")
            continue
        asset_id = ALL_ASSETS.index(asset)
        todo.append((asset, asset_id, str(data_dir), str(champions_path)))

    print(f"\nA traiter : {len(todo)} / {len(ALL_ASSETS)}\n")

    if todo:
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=28) as ex:
            futs = {ex.submit(process_asset, *t): t[0] for t in todo}
            for fut in as_completed(futs):
                try:
                    r = fut.result()
                except Exception as e:
                    r = {"status": "exception", "error": str(e)[:200]}
                status = r.get("status", "?")
                if status == "ok":
                    print(f"  >>> {r['asset']:<11} : {r['n_seqs']:>5} seqs ({r['elapsed']:.1f}s)", flush=True)
                else:
                    print(f"  >>> {r.get('asset', '?'):<11} : {status} {r.get('error', '')}", flush=True)
        print(f"\nTermine en {(time.time()-t0)/60:.1f} min")

    # Consolidation
    print("\n=== CONSOLIDATION ===")
    chunks = sorted(chunks_dir.glob("*.npz"))
    print(f"Fichiers : {len(chunks)}")

    parts = {"m1_seqs": [], "m15_seqs": [], "h1_seqs": [],
             "ict_feats": [], "asset_ids": [], "labels": [],
             "pnl_Rs": [], "timestamps": []}
    for c in chunks:
        d = np.load(c, allow_pickle=True)
        for k in parts:
            parts[k].append(d[k])

    merged = {k: np.concatenate(v) for k, v in parts.items()}
    print(f"\nTotal seqs : {len(merged['labels']):,}")
    print(f"  WIN  (1) : {(merged['labels']==1).sum():,}")
    print(f"  LOSS (0) : {(merged['labels']==0).sum():,}")

    out_path = Path(f"{ROOT}/data/v19_sequences.npz")
    np.savez_compressed(out_path, **merged)
    print(f"\nSaved : {out_path.name} ({out_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
