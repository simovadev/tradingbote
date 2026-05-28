"""V21 PREPARE : sequences avec CROSS-ASSET (5 actifs reference).

Pour chaque trade :
- Focus : bougies M1 (240) + M15 (60) + H1 (30) + D1 (15) de l'actif tradeé
- Ref   : bougies M15 (60) des 5 actifs reference (XAUUSD, USDJPY, SP500, BTCUSD, USDCHF)

Difference vs V20 : input = OHLCV brut (pas de features ICT calculees).
Le model V21 decouvre ses propres features.

Output : data/v21_chunks/<ASSET>.npz dans /dev/shm si dispo
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/workspace/TradingBot") if sys.platform == "linux" else Path("c:/Users/Shadow/TradingBot")
sys.path.insert(0, str(ROOT))

M1_LEN = 240
M15_LEN = 60
H1_LEN = 30
D1_LEN = 15

# 5 actifs reference universels (capturent contexte macro)
REF_ASSETS = ["XAUUSD", "USDJPY", "SP500", "BTCUSD", "USDCHF"]
N_REF = len(REF_ASSETS)


def load_ohlcv(asset: str, tf: str, data_dir: Path):
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
    arr = df[["open", "high", "low", "close", "volume"]].values.astype(np.float32)
    return arr, df.index.values.astype("datetime64[ns]")


def normalize_chunk(ohlcv_chunk):
    """V21 FIX : z-score par sample sur prix + log-returns pour amplitude.

    Avant : (x/ref - 1) -> close[-1] toujours 0, returns microscopiques ~0.002
    Maintenant : log-returns par-pair + z-score -> signal amplifie 100x.
    """
    closes = ohlcv_chunk[:, 3]
    ref = closes[-1]
    if ref == 0 or np.isnan(ref) or len(closes) < 2:
        return None
    out = np.empty_like(ohlcv_chunk)
    # Log returns vs ref : log(x/ref) ~ (x-ref)/ref pour x proche de ref
    # Multiplier par 100 pour avoir des valeurs en %
    log_close = np.log(closes / ref) * 100
    out[:, 0] = np.log(ohlcv_chunk[:, 0] / ref) * 100
    out[:, 1] = np.log(ohlcv_chunk[:, 1] / ref) * 100
    out[:, 2] = np.log(ohlcv_chunk[:, 2] / ref) * 100
    out[:, 3] = log_close
    # Z-score par sample sur les 4 prix (echelle uniforme)
    price_std = out[:, :4].std()
    if price_std > 1e-6:
        out[:, :4] = out[:, :4] / price_std
    # Volume normalise
    vol_mean = ohlcv_chunk[:, 4].mean()
    out[:, 4] = (ohlcv_chunk[:, 4] / vol_mean - 1.0) if vol_mean > 0 else 0.0
    # Clip raisonnable
    return np.clip(out, -10.0, 10.0)


def get_chunks_dir():
    return Path("/dev/shm/v21_chunks") if Path("/dev/shm").exists() else ROOT / "data" / "v21_chunks"


# === REF cache global (charge 1x par worker, partage entre actifs focus) ===
_REF_CACHE = None


def _load_ref_cache(data_dir: Path):
    """Charge les 5 actifs ref une seule fois."""
    global _REF_CACHE
    if _REF_CACHE is not None:
        return _REF_CACHE
    cache = {}
    for ref in REF_ASSETS:
        arr, ts = load_ohlcv(ref, "M15", data_dir)
        if arr is None:
            print(f"  WARN: ref {ref} M15 introuvable", flush=True)
            cache[ref] = (None, None)
        else:
            cache[ref] = (arr, ts)
    _REF_CACHE = cache
    return cache


def process_asset(asset, data_dir_str, champions_path_str):
    t0 = time.time()
    try:
        data_dir = Path(data_dir_str)
        champions_df = pd.read_parquet(champions_path_str)
        df_a = champions_df[champions_df["asset_name"] == asset].copy()
        if len(df_a) == 0:
            return {"status": "no_trades", "asset": asset}

        # Charge focus
        m1_arr, m1_ts = load_ohlcv(asset, "M1", data_dir)
        m15_arr, m15_ts = load_ohlcv(asset, "M15", data_dir)
        h1_arr, h1_ts = load_ohlcv(asset, "H1", data_dir)
        d1_arr, d1_ts = load_ohlcv(asset, "D1", data_dir)
        if any(x is None for x in (m1_arr, m15_arr, h1_arr, d1_arr)):
            return {"status": "no_ohlcv", "asset": asset}

        # Charge refs (cache global)
        refs = _load_ref_cache(data_dir)

        # Trade timestamps
        trade_ts = pd.to_datetime(df_a["ts"]).values.astype("datetime64[ns]")

        # Index pour chaque TF focus
        m1_idx = np.searchsorted(m1_ts, trade_ts, side="left")
        m15_idx = np.searchsorted(m15_ts, trade_ts, side="left")
        h1_idx = np.searchsorted(h1_ts, trade_ts, side="left")
        d1_idx = np.searchsorted(d1_ts, trade_ts, side="left")

        # Valid si on a assez d'historique sur TOUS les TF
        valid = (m1_idx >= M1_LEN) & (m15_idx >= M15_LEN) & (h1_idx >= H1_LEN) & (d1_idx >= D1_LEN)

        # Et valid si toutes les refs ont assez d'historique
        ref_idx_arrs = {}
        for ref in REF_ASSETS:
            ref_arr, ref_ts = refs[ref]
            if ref_arr is None:
                valid &= False
                ref_idx_arrs[ref] = None
                continue
            ref_idx = np.searchsorted(ref_ts, trade_ts, side="left")
            valid &= (ref_idx >= M15_LEN)
            ref_idx_arrs[ref] = ref_idx

        df_valid = df_a.iloc[valid].copy()
        n = len(df_valid)
        if n == 0:
            return {"status": "no_valid", "asset": asset}

        m1_idx = m1_idx[valid]
        m15_idx = m15_idx[valid]
        h1_idx = h1_idx[valid]
        d1_idx = d1_idx[valid]
        for ref in REF_ASSETS:
            ref_idx_arrs[ref] = ref_idx_arrs[ref][valid]
        ts_valid = trade_ts[valid]

        # Alloue
        m1_seqs = np.empty((n, M1_LEN, 5), dtype=np.float32)
        m15_seqs = np.empty((n, M15_LEN, 5), dtype=np.float32)
        h1_seqs = np.empty((n, H1_LEN, 5), dtype=np.float32)
        d1_seqs = np.empty((n, D1_LEN, 5), dtype=np.float32)
        ref_seqs = np.empty((n, N_REF, M15_LEN, 5), dtype=np.float32)
        keep = np.ones(n, dtype=bool)

        for i in range(n):
            r_m1 = normalize_chunk(m1_arr[m1_idx[i]-M1_LEN:m1_idx[i]])
            r_m15 = normalize_chunk(m15_arr[m15_idx[i]-M15_LEN:m15_idx[i]])
            r_h1 = normalize_chunk(h1_arr[h1_idx[i]-H1_LEN:h1_idx[i]])
            r_d1 = normalize_chunk(d1_arr[d1_idx[i]-D1_LEN:d1_idx[i]])
            if any(x is None for x in (r_m1, r_m15, r_h1, r_d1)):
                keep[i] = False
                continue
            m1_seqs[i] = r_m1
            m15_seqs[i] = r_m15
            h1_seqs[i] = r_h1
            d1_seqs[i] = r_d1
            # Refs
            ok_refs = True
            for j, ref in enumerate(REF_ASSETS):
                ref_arr, _ = refs[ref]
                idx = ref_idx_arrs[ref][i]
                r_ref = normalize_chunk(ref_arr[idx-M15_LEN:idx])
                if r_ref is None:
                    ok_refs = False
                    break
                ref_seqs[i, j] = r_ref
            if not ok_refs:
                keep[i] = False

        m1_seqs = m1_seqs[keep]
        m15_seqs = m15_seqs[keep]
        h1_seqs = h1_seqs[keep]
        d1_seqs = d1_seqs[keep]
        ref_seqs = ref_seqs[keep]
        df_valid = df_valid.iloc[keep]
        ts_valid = ts_valid[keep]

        # Labels + RR target (pour multi-task)
        labels = (df_valid["outcome"].values == "WIN").astype(np.int32)
        rr_target = df_valid["rr"].values.astype(np.float32)
        pnl_R = np.where(df_valid["outcome"].values == "WIN", rr_target, -1.0).astype(np.float32)

        chunks_root = get_chunks_dir()
        out_path = chunks_root / f"{asset}.npz"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_path,
                  m1_seqs=m1_seqs, m15_seqs=m15_seqs,
                  h1_seqs=h1_seqs, d1_seqs=d1_seqs,
                  ref_seqs=ref_seqs,
                  labels=labels, rr_target=rr_target, pnl_Rs=pnl_R,
                  timestamps=ts_valid)
        elapsed = time.time() - t0
        return {"status": "ok", "asset": asset, "n_seqs": len(df_valid),
                "elapsed": elapsed}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "exception", "asset": asset, "error": str(e)[:300]}


def main():
    print(f"=== V21 PREPARE : 78 workers, cross-asset (refs: {REF_ASSETS}) ===\n", flush=True)

    champions_path = ROOT / "data" / "champions_v20.parquet"  # reutilise V20 trades
    if not champions_path.exists():
        print(f"FAIL : {champions_path} introuvable")
        return
    champions = pd.read_parquet(champions_path)
    print(f"Trades : {len(champions):,}")

    data_dir = ROOT / "data_vantage"
    print(f"OHLCV dir : {data_dir}\n")

    ALL_ASSETS = sorted(champions["asset_name"].unique().tolist())
    print(f"Actifs focus : {len(ALL_ASSETS)}")
    chunks_dir = get_chunks_dir()
    chunks_dir.mkdir(parents=True, exist_ok=True)
    print(f"Chunks dir : {chunks_dir}")

    todo = []
    for asset in ALL_ASSETS:
        out = chunks_dir / f"{asset}.npz"
        if out.exists():
            print(f"  {asset:<12} : deja fait (skip)")
            continue
        todo.append((asset, str(data_dir), str(champions_path)))

    print(f"\nA traiter : {len(todo)} / {len(ALL_ASSETS)}\n", flush=True)

    if todo:
        n_workers = min(len(todo), int(os.environ.get("N_WORKERS", "78")))
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(process_asset, *t): t[0] for t in todo}
            for fut in as_completed(futs):
                try:
                    r = fut.result()
                except Exception as e:
                    r = {"status": "exception", "error": str(e)[:200]}
                status = r.get("status", "?")
                if status == "ok":
                    print(f"  >>> {r['asset']:<12} : {r['n_seqs']:>6} seqs ({r['elapsed']:.1f}s)", flush=True)
                else:
                    print(f"  >>> {r.get('asset', '?'):<12} : {status} {r.get('error', '')}", flush=True)
        print(f"\nTermine en {(time.time()-t0)/60:.1f} min", flush=True)

    # Stats
    print("\n=== STATS ===", flush=True)
    chunks = sorted(chunks_dir.glob("*.npz"))
    total_n = 0
    total_wins = 0
    for c in chunks:
        d = np.load(c, allow_pickle=True)
        n = len(d["labels"])
        total_n += n
        total_wins += int(d["labels"].sum())
    print(f"Total seqs : {total_n:,}")
    print(f"WR brut    : {total_wins/total_n*100:.1f}%")
    print(f"Chunks    : {chunks_dir}")


if __name__ == "__main__":
    main()
