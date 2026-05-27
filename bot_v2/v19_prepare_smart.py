"""V19 ETAPE 2 SMART : 128 workers + resume + sauvegarde par actif.

Vs versions precedentes :
- 128 workers (saturation max Vast)
- Sauvegarde par actif (.npz/asset) -> resume si crash
- Chunking trades intra-actif pour ne PAS bloquer 1 worker sur 1 gros actif
- Skip actifs deja faits

Strategie :
- Chunks de 500 trades par worker
- Donc BTCUSD (13k trades) = 26 chunks = 26 workers en parallele
- Tous les actifs charges en parallel = saturation 128 workers totale
- Concat final a la fin
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

from bot_v2.v19_prepare_sequences import (
    M1_LEN, M15_LEN, H1_LEN,
    find_data_dir, load_ohlcv, extract_seq_for_trade, ICT_FEATURES,
)

N_PARALLEL = int(os.environ.get("V19_SMART_PARALLEL", "100"))
CHUNK_SIZE = int(os.environ.get("V19_CHUNK_SIZE", "200"))  # 200 trades par worker


def _process_chunk(args):
    """Process 1 chunk de N trades pour 1 actif. Subprocess isole."""
    asset, asset_id, chunk_df_path, data_dir_str, chunk_idx = args

    try:
        import pandas as pd
        import numpy as np
        from pathlib import Path
        from bot_v2.v19_prepare_sequences import (
            extract_seq_for_trade, ICT_FEATURES, M1_LEN, M15_LEN, H1_LEN, load_ohlcv,
        )

        chunk_df = pd.read_parquet(chunk_df_path)
        data_dir = Path(data_dir_str)

        # Load OHLCV pour cet actif (par chunk = redondant mais simple)
        df_m1 = load_ohlcv(asset, "M1", data_dir)
        df_m15 = load_ohlcv(asset, "M15", data_dir)
        df_h1 = load_ohlcv(asset, "H1", data_dir)
        if df_m1 is None or df_m15 is None or df_h1 is None:
            return {"status": "no_ohlcv", "asset": asset, "chunk_idx": chunk_idx}

        m1_seqs, m15_seqs, h1_seqs = [], [], []
        ict_feats, asset_ids, labels, pnl_Rs, timestamps = [], [], [], [], []

        for _, row in chunk_df.iterrows():
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

            ict = []
            for f in ICT_FEATURES:
                if f in row.index:
                    v = row[f]
                    if isinstance(v, bool):
                        v = int(v)
                    ict.append(float(v))
                else:
                    ict.append(0.0)

            m1_seqs.append(m1)
            m15_seqs.append(m15)
            h1_seqs.append(h1)
            ict_feats.append(np.array(ict, dtype=np.float32))
            asset_ids.append(asset_id)
            labels.append(int(row["label_v19"]))
            pnl_R = float(row.get("rr", 2.0)) if row["outcome"] == "WIN" else -1.0
            pnl_Rs.append(pnl_R)
            timestamps.append(ts)

        # Sauvegarde immediate du chunk (resume safe)
        chunk_out = Path(f"{ROOT}/data/v19_chunks/{asset}_chunk{chunk_idx:04d}.npz")
        chunk_out.parent.mkdir(parents=True, exist_ok=True)
        if not m1_seqs:
            return {"status": "no_seqs", "asset": asset, "chunk_idx": chunk_idx}

        np.savez(chunk_out,
                  m1_seqs=np.array(m1_seqs, dtype=np.float32),
                  m15_seqs=np.array(m15_seqs, dtype=np.float32),
                  h1_seqs=np.array(h1_seqs, dtype=np.float32),
                  ict_feats=np.array(ict_feats, dtype=np.float32),
                  asset_ids=np.array(asset_ids, dtype=np.int32),
                  labels=np.array(labels, dtype=np.int32),
                  pnl_Rs=np.array(pnl_Rs, dtype=np.float32),
                  timestamps=pd.DatetimeIndex(timestamps).values.astype("datetime64[ns]"))

        # Cleanup chunk_df
        Path(chunk_df_path).unlink()
        return {"status": "ok", "asset": asset, "chunk_idx": chunk_idx, "n_seqs": len(m1_seqs)}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "exception", "asset": asset, "chunk_idx": chunk_idx, "error": str(e)[:300]}


def main():
    print(f"=== V19 PREPARE SMART : {N_PARALLEL} workers, chunks={CHUNK_SIZE} trades ===\n")

    champions_path = Path(f"{ROOT}/data/champions_v19.parquet")
    champions = pd.read_parquet(champions_path)
    print(f"Champions : {len(champions):,}")

    data_dir = find_data_dir()
    ALL_ASSETS = sorted(champions["asset_name"].unique().tolist())

    # Cleanup chunks dir
    chunks_dir = Path(f"{ROOT}/data/v19_chunks")
    chunks_dir.mkdir(parents=True, exist_ok=True)

    # Skip si chunks deja existants pour resume
    existing = list(chunks_dir.glob("*.npz"))
    existing_assets = set(p.stem.split("_chunk")[0] for p in existing)
    print(f"Chunks deja faits : {len(existing)} (actifs touches : {len(existing_assets)})")

    # Split chaque actif en chunks de CHUNK_SIZE
    tasks = []
    for asset in ALL_ASSETS:
        asset_id = ALL_ASSETS.index(asset)
        df_a = champions[champions["asset_name"] == asset].copy()
        n_chunks = (len(df_a) + CHUNK_SIZE - 1) // CHUNK_SIZE
        for i in range(n_chunks):
            # Skip si chunk existe deja
            chunk_path = chunks_dir / f"{asset}_chunk{i:04d}.npz"
            if chunk_path.exists():
                continue

            chunk_df = df_a.iloc[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE]
            chunk_df_path = chunks_dir / f"_tmp_{asset}_chunk{i:04d}.parquet"
            chunk_df.to_parquet(chunk_df_path)
            tasks.append((asset, asset_id, str(chunk_df_path), str(data_dir), i))

    print(f"Tasks a executer : {len(tasks)}")
    print(f"Actifs * chunks : {sum(((champions['asset_name']==a).sum() + CHUNK_SIZE - 1) // CHUNK_SIZE for a in ALL_ASSETS)}")
    print()

    if not tasks:
        print("Tous les chunks deja faits, on consolide directement")
    else:
        t0 = time.time()
        ok_count = 0
        skipped = 0
        with ProcessPoolExecutor(max_workers=N_PARALLEL) as ex:
            futs = {ex.submit(_process_chunk, t): t for t in tasks}
            for fut in as_completed(futs):
                try:
                    r = fut.result()
                except Exception as e:
                    r = {"status": "exception", "error": str(e)[:200]}
                if r.get("status") == "ok":
                    ok_count += 1
                    if ok_count % 20 == 0:
                        elapsed = (time.time() - t0) / 60
                        rate = ok_count / elapsed if elapsed > 0 else 0
                        print(f"  done={ok_count}/{len(tasks)} ({elapsed:.1f}min, {rate:.0f} chunks/min)", flush=True)
                else:
                    skipped += 1
        print(f"\nTermine en {(time.time()-t0)/60:.1f} min : {ok_count} OK, {skipped} skipped/failed")

    # Consolidation finale
    print("\n=== CONSOLIDATION ===")
    chunks = sorted(chunks_dir.glob("*.npz"))
    print(f"Total chunks a merger : {len(chunks)}")

    m1_list, m15_list, h1_list = [], [], []
    ict_list, aid_list, lbl_list, pnl_list, ts_list = [], [], [], [], []
    for c in chunks:
        d = np.load(c, allow_pickle=True)
        m1_list.append(d["m1_seqs"])
        m15_list.append(d["m15_seqs"])
        h1_list.append(d["h1_seqs"])
        ict_list.append(d["ict_feats"])
        aid_list.append(d["asset_ids"])
        lbl_list.append(d["labels"])
        pnl_list.append(d["pnl_Rs"])
        ts_list.append(d["timestamps"])

    m1 = np.concatenate(m1_list)
    m15 = np.concatenate(m15_list)
    h1 = np.concatenate(h1_list)
    ict = np.concatenate(ict_list)
    aid = np.concatenate(aid_list)
    lbl = np.concatenate(lbl_list)
    pnl = np.concatenate(pnl_list)
    ts_arr = np.concatenate(ts_list)

    print(f"\nTotal sequences : {len(lbl):,}")
    print(f"  m1_seqs   : {m1.shape}")
    print(f"  WIN  (1) : {(lbl==1).sum():,}")
    print(f"  LOSS (0) : {(lbl==0).sum():,}")

    out_path = Path(f"{ROOT}/data/v19_sequences.npz")
    np.savez_compressed(out_path,
                         m1_seqs=m1, m15_seqs=m15, h1_seqs=h1,
                         ict_feats=ict, asset_ids=aid, labels=lbl,
                         pnl_Rs=pnl, timestamps=ts_arr)
    size_mb = out_path.stat().st_size / 1e6
    print(f"\nSaved : {out_path.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
