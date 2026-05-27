"""V19 ETAPE 2 PARALLEL : prepare sequences avec 14 actifs en parallele.

vs version sequentielle :
- 28 actifs traites un par un = 1-2h
- 14 actifs en parallele = 5-10 min
- Chaque worker traite 1 actif complet (load OHLCV + extract toutes seqs)

Output : v19_sequences.npz (identique a v19_prepare_sequences.py)
"""
from __future__ import annotations

import json
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
    prepare_one_asset,
    find_data_dir,
)

N_PARALLEL = int(os.environ.get("V19_PREPARE_PARALLEL", "28"))


def _worker(args):
    """Wrapper top-level pour pickling."""
    asset, asset_id, data_dir_str, champions_file = args
    # Re-import & load dans le subprocess (memory isolation)
    import pandas as pd
    from pathlib import Path
    from bot_v2.v19_prepare_sequences import prepare_one_asset

    data_dir = Path(data_dir_str)
    champions_df = pd.read_parquet(champions_file)

    try:
        t0 = time.time()
        res = prepare_one_asset(asset, asset_id, data_dir, champions_df, verbose=False)
        elapsed = time.time() - t0
        if res is None or res["m1_seqs"] is None:
            return {"asset": asset, "status": "no_data", "elapsed": elapsed}
        return {
            "asset": asset, "status": "ok",
            "elapsed": elapsed,
            "n_seqs": len(res["labels"]),
            "data": res,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"asset": asset, "status": "exception", "error": str(e)[:300]}


def main():
    print(f"=== V19 PREPARE SEQUENCES PARALLEL : {N_PARALLEL} actifs en parallele ===\n")

    champions_path = Path(f"{ROOT}/data/champions_v19.parquet")
    if not champions_path.exists():
        print(f"!! Missing {champions_path}")
        return

    champions = pd.read_parquet(champions_path)
    print(f"Champions loaded : {len(champions):,} samples")
    print(f"  Champions WIN : {champions['is_champion'].sum():,}")
    print(f"  Clear LOSS    : {champions['is_clear_loss'].sum():,}")

    data_dir = find_data_dir()
    print(f"OHLCV dir : {data_dir}\n")

    ALL_ASSETS = sorted(champions["asset_name"].unique().tolist())
    print(f"Processing {len(ALL_ASSETS)} assets en parallele...\n")

    # Build args for workers
    args_list = []
    for i, asset in enumerate(ALL_ASSETS):
        args_list.append((asset, i, str(data_dir), str(champions_path)))

    t0 = time.time()
    all_data = {}
    with ProcessPoolExecutor(max_workers=N_PARALLEL) as ex:
        futs = {ex.submit(_worker, args): args[0] for args in args_list}
        for fut in as_completed(futs):
            a = futs[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {"asset": a, "status": "exception", "error": str(e)[:200]}
            if r.get("status") == "ok":
                all_data[a] = r["data"]
                print(f"  >>> [{len(all_data)}/{len(ALL_ASSETS)}] {a:<10} "
                      f"{r['n_seqs']:>5} seqs ({r['elapsed']:.1f}s)", flush=True)
            else:
                print(f"  >>> [{a}] {r.get('status', '?')} ({r.get('elapsed', 0):.1f}s)", flush=True)

    elapsed_total = time.time() - t0
    print(f"\n=== PREPARE done in {elapsed_total/60:.1f} min ===")

    if not all_data:
        print("!! Aucune data collectee")
        return

    # Concat
    m1_list, m15_list, h1_list = [], [], []
    ict_list, aid_list, lbl_list, pnl_list, ts_list = [], [], [], [], []
    for asset, d in all_data.items():
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
    ts_arr = np.concatenate([pd.DatetimeIndex(t).values.astype("datetime64[ns]") for t in ts_list])

    print(f"\n=== TOTAL SEQUENCES ===")
    print(f"  m1_seqs   : {m1.shape}")
    print(f"  m15_seqs  : {m15.shape}")
    print(f"  h1_seqs   : {h1.shape}")
    print(f"  ict_feats : {ict.shape}")
    print(f"  labels    : {lbl.shape} (WIN={int((lbl==1).sum())}, LOSS={int((lbl==0).sum())})")

    out_path = Path(f"{ROOT}/data/v19_sequences.npz")
    np.savez_compressed(out_path,
                         m1_seqs=m1, m15_seqs=m15, h1_seqs=h1,
                         ict_feats=ict, asset_ids=aid, labels=lbl,
                         pnl_Rs=pnl, timestamps=ts_arr)
    size_mb = out_path.stat().st_size / 1e6
    print(f"\nSaved : {out_path.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
