"""Build dataset V10 TEST : 1 an, 3 actifs, pour iteration rapide.

V10 = V9 + nouvelles features (po3 enrichi, momentum, displacement, atr_regime)
+ swing_strength=1 (plus d'OB) + RR configurable via RR_OVERRIDE.

Usage : python -m bot_v2.build_v10_test
        RR_OVERRIDE=1.5 python -m bot_v2.build_v10_test
"""
from __future__ import annotations
import os, sys
from pathlib import Path

os.environ.setdefault("BUILD_DATA_DIR", "data_vantage")
ROOT = "c:/Users/Shadow/TradingBot" if sys.platform == "win32" else "/workspace/TradingBot"
sys.path.insert(0, ROOT)

import pandas as pd
from bot_v2.data_loader import load
from bot_v2.ml_dataset import build_dataset

# Test : 3 actifs representatifs, 1 an
TEST_ASSETS = ["XAUUSD", "EURUSD", "NAS100"]
TRAIN_START = pd.Timestamp("2025-05-22", tz="UTC")
TRAIN_END = pd.Timestamp("2026-05-22", tz="UTC")


def main():
    rr = os.getenv("RR_OVERRIDE", "2.0")
    suffix = f"_V10TEST_RR{rr.replace('.','')}"
    print(f"BUILD V10 TEST | RR={rr} | suffix={suffix}", flush=True)
    for asset in TEST_ASSETS:
        df = load(asset, "M1")
        ts = max(TRAIN_START, df.index[0])
        te = min(TRAIN_END, df.index[-1])
        print(f"\n=== {asset} V10 TEST ({ts.date()} -> {te.date()}) ===", flush=True)
        out = Path(f"{ROOT}/data/ml_dataset_{asset}_v10test_rr{rr.replace('.','')}.parquet")
        cpu = os.cpu_count() or 4
        chunk = 0.25 if cpu >= 64 else 1.0
        dfr = build_dataset(ts, te, [asset], output_path=out,
                            chunk_months=chunk, ltf="M1", version_suffix=suffix)
        if len(dfr) and "outcome" in dfr.columns:
            closed = dfr[dfr["outcome"].isin(["WIN", "LOSS"])]
            wr = (closed["outcome"] == "WIN").mean() * 100 if len(closed) else 0
            print(f">>> {asset} V10 TEST : {len(dfr):,} candidats, {len(closed):,} fermes, WR brut {wr:.1f}%", flush=True)


if __name__ == "__main__":
    main()
