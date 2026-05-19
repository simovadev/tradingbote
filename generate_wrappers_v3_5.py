"""Genere les wrappers V3.5 pour les 13 actifs.

Output : bot_v2/ml_dataset_{ASSET}_admiral_8ans_V3_5.py x13
"""
from pathlib import Path

ASSETS = [
    "NAS100", "SP500", "DJ30",
    "GER40", "UK100", "FRA40", "JP225",
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF",
    "BTCUSD",
]

TEMPLATE = '''"""Build dataset ML {ASSET} V3.5 - Admiral 8 ans, filtres relaches.

Auto-genere depuis generate_wrappers_v3_5.py
"""
from __future__ import annotations

import os
import sys

ROOT = "c:/Users/Shadow/TradingBot" if sys.platform == "win32" else "/workspace/TradingBot"
sys.path.insert(0, ROOT)

from pathlib import Path
import pandas as pd

from bot_v2.data_loader import load
from bot_v2.ml_dataset import build_dataset


def main():
    instruments = ["{ASSET}"]
    df = load("{ASSET}", "M1")
    earliest_end = df.index[-1]
    earliest_start = df.index[0]

    cpu_count = os.cpu_count() or 4
    chunk_months = 1 if cpu_count >= 64 else 3

    print(f"=== ML DATASET {ASSET} V3.5 ===")
    print(f"Plage : {{earliest_start.date()}} -> {{earliest_end.date()}}")
    print(f"Bougies M1 : {{len(df):,}}")
    print(f"Annees : {{(earliest_end - earliest_start).days / 365.25:.2f}}")
    print(f"Cores : {{cpu_count}} -> chunks de {{chunk_months}} mois")
    print()

    output_path = Path(f"{{ROOT}}/data/ml_dataset_{ASSET}_admiral_8ans_V3_5.parquet")
    build_dataset(
        earliest_start, earliest_end, instruments,
        output_path=output_path,
        chunk_months=chunk_months,
        ltf="M1",
    )


if __name__ == "__main__":
    main()
'''

OUTPUT_DIR = Path("c:/Users/Shadow/TradingBot/bot_v2")
for asset in ASSETS:
    content = TEMPLATE.format(ASSET=asset)
    path = OUTPUT_DIR / f"ml_dataset_{asset}_admiral_8ans_V3_5.py"
    path.write_text(content, encoding="utf-8")
    print(f"OK : {path.name}")

print(f"\n{len(ASSETS)} wrappers generes")
