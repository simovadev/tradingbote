"""Build dataset multi-TF pour XAUUSD : M5 et M15 (M30 optionnel).

M1 deja fait avec v7 (ml_dataset.parquet actuel).
Ici on construit M5 et M15 en parallele.
"""
import sys
sys.path.insert(0, 'c:/Users/Shadow/TradingBot')

import time
import pandas as pd
from bot_v2.ml_dataset import build_dataset
from bot_v2.data_loader import load


def build_for_tf(ltf: str, years: int = 5):
    print(f"\n{'='*60}")
    print(f"  BUILD XAUUSD pour {ltf} ({years} ans)")
    print(f"{'='*60}\n", flush=True)

    df = load("XAUUSD", ltf)
    end = df.index[-1]
    # On prend max(years, data_disponible)
    requested_start = end - pd.Timedelta(days=365 * years)
    start = max(df.index[0], requested_start)
    nyears_real = (end - start).days / 365.25
    print(f"Plage : {start.date()} -> {end.date()} ({nyears_real:.1f} ans, {len(df)} bougies)", flush=True)

    t0 = time.time()
    build_dataset(start, end, instruments=["XAUUSD"], chunk_months=3, ltf=ltf)
    elapsed = time.time() - t0
    print(f"\n>>> {ltf} fini en {elapsed/60:.1f} min", flush=True)


if __name__ == "__main__":
    years = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    # M5 uniquement (user 2026-05-16)
    build_for_tf("M5", years=years)
    print("\nDone. Dataset sauve :")
    print("  c:/Users/Shadow/TradingBot/data/ml_dataset_M5.parquet")
