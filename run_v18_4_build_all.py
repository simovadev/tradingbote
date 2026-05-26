"""V18.4 BUILD FINAL — TOUS FIXES + configs par actif.

Corrections incluses (vs V18.3) :
- BUG #1 : score/quality retires des features ML
- BUG #3 : calibration isotonique (cote train, pas ici)
- BUG #4 : embargo 2j entre splits (cote train, pas ici)
- BUG #5 : configs genetique recalculees avec WINDOW < 2025-05-22 (cote genetic)
- BUG #6 : AMBIGUOUS filtre dans le dataset
- BONUS : hour_sin/hour_cos ajoutes

Build : pour chaque actif, charge sa config champion depuis JSON et build 8 ans.
Output : data/ml_partial_M1_V18_4_FIXED/<asset>_M1_<date>.parquet
"""
import os
import sys
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# 28 actifs (BTCUSD inclus, on rebuild tout proprement)
ASSETS = [
    "BTCUSD", "ETHUSD", "USDMXN", "XAUUSD", "USDJPY", "USDZAR", "USDCAD", "NZDUSD",
    "GBPUSD", "CL-OIL", "EURUSD", "AUDUSD", "USDCHF", "NAS100", "XAGUSD",
    "DJ30", "GAS-C", "HK50", "GER40", "FRA40", "UK100", "Nikkei225",
    "BVSPX", "SP500", "Coffee-C", "Cocoa-C", "Wheat-C", "Sugar-C",
]

TRAIN_START = "2018-03-01"
TRAIN_END = "2026-05-22"


def load_config(asset: str) -> dict:
    """Charge la config champion depuis genetic_v4_<ASSET>_results.json."""
    p = ROOT / f"genetic_v4_{asset}_results.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    best = data.get("best_robust", {})
    return best.get("config", {}) if best else None


def build_one(asset: str, cfg: dict) -> dict:
    """Build dataset 8 ans pour 1 actif avec sa config."""
    print(f"\n{'='*70}", flush=True)
    print(f"=== ACTIF : {asset} ===", flush=True)
    print(f"{'='*70}", flush=True)
    cfg_str = ' '.join(f'{k}={v}' for k, v in sorted(cfg.items()))
    print(f"Config: {cfg_str}", flush=True)

    env = os.environ.copy()
    env["BUILD_DATA_DIR"] = "data_vantage"
    env["BUILD_V12_MODE"] = "1"
    env["OB_VALIDATION_MODE"] = "v18"
    env["N_WORKERS"] = "128"
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    for k, v in cfg.items():
        env[k] = str(v)

    t0 = time.time()
    build_script = f"""
import os, sys
sys.path.insert(0, '{ROOT.as_posix()}')
import pandas as pd
from bot_v2.ml_dataset import build_dataset
df = build_dataset(
    pd.Timestamp('{TRAIN_START}', tz='UTC'),
    pd.Timestamp('{TRAIN_END}', tz='UTC'),
    ['{asset}'],
    output_path='{(ROOT / "data" / f"ml_dataset_{asset}_vantage_v18_4.parquet").as_posix()}',
    chunk_months=0.25, ltf='M1',
    version_suffix='_V18_4_FIXED',
)
print(f'BUILD {asset}: {{len(df):,}} trades')
if "outcome" in df.columns:
    print(df["outcome"].value_counts().to_string())
"""
    try:
        result = subprocess.run(
            ["python3", "-c", build_script],
            env=env, timeout=2400,  # 40 min max par actif
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        elapsed = time.time() - t0
        out_lines = result.stdout.split("\n")
        # Extract n trades
        n_trades = 0
        for line in out_lines:
            if line.startswith("BUILD "):
                try:
                    n_trades = int(line.split(": ")[1].replace(",", "").split(" ")[0])
                except Exception:
                    pass
        print(f"--- {asset} fini en {elapsed/60:.1f} min : {n_trades:,} trades ---", flush=True)
        # Save tail log
        log_path = ROOT / f"build_v18_4_{asset}.log"
        log_path.write_text("\n".join(out_lines[-20:]))
        return {"asset": asset, "n_trades": n_trades, "elapsed_min": elapsed/60, "status": "ok"}
    except subprocess.TimeoutExpired:
        return {"asset": asset, "status": "timeout"}
    except Exception as e:
        return {"asset": asset, "status": "error", "error": str(e)[:200]}


def main():
    print(f"=== V18.4 BUILD FINAL : {len(ASSETS)} actifs ===")
    print(f"Period : {TRAIN_START} -> {TRAIN_END}")
    print(f"Fixes : score/quality OUT, AMBIGUOUS in dataset, hour_sin/cos features")
    t_global = time.time()
    results = []

    for i, asset in enumerate(ASSETS, 1):
        cfg = load_config(asset)
        if cfg is None:
            print(f"\n>>> [{i}/{len(ASSETS)}] {asset} : [SKIP] pas de config")
            continue
        print(f"\n>>> [{i}/{len(ASSETS)}] {asset}", flush=True)
        try:
            r = build_one(asset, cfg)
        except Exception as e:
            r = {"asset": asset, "status": "exception", "error": str(e)[:200]}
        results.append(r)
        # Sauvegarde incrémentale
        (ROOT / "v18_4_build_results.json").write_text(
            json.dumps(results, indent=2, default=str)
        )

    print(f"\n{'='*70}")
    print(f"=== TERMINE en {(time.time()-t_global)/60:.1f} min ===")
    print(f"{'='*70}")

    print(f"\n=== STATUS BUILD V18.4 ===")
    print(f"{'Asset':<12}{'Status':<12}{'Trades':<10}{'Time (min)'}")
    print("-" * 50)
    total_trades = 0
    for r in results:
        n = r.get('n_trades', 0)
        total_trades += n
        print(f"  {r['asset']:<10}{r.get('status','?'):<12}{n:<10,}{r.get('elapsed_min','?')}")
    print(f"\nTOTAL trades : {total_trades:,}")


if __name__ == "__main__":
    main()
