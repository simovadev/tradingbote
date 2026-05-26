"""Wrapper PARALLEL pour train_v18_7_vantage : lance 8 actifs en parallele.

Au lieu de :
  python -m bot_v2.train_v18_7_vantage --all  (1 actif a la fois, 30-40 min)
On fait :
  8 actifs en parallele via ProcessPoolExecutor
  Chaque process fait son train (core + full + WF CV) avec n_jobs=8
  Total cores actifs : 8 actifs * 8 jobs/lgb = 64 cores
  -> ~5-8 min total au lieu de 30-40 min
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = "c:/Users/Shadow/TradingBot" if sys.platform == "win32" else "/workspace/TradingBot"
sys.path.insert(0, ROOT)

ALL_ASSETS = [
    "BTCUSD", "ETHUSD", "USDMXN", "XAUUSD", "USDJPY", "USDZAR", "USDCAD", "NZDUSD",
    "GBPUSD", "CL-OIL", "EURUSD", "AUDUSD", "USDCHF", "NAS100", "XAGUSD",
    "DJ30", "GAS-C", "HK50", "GER40", "FRA40", "UK100", "Nikkei225",
    "BVSPX", "SP500", "Coffee-C", "Cocoa-C", "Wheat-C", "Sugar-C",
]

N_PARALLEL = int(os.environ.get("V18_7_PARALLEL", "6"))


def _worker(asset: str) -> dict:
    """Train 1 actif (re-importe pour avoir un process isole)."""
    # Limite threads par process pour ne pas exploser
    # 6 actifs * 4 threads = 24 cores actifs, sans thrashing
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["LIGHTGBM_EXEC_THREADS"] = "4"

    try:
        from bot_v2.train_v18_7_vantage import train_one
        t0 = time.time()
        result = train_one(asset)
        result["elapsed_sec"] = round(time.time() - t0, 1)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"asset": asset, "status": "exception", "error": str(e)[:300]}


def main():
    print(f"=== TRAIN V18.7 PARALLEL : {len(ALL_ASSETS)} actifs, {N_PARALLEL} en parallele ===")
    t0 = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=N_PARALLEL) as ex:
        futs = {ex.submit(_worker, a): a for a in ALL_ASSETS}
        for fut in as_completed(futs):
            a = futs[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {"asset": a, "status": "exception", "error": str(e)[:300]}
            results.append(r)
            verdict = r.get("verdict", r.get("status", "?"))
            elapsed = r.get("elapsed_sec", "?")
            chosen = r.get("chosen", "?")
            bm = r.get("metrics_best", {})
            auc_oos = bm.get("auc_oos", 0)
            wf = bm.get("wf_auc_mean", 0)
            gap = bm.get("gap_train_oos", 0)
            wr = bm.get("oos_wr@0.65", 0)
            print(f"  >>> [{len(results)}/{len(ALL_ASSETS)}] {a:<10} "
                  f"{verdict:<6} "
                  f"chosen={chosen:<5} "
                  f"AUC={auc_oos:.3f} WF={wf:.3f} gap={gap:+.3f} "
                  f"WR@65={wr:.1f}% ({elapsed}s)", flush=True)

    elapsed_total = (time.time() - t0) / 60
    print(f"\n=== TERMINE en {elapsed_total:.1f} min ===")

    # Recap final
    print(f"\n{'='*110}")
    print(f"=== RECAP V18.7 PARALLEL ===")
    print(f"{'='*110}")
    print(f"{'Asset':<11}{'Chosen':<8}{'AUC oos':<9}{'WF AUC':<9}{'GAP':<9}{'WR@65':<8}{'N@65':<7}{'Verdict':<8}")
    for r in sorted(results, key=lambda x: -(x.get("metrics_best", {}).get("auc_oos", 0))):
        if r.get("status") not in ("trained",):
            print(f"{r['asset']:<11}{r.get('status', '?'):<8}")
            continue
        bm = r.get("metrics_best", {})
        print(f"{r['asset']:<11}"
              f"{r.get('chosen', '?'):<8}"
              f"{bm.get('auc_oos', 0):<9.3f}"
              f"{bm.get('wf_auc_mean', 0):<9.3f}"
              f"{bm.get('gap_train_oos', 0):<+9.3f}"
              f"{bm.get('oos_wr@0.65', 0):<8.1f}"
              f"{bm.get('oos_n@0.65', '--')!s:<7}"
              f"{r.get('verdict', '--'):<8}")
    n_pass = sum(1 for r in results if r.get("verdict") == "PASS")
    n_core = sum(1 for r in results if r.get("chosen") == "core")
    n_full = sum(1 for r in results if r.get("chosen") == "full")
    print(f"\nPASS : {n_pass} / {len(results)}")
    print(f"ABLATION : {n_core} actifs preferent CORE (drift-free), {n_full} preferent FULL")

    recap = Path(f"{ROOT}/ml_metrics_v18_7_recap.json")
    recap.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nRecap : {recap.name}")


if __name__ == "__main__":
    main()
