"""Bench multi-threade : trouve la combo (workers, threads_par_worker) optimale.

Methode :
- Lance une matrice (workers, threads) sur une tache representative du backtest
- Tache = operations numpy lourdes (similaire a find_swings, detect_fvg, ATR)
- Mesure throughput
- Identifie la config qui sature le mieux les 512 cores

Configs testees :
  workers x threads = (32x16, 64x8, 128x4, 256x2, 64x4, 128x2, etc.)
"""
from __future__ import annotations
import multiprocessing as mp
import numpy as np
import os
import time
import sys


def numpy_intensive_task(args):
    """Tache numpy intensive (proche de detect_fvg + find_swings + ATR).
    Les threads via OMP_NUM_THREADS doivent accelerer si numpy est multithread.
    """
    seed, size = args
    rng = np.random.default_rng(seed)
    arr = rng.normal(size=size).astype(np.float64)
    # Operations vectorisees (numpy peut multithreader)
    # 1. Resample-like (rolling mean)
    win = 100
    weights = np.ones(win) / win
    out1 = np.convolve(arr, weights, mode='valid')
    # 2. Sliding window max (like find_swings)
    from numpy.lib.stride_tricks import sliding_window_view
    sw = sliding_window_view(arr, 50)
    out2 = sw.max(axis=1)
    out3 = sw.min(axis=1)
    # 3. Lots of arithmetic
    out4 = np.sin(arr) * np.cos(arr * 2) + np.exp(arr / 100)
    # 4. Reduction
    s = float(out1.sum() + out2.sum() + out3.sum() + out4.sum())
    return s


def init_worker(n_threads):
    """Set thread env in each worker process at startup."""
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n_threads)


def bench_combo(n_workers: int, n_threads: int, n_tasks: int = None, size: int = 2_000_000) -> dict:
    if n_tasks is None:
        n_tasks = n_workers * 4
    t0 = time.time()
    with mp.Pool(n_workers, initializer=init_worker, initargs=(n_threads,)) as pool:
        results = pool.map(numpy_intensive_task, [(i, size) for i in range(n_tasks)])
    elapsed = time.time() - t0
    total_cores = n_workers * n_threads
    return {
        "workers": n_workers,
        "threads": n_threads,
        "total_cores": total_cores,
        "n_tasks": n_tasks,
        "elapsed_s": elapsed,
        "tasks_per_s": n_tasks / elapsed,
    }


def main():
    print(f"=== BENCH WORKERS x THREADS ===")
    print(f"CPU cores (nproc) : {mp.cpu_count()}")
    print()

    # Combos a tester : 4 niveaux de workers x 4 niveaux de threads
    combos = [
        # (workers, threads)  -> total_cores
        (32, 1),    # 32   - baseline mono
        (64, 1),    # 64   - mono
        (128, 1),   # 128  - mono actuel
        (256, 1),   # 256  - mono surdimensionne
        (32, 4),    # 128  - threadable
        (64, 4),    # 256  - threadable
        (128, 4),   # 512  - max theorique
        (32, 8),    # 256  - gros threads
        (64, 8),    # 512  - max via threads
        (128, 2),   # 256  - balance
        (256, 2),   # 512  - surcharge
    ]

    results = []
    for w, t in combos:
        print(f"--- workers={w} threads={t} (total={w*t}) ---", flush=True)
        try:
            r = bench_combo(w, t, n_tasks=w*4)
            results.append(r)
            print(f"  elapsed       : {r['elapsed_s']:.2f}s")
            print(f"  tasks/s       : {r['tasks_per_s']:.1f}")
            print()
        except Exception as e:
            print(f"  ERREUR : {e}")
            break

    # Tri par throughput
    results.sort(key=lambda r: -r['tasks_per_s'])

    print(f"\n{'='*70}")
    print(f"=== RESUME (trie par tasks/s) ===")
    print(f"{'workers':>8} {'threads':>8} {'total_c':>8} {'tasks/s':>10} {'elapsed':>9}")
    for r in results:
        print(f"{r['workers']:>8} {r['threads']:>8} {r['total_cores']:>8} {r['tasks_per_s']:>10.1f} {r['elapsed_s']:>9.2f}")

    best = results[0]
    print(f"\nBEST : workers={best['workers']} threads={best['threads']} = {best['total_cores']} cores")
    print(f"       {best['tasks_per_s']:.1f} tasks/s en {best['elapsed_s']:.1f}s")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
