"""Benchmark : trouve le nombre max de workers ProcessPool utilisable.

Methode :
1. Lance N workers ProcessPool avec une tache CPU bound courte (~1s)
2. Mesure le throughput (tasks/s)
3. Repeat avec N croissant : 32, 64, 128, 200, 300, 400, 500, 600
4. Le throughput plateau (ou diminue) = limite atteinte

Usage:
    ulimit -n 65536 && python3 bench_max_workers.py
"""
from __future__ import annotations
import multiprocessing as mp
import time
import sys
import os


def cpu_task(n: int) -> int:
    """Tache CPU bound : ~0.5-1s de calcul."""
    s = 0
    for i in range(n):
        s += (i * 7919) % 991  # nombre premier random
    return s


def bench_workers(n_workers: int, n_tasks: int = None, work_size: int = 5_000_000) -> dict:
    """Lance n_workers et fait n_tasks taches de 'work_size'. Retourne timing."""
    if n_tasks is None:
        n_tasks = n_workers * 4  # 4 taches par worker pour bien repartir

    t0 = time.time()
    with mp.Pool(n_workers) as pool:
        results = pool.map(cpu_task, [work_size] * n_tasks)
    elapsed = time.time() - t0
    return {
        "n_workers": n_workers,
        "n_tasks": n_tasks,
        "elapsed_s": elapsed,
        "tasks_per_s": n_tasks / elapsed,
        "tasks_per_s_per_worker": (n_tasks / elapsed) / n_workers,
    }


def main():
    print(f"=== BENCH MAX WORKERS ===")
    print(f"CPU cores (nproc) : {mp.cpu_count()}")
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        print(f"FD limit (soft/hard) : {soft}/{hard}")
    except ImportError:
        pass
    print()

    # Configurations croissantes
    configs = [32, 64, 128, 200, 300, 400, 500, 600, 800, 1000]
    results = []
    prev_throughput = 0

    for n in configs:
        try:
            print(f"--- {n} workers ---", flush=True)
            r = bench_workers(n)
            results.append(r)
            print(f"  elapsed       : {r['elapsed_s']:.2f}s")
            print(f"  tasks/s total : {r['tasks_per_s']:.1f}")
            print(f"  tasks/s/worker: {r['tasks_per_s_per_worker']:.3f}")
            # Stop si throughput plateau (gain < 5%)
            if prev_throughput > 0:
                gain = (r['tasks_per_s'] - prev_throughput) / prev_throughput * 100
                print(f"  gain vs prec  : {gain:+.1f}%")
                if gain < 5 and n >= 200:
                    print(f"\n  → Plateau atteint a {n} workers (gain < 5%)")
                    break
            prev_throughput = r['tasks_per_s']
            print()
        except OSError as e:
            print(f"  ERREUR OS : {e}")
            print(f"  → Limite OS atteinte a {n} workers (cause probable: fd, mem)")
            break
        except Exception as e:
            print(f"  ERREUR : {e}")
            break

    # === Resume ===
    print(f"\n{'='*60}\n=== RESUME ===")
    print(f"{'N':>5}  {'elapsed_s':>10}  {'tasks/s':>10}  {'per_worker':>10}")
    for r in results:
        print(f"{r['n_workers']:>5}  {r['elapsed_s']:>10.2f}  {r['tasks_per_s']:>10.1f}  {r['tasks_per_s_per_worker']:>10.3f}")
    if results:
        best = max(results, key=lambda r: r['tasks_per_s'])
        print(f"\nBEST : {best['n_workers']} workers, {best['tasks_per_s']:.1f} tasks/s")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
