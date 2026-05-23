"""Compare LIMIT vs MARKET (avec differentes latences) sur le 19/05 en tick.

Lance 4 scenarios en parallele via subprocess :
1. LIMIT, latence 0 (baseline)
2. MARKET, latence 5s
3. MARKET, latence 18s
4. MARKET, latence 30s

Commission 7% (0.07R) appliquee partout.
Chaque scenario tourne avec 64 workers (4*64=256 cores utilises sur 512).
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = "/workspace/TradingBot"

SCENARIOS = [
    ("limit_lat0_c07",   "limit",  0.0,  0.07),
    ("market_lat5_c07",  "market", 5.0,  0.07),
    ("market_lat18_c07", "market", 18.0, 0.07),
    ("market_lat30_c07", "market", 30.0, 0.07),
]


def launch(name, mode, latency, commission):
    log_path = f"/workspace/bt_{name}.log"
    cmd = [
        "python3", "-u", f"{ROOT}/backtest_v12_tick_vast.py",
        "--date", "2026-05-19",
        "--scan_hours", "2",
        "--workers", "64",   # 64 workers x 4 scenarios = 256 cores
        "--step", "5",
        "--entry_mode", mode,
        "--latency_s", str(latency),
        "--commission_r", str(commission),
    ]
    log = open(log_path, "w")
    import os
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    print(f"  Launched {name} (PID {p.pid}) -> {log_path}", flush=True)
    return p, log_path, name


def main():
    print(f"=== COMPARE ENTRY MODES (4 scenarios en parallele) ===")
    procs = []
    for name, mode, lat, com in SCENARIOS:
        procs.append(launch(name, mode, lat, com))
        time.sleep(2)
    print()

    print("Attente fin des 4 scenarios...")
    for p, log_path, name in procs:
        p.wait()
        print(f"  {name} : DONE")

    print()
    print("=" * 70)
    print("=== RECAP COMPARATIF ===")
    print("=" * 70)
    print(f"{'SCENARIO':<22}{'FERMES':<8}{'WR':<8}{'PnL(R)':<10}{'INVALID':<10}{'NO_FILL'}")
    print("-" * 70)
    for _, _, name in procs:
        log_path = f"/workspace/bt_{name}.log"
        try:
            with open(log_path) as f:
                content = f.read()
            # Extrait depuis "RECAP GLOBAL TICK"
            idx = content.find("RECAP GLOBAL TICK")
            if idx == -1:
                print(f"{name:<22}NO_RESULT"); continue
            section = content[idx:idx+800]
            def grab(key, default="?"):
                import re
                m = re.search(rf"{key}\s*:\s*([+\-\d.,]+)", section)
                return m.group(1) if m else default
            fermes = grab("Trades fermes")
            wr = grab("WR")
            pnl = grab("PnL .R.")
            invalid = grab("INVALID_PRICE")
            nofill = grab("NO_FILL")
            print(f"{name:<22}{fermes:<8}{wr:<8}{pnl:<10}{invalid:<10}{nofill}")
        except Exception as e:
            print(f"{name:<22}ERR {e}")


if __name__ == "__main__":
    main()
