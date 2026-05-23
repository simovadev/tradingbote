"""Compare LIMIT vs MARKET 30s sur 5 jours (14, 15, 18, 19, 20 mai 2026).

Lance 2 scenarios (LIMIT et MARKET 30s) en parallele sur Vast.
Chaque scenario backteste 14 actifs x 5 jours en mode tick.

Resultat : qui gagne sur 5 jours d'echantillon ? Le MARKET tient-il
ou c'etait juste le 19/05 qui etait favorable ?
"""
import subprocess
import sys
import time
import os

ROOT = "/workspace/TradingBot"
DATES = ["2026-05-14", "2026-05-15", "2026-05-18", "2026-05-19", "2026-05-20"]
SCENARIOS = [
    ("limit_5j",    "limit",  0.0,  0.07),
    ("market30_5j", "market", 30.0, 0.07),
]


def launch_one_day(name, mode, lat, comm, date):
    log_path = f"/workspace/bt_{name}_{date}.log"
    cmd = [
        "python3", "-u", f"{ROOT}/backtest_v12_tick_vast.py",
        "--date", date,
        "--scan_hours", "2",
        "--workers", "64",
        "--step", "5",
        "--entry_mode", mode,
        "--latency_s", str(lat),
        "--commission_r", str(comm),
    ]
    log = open(log_path, "w")
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), log_path


def main():
    print(f"=== COMPARE 5J : LIMIT vs MARKET 30s ===")
    print(f"Dates : {DATES}")
    print()

    # Lance les 10 (2 scenarios x 5 jours) EN SEQUENCE (chaque run prend ~3 min)
    # mais on parallelise LIMIT et MARKET pour le meme jour (2 en parallele = 128 cores)
    all_procs = []
    for date in DATES:
        print(f"--- Jour {date} ---")
        day_procs = []
        for name, mode, lat, comm in SCENARIOS:
            p, log = launch_one_day(name, mode, lat, comm, date)
            day_procs.append((p, log, f"{name}_{date}"))
            print(f"  Launched {name}_{date} (PID {p.pid})")
        # Attendre que les 2 du meme jour finissent avant de lancer le suivant
        for p, log, n in day_procs:
            p.wait()
            print(f"  {n} : DONE")
        all_procs.extend(day_procs)

    print()
    print("=" * 80)
    print("=== RECAP COMPARATIF 5 JOURS ===")
    print("=" * 80)
    import re

    def parse_log(log_path):
        try:
            with open(log_path) as f:
                content = f.read()
            idx = content.find("RECAP GLOBAL TICK")
            if idx == -1:
                return None
            section = content[idx:idx+800]
            def grab(key):
                m = re.search(rf"{key}\s*:\s*([+\-\d.]+)", section)
                return m.group(1) if m else "?"
            return {
                "fermes": grab("Trades fermes"),
                "wr": grab("WR"),
                "pnl": grab(r"PnL \(R\)"),
                "invalid": grab("INVALID_PRICE"),
                "nofill": grab("NO_FILL"),
            }
        except Exception:
            return None

    # Tableau jour par jour
    print(f"{'JOUR':<12}{'MODE':<10}{'FERMES':<8}{'WR':<8}{'PnL(R)':<10}{'INVALID':<10}{'NO_FILL'}")
    print("-" * 80)
    totals = {"limit_5j": {"fermes": 0, "pnl": 0.0}, "market30_5j": {"fermes": 0, "pnl": 0.0}}
    for date in DATES:
        for name, _, _, _ in SCENARIOS:
            log = f"/workspace/bt_{name}_{date}.log"
            r = parse_log(log)
            if r is None:
                print(f"{date:<12}{name:<10}NO_RESULT")
                continue
            print(f"{date:<12}{name:<10}{r['fermes']:<8}{r['wr']:<8}{r['pnl']:<10}{r['invalid']:<10}{r['nofill']}")
            try:
                totals[name]["fermes"] += int(float(r["fermes"]))
                totals[name]["pnl"] += float(r["pnl"])
            except Exception:
                pass

    print("-" * 80)
    print("\n=== TOTAUX SUR 5 JOURS ===")
    for name in ("limit_5j", "market30_5j"):
        t = totals[name]
        print(f"  {name:<14} : {t['fermes']} trades, PnL total = {t['pnl']:+.1f}R")
    diff = totals["market30_5j"]["pnl"] - totals["limit_5j"]["pnl"]
    print(f"\n  Avantage MARKET : {diff:+.1f}R")
    if diff > 0:
        print(f"  => MARKET GAGNE de {diff/abs(totals['limit_5j']['pnl'])*100 if totals['limit_5j']['pnl']!=0 else 0:.0f}% sur 5 jours")
    else:
        print(f"  => LIMIT gagne, MARKET etait juste favorable le 19/05")


if __name__ == "__main__":
    main()
