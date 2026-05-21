"""Visualiser les logs du bot live.

Le bot ecrit tout en INFO dans live.log mais n'affiche que les messages
IMPORTANTS dans la console pour ne pas saturer (RAM/IO).

Ce script permet de :
- Suivre en temps reel TOUS les logs (mode follow)
- Filtrer par actif, type, niveau
- Voir les derniers N rejets ML
- Afficher la latence par actif sur le dernier cycle

Usage :
    python tools/view_logs.py                  # tail -f en temps reel
    python tools/view_logs.py --last 100       # 100 dernieres lignes
    python tools/view_logs.py --asset XAUUSD   # filtre par actif
    python tools/view_logs.py --setups         # seulement les SETUP
    python tools/view_logs.py --trades         # seulement les trades pris
    python tools/view_logs.py --rejects        # seulement les rejets ML
    python tools/view_logs.py --latency        # tableau latence dernier cycle
    python tools/view_logs.py --errors         # seulement WARNING+
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "live.log"


def open_log():
    if not LOG_PATH.exists():
        print(f"!! Log absent : {LOG_PATH}")
        print("   Lance le bot d'abord : python -m bot_v2.live_runner_v2")
        raise SystemExit(1)
    return LOG_PATH.open("r", encoding="utf-8", errors="replace")


def stream_filtered(match_fn, follow: bool = True):
    """Affiche les lignes qui matchent match_fn. Si follow=True, tail -f."""
    f = open_log()
    # Va a la fin du fichier
    f.seek(0, 2)
    try:
        if not follow:
            # On affiche les dernieres lignes seulement
            return
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.5)
                continue
            if match_fn(line):
                print(line.rstrip())
    except KeyboardInterrupt:
        pass
    finally:
        f.close()


def last_n_filtered(n: int, match_fn) -> list[str]:
    """Retourne les N dernieres lignes du log qui matchent match_fn."""
    with open_log() as f:
        all_lines = [l for l in f if match_fn(l)]
    return all_lines[-n:]


def tail_follow(match_fn=lambda l: True, n_first: int = 50) -> None:
    """Affiche les N dernieres lignes filtrees, puis suit le fichier (tail -f)."""
    # 1) Affiche N dernieres lignes deja presentes
    initial = last_n_filtered(n_first, match_fn)
    for l in initial:
        print(l.rstrip())
    if initial:
        print("--- live tail (Ctrl+C pour quitter) ---")
    # 2) Suit le fichier
    f = open_log()
    f.seek(0, 2)
    try:
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.5)
                continue
            if match_fn(line):
                print(line.rstrip(), flush=True)
    except KeyboardInterrupt:
        print("\n(arret manuel)")
    finally:
        f.close()


# ============ FILTRES ============

def f_all(line: str) -> bool:
    return True


def f_asset(asset: str):
    asset_up = asset.upper()
    def _filter(line: str) -> bool:
        return asset_up in line
    return _filter


def f_setups(line: str) -> bool:
    return "SETUP " in line or "PENDING ORDER" in line


def f_trades(line: str) -> bool:
    return ("PENDING ORDER" in line or "TRADE CLOSED" in line or
            "LIMIT OK" in line or "LIMIT REJECTED" in line)


def f_rejects(line: str) -> bool:
    return "ml_below_thr" in line or "rejete" in line.lower()


def f_errors(line: str) -> bool:
    return "[WARNING]" in line or "[ERROR]" in line or "exception" in line.lower()


def f_cycles(line: str) -> bool:
    return "CYCLE scan" in line


def f_stats(line: str) -> bool:
    return "STATS |" in line


# ============ VUES SPECIALES ============

def show_latency_last_cycle() -> None:
    """Extrait la latence par actif du dernier cycle complet."""
    with open_log() as f:
        lines = f.readlines()

    # Cherche le dernier "CYCLE scan" et reprend les LATENCY juste avant
    cycle_idx = None
    for i in range(len(lines) - 1, -1, -1):
        if "CYCLE scan" in lines[i]:
            cycle_idx = i
            break

    if cycle_idx is None:
        print("Aucun cycle scan trouve dans le log.")
        return

    # Cherche les LATENCY remontees (juste avant le CYCLE scan)
    latencies = {}
    pattern = re.compile(r"LATENCY (\w+): total=(\d+)ms.*cache_build=(\d+)ms")
    for j in range(max(0, cycle_idx - 40), cycle_idx):
        m = pattern.search(lines[j])
        if m:
            latencies[m.group(1)] = (int(m.group(2)), int(m.group(3)))

    print(f"\n=== LATENCE DERNIER CYCLE ===")
    print(lines[cycle_idx].rstrip())
    print()
    print(f"{'Actif':<10} {'Total':<10} {'Cache build':<12}")
    print("-" * 35)
    for asset, (total, cache) in sorted(latencies.items(), key=lambda x: -x[1][0]):
        marker = "*" if total > 8000 else " "
        print(f"{asset:<10} {total:>5} ms {marker} {cache:>5} ms")


def show_recent_setups(n: int = 20) -> None:
    """Affiche les N derniers setups detectes."""
    pattern_setup = re.compile(r"SETUP (\w+)")
    pattern_pending = re.compile(r"PENDING ORDER (\w+)")
    with open_log() as f:
        lines = [l for l in f if pattern_setup.search(l) or pattern_pending.search(l)]
    print(f"\n=== {min(n, len(lines))} derniers SETUPS / ORDERS ===")
    for l in lines[-n:]:
        print(l.rstrip())


def show_recent_rejets(n: int = 30) -> None:
    """Affiche les N derniers rejets ML."""
    with open_log() as f:
        lines = [l for l in f if "ml_below_thr" in l or "probas_ML=" in l]
    print(f"\n=== {min(n, len(lines))} derniers REJETS ML ===")
    for l in lines[-n:]:
        print(l.rstrip())


# ============ CLI ============

def main():
    p = argparse.ArgumentParser(description="Visualisateur logs bot live")
    p.add_argument("--last", type=int, default=None,
                   help="Affiche les N dernieres lignes puis suit")
    p.add_argument("--asset", type=str, help="Filtre par actif (XAUUSD, NAS100, ...)")
    p.add_argument("--setups", action="store_true",
                   help="Seulement les SETUP / PENDING ORDER")
    p.add_argument("--trades", action="store_true",
                   help="Seulement les trades places/fermes")
    p.add_argument("--rejects", action="store_true", help="Seulement les rejets ML")
    p.add_argument("--cycles", action="store_true", help="Seulement les CYCLE scan")
    p.add_argument("--stats", action="store_true", help="Seulement les STATS")
    p.add_argument("--errors", action="store_true", help="Seulement WARNING+")
    p.add_argument("--latency", action="store_true",
                   help="Tableau latence du dernier cycle")
    p.add_argument("--recent-setups", type=int, metavar="N",
                   help="Affiche les N derniers setups (snapshot, pas de follow)")
    p.add_argument("--recent-rejets", type=int, metavar="N",
                   help="Affiche les N derniers rejets ML (snapshot)")
    args = p.parse_args()

    if args.latency:
        show_latency_last_cycle()
        return
    if args.recent_setups:
        show_recent_setups(args.recent_setups)
        return
    if args.recent_rejets:
        show_recent_rejets(args.recent_rejets)
        return

    # Compose le filtre
    if args.asset:
        match_fn = f_asset(args.asset)
        label = f"actif={args.asset}"
    elif args.setups:
        match_fn = f_setups
        label = "SETUPS"
    elif args.trades:
        match_fn = f_trades
        label = "TRADES"
    elif args.rejects:
        match_fn = f_rejects
        label = "REJETS"
    elif args.cycles:
        match_fn = f_cycles
        label = "CYCLES"
    elif args.stats:
        match_fn = f_stats
        label = "STATS"
    elif args.errors:
        match_fn = f_errors
        label = "ERREURS"
    else:
        match_fn = f_all
        label = "ALL"

    n_first = args.last if args.last is not None else 50
    print(f"=== Tail {label} (last {n_first} + follow) -> {LOG_PATH} ===\n")
    tail_follow(match_fn, n_first=n_first)


if __name__ == "__main__":
    main()
