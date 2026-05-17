"""Calcule le nombre de trades pris par jour, par actif."""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.portfolio import FictivePortfolio
from bot.strategy import scan_all
from bot.training import load_all_dfs, load_correlates_m1, slice_day
from config import INITIAL_BALANCE, INSTRUMENTS
from db.models import init_db


def main():
    init_db()
    all_by_day: dict = defaultdict(lambda: defaultdict(int))   # day -> instrument -> count
    detected_by_day: dict = defaultdict(lambda: defaultdict(int))

    for inst in INSTRUMENTS:
        dfs = load_all_dfs(inst)
        df_m1 = dfs.get("M1")
        if df_m1 is None or df_m1.empty:
            continue
        correlates = load_correlates_m1(inst)
        days = sorted(set(df_m1.index.normalize().unique().tolist()))
        days = [d for d in days if (d.tz_localize("UTC") if d.tz is None else d).weekday() < 5]

        for d in days:
            if d.tz is None:
                d = d.tz_localize("UTC")
            sliced = slice_day(dfs, d)
            portfolio = FictivePortfolio(balance=INITIAL_BALANCE)
            trades = scan_all(sliced, portfolio, instrument=inst,
                              correlates_m1=correlates, score_threshold=0)
            taken = [t for t in trades if t.would_trade]
            detected_by_day[d.date()][inst] = len(trades)
            all_by_day[d.date()][inst] = len(taken)

    # Print par jour
    print(f"\n{'Date':<12} | XAU(D/P) | NAS(D/P) | GER(D/P) | OIL(D/P) | TOTAL pris")
    print("-" * 80)
    total_per_inst = defaultdict(int)
    total_detected = defaultdict(int)
    days_with_trades = 0

    for d in sorted(all_by_day.keys()):
        row = all_by_day[d]
        det = detected_by_day[d]
        total = sum(row.values())
        if total > 0:
            days_with_trades += 1
        print(f"{d}   | "
              f"{det.get('XAUUSD',0):2d}/{row.get('XAUUSD',0):2d}    | "
              f"{det.get('NAS100',0):2d}/{row.get('NAS100',0):2d}    | "
              f"{det.get('GER40',0):2d}/{row.get('GER40',0):2d}    | "
              f"{det.get('USOIL',0):2d}/{row.get('USOIL',0):2d}    | "
              f"{total}")
        for inst, c in row.items():
            total_per_inst[inst] += c
        for inst, c in det.items():
            total_detected[inst] += c

    total_days = len(all_by_day)
    print("-" * 80)
    print(f"\nResume sur {total_days} jours (jours ouvres uniquement):\n")
    print(f"{'Actif':<10} | Detectes | Pris  | Moy detect/j | Moy pris/j")
    print("-" * 60)
    for inst in INSTRUMENTS:
        det = total_detected[inst]
        pris = total_per_inst[inst]
        print(f"{inst:<10} | {det:8d} | {pris:5d} | {det/total_days:11.2f}  | {pris/total_days:.2f}")
    total_pris = sum(total_per_inst.values())
    total_det = sum(total_detected.values())
    print("-" * 60)
    print(f"{'TOTAL 4 actifs':<10} | {total_det:8d} | {total_pris:5d} | {total_det/total_days:11.2f}  | {total_pris/total_days:.2f}")
    print(f"\nJours avec au moins 1 trade pris : {days_with_trades}/{total_days} ({days_with_trades/total_days*100:.0f}%)")


if __name__ == "__main__":
    main()
