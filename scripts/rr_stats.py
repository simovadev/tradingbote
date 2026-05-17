"""Stats RR par trade pris."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.portfolio import FictivePortfolio
from bot.strategy import scan_all
from bot.training import load_all_dfs, load_correlates_m1, slice_day
from config import INITIAL_BALANCE, INSTRUMENTS
from db.models import init_db


def main():
    init_db()
    all_taken = []

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
            # On garde le meilleur du jour si pris
            def k(t):
                sa = t.setup_analysis or {}
                hf = sa.get("hard_filters", [])
                passes = sum(1 for f in hf if f.get("passed")) / (len(hf) or 1)
                return (passes, t.would_trade, t.score, min(t.risk_reward, 10))
            if trades:
                best = max(trades, key=k)
                if best.would_trade and best.status in ("win", "loss"):
                    all_taken.append((inst, best))

    # Par actif
    print(f"\n{'='*80}")
    print(f"{'Actif':<10} | {'N':>3} | {'RR demande moy':<15} | {'RR realise moy':<15} | {'RR realise/trade'}")
    print(f"{'-'*80}")
    by_inst: dict = {}
    for inst, t in all_taken:
        by_inst.setdefault(inst, []).append(t)

    for inst, trades in by_inst.items():
        wins = [t for t in trades if t.status == "win"]
        losses = [t for t in trades if t.status == "loss"]
        n = len(trades)
        avg_rr_target = sum(t.risk_reward for t in trades) / n if n else 0
        # RR realise : win = +RR, loss = -1
        realized = [t.risk_reward if t.status == "win" else -1.0 for t in trades]
        avg_realized = sum(realized) / n if n else 0
        print(f"{inst:<10} | {n:3d} | {avg_rr_target:13.2f}  | {avg_realized:+13.2f}  | "
              f"wins:{[round(t.risk_reward,1) for t in wins]}")

    # Global
    print(f"{'-'*80}")
    n_total = len(all_taken)
    if n_total:
        avg_target = sum(t.risk_reward for _, t in all_taken) / n_total
        wins_total = [t for _, t in all_taken if t.status == "win"]
        losses_total = [t for _, t in all_taken if t.status == "loss"]
        avg_rr_wins = sum(t.risk_reward for t in wins_total) / len(wins_total) if wins_total else 0
        realized_all = [t.risk_reward if t.status == "win" else -1.0 for _, t in all_taken]
        avg_realized = sum(realized_all) / n_total
        winrate = len(wins_total) / n_total * 100
        expectancy = (winrate/100) * avg_rr_wins - (1 - winrate/100)
        print(f"{'TOTAL':<10} | {n_total:3d} | {avg_target:13.2f}  | {avg_realized:+13.2f}  | "
              f"winrate {winrate:.0f}%, RR moy wins {avg_rr_wins:.2f}")
        print(f"\nEXPECTANCY = {expectancy:+.2f}R par trade")
        print(f"({winrate:.0f}% × {avg_rr_wins:.2f} - {100-winrate:.0f}%)")

    # Distribution
    print(f"\nDistribution RR demande :")
    buckets = {"2-2.5": 0, "2.5-3": 0, "3-4": 0, "4-5": 0, "5-7": 0, "7-10": 0, ">10 capped": 0}
    for _, t in all_taken:
        rr = t.risk_reward
        if rr < 2.5: buckets["2-2.5"] += 1
        elif rr < 3: buckets["2.5-3"] += 1
        elif rr < 4: buckets["3-4"] += 1
        elif rr < 5: buckets["4-5"] += 1
        elif rr < 7: buckets["5-7"] += 1
        elif rr < 10: buckets["7-10"] += 1
        else: buckets[">10 capped"] += 1
    for b, c in buckets.items():
        print(f"  {b:<15} : {c}")


if __name__ == "__main__":
    main()
