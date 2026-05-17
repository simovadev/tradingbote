"""Scan complet : pour chaque jour disponible, regarde si le bot detecte
un trade VALIDE (tous filtres durs OK + would_trade=True).

Output: stats consolidees + liste des trades pris.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from bot.training import find_best_trade_of_day, load_all_dfs, slice_day
from db.models import init_db


OUT = Path(__file__).parent / "output" / "full_scan.md"


def main() -> None:
    init_db()
    dfs = load_all_dfs()
    df_m1 = dfs.get("M1")
    if df_m1 is None:
        print("ERREUR: M1 manquant")
        return

    days = sorted(set(df_m1.index.normalize().unique().tolist()))
    print(f"Scan de {len(days)} jours...\n")

    all_trades = []
    taken_trades = []
    rejected_trades = []
    no_setup_days = 0

    for d in days:
        if d.tz is None:
            d = d.tz_localize("UTC")
        sliced = slice_day(dfs, d)
        trade = find_best_trade_of_day(sliced)
        if trade is None:
            no_setup_days += 1
            continue
        all_trades.append((d, trade))
        if trade.would_trade:
            taken_trades.append((d, trade))
        else:
            rejected_trades.append((d, trade))

    # Stats trades pris
    wins = sum(1 for d, t in taken_trades if t.status == "win")
    losses = sum(1 for d, t in taken_trades if t.status == "loss")
    pendings = sum(1 for d, t in taken_trades if t.status == "pending")
    total_pnl = sum(t.pnl for d, t in taken_trades)
    winrate = (wins / (wins + losses) * 100) if (wins + losses) else 0

    avg_rr_wins = sum(t.risk_reward for d, t in taken_trades if t.status == "win") / wins if wins else 0
    expectancy = (winrate / 100) * avg_rr_wins - (1 - winrate / 100)

    print(f"\n=== Scan termine ===")
    print(f"Jours scannees      : {len(days)}")
    print(f"Sans setup          : {no_setup_days}")
    print(f"Trades detectes     : {len(all_trades)}")
    print(f"  - PRIS (filtres OK + score>=65) : {len(taken_trades)}")
    print(f"     wins  : {wins}")
    print(f"     loss  : {losses}")
    print(f"     pending : {pendings}")
    print(f"     winrate : {winrate:.1f}%")
    print(f"     PnL total : {total_pnl:+.2f}$")
    print(f"     RR moyen wins : {avg_rr_wins:.2f}")
    print(f"     Expectancy   : {expectancy:+.2f}R")
    print(f"  - REJETES : {len(rejected_trades)}")

    # Markdown
    lines = ["# Full scan results\n\n"]
    lines.append(f"- Jours : {len(days)} (dont {no_setup_days} sans setup)\n")
    lines.append(f"- Trades pris : **{len(taken_trades)}** "
                 f"(W:{wins} / L:{losses} = winrate **{winrate:.1f}%**)\n")
    lines.append(f"- PnL total : **{total_pnl:+.2f}$** sur 10 000$\n")
    lines.append(f"- Expectancy : **{expectancy:+.2f}R**\n\n")

    lines.append("## Trades PRIS\n\n")
    lines.append("| Jour | Dir | Score | Verdict | RR | Outcome | PnL | KZ |\n")
    lines.append("|------|-----|-------|---------|-----|---------|-----|-----|\n")
    for d, t in taken_trades:
        sa = t.setup_analysis or {}
        kz = next((hf.get("detail", "").split(" - ")[-1] for hf in sa.get("hard_filters", [])
                   if hf.get("name") == "entry_in_killzone"), "?")
        lines.append(
            f"| {d.date()} | {t.direction} | {t.score} | {t.verdict} | "
            f"{t.risk_reward:.2f} | {t.status} | {t.pnl:+.2f} | {kz} |\n"
        )

    lines.append("\n## Trades REJETES (raison)\n\n")
    lines.append("| Jour | Dir | Score | Raison reject |\n")
    lines.append("|------|-----|-------|---------------|\n")
    for d, t in rejected_trades:
        sa = t.setup_analysis or {}
        failed = [hf.get("label") for hf in sa.get("hard_filters", []) if not hf.get("passed")]
        reason = ", ".join(failed) if failed else "score < 65"
        lines.append(f"| {d.date()} | {t.direction} | {t.score} | {reason} |\n")

    OUT.write_text("".join(lines), encoding="utf-8")
    print(f"\nRapport : {OUT}")


if __name__ == "__main__":
    main()
