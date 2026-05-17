"""Scan complet sur tous les actifs + tous les jours dispo.

Pour chaque (instrument, jour) :
  - Charge les 6 TFs
  - Charge les correles M1 (pour SMT)
  - Trouve les setups detectes
  - Compte ceux qui passent (would_trade) et leur outcome

Sortie : rapport markdown complet + stats par actif.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from bot.portfolio import FictivePortfolio
from bot.strategy import scan_all
from bot.training import load_all_dfs, load_correlates_m1, slice_day
from config import INITIAL_BALANCE, INSTRUMENTS
from db.models import init_db


OUT = Path(__file__).parent / "output" / "full_scan_multi.md"


def scan_one_instrument(instrument: str) -> dict:
    """Scan complet d'un instrument."""
    dfs = load_all_dfs(instrument)
    df_m1 = dfs.get("M1")
    if df_m1 is None or df_m1.empty:
        return {"instrument": instrument, "error": "no M1 data"}

    correlates = load_correlates_m1(instrument)

    days = sorted(set(df_m1.index.normalize().unique().tolist()))
    days = [d for d in days if (d.tz_localize("UTC") if d.tz is None else d).weekday() < 5]

    portfolio = FictivePortfolio(balance=INITIAL_BALANCE)
    all_trades = []
    taken_trades = []
    rejected_trades = []
    no_setup_days = 0

    for d in days:
        if d.tz is None:
            d = d.tz_localize("UTC")
        sliced = slice_day(dfs, d)
        # Pas trop chargé : on recree portfolio par jour, on consolide a la fin
        day_portfolio = FictivePortfolio(balance=portfolio.balance)
        trades_day = scan_all(sliced, day_portfolio, instrument=instrument,
                              correlates_m1=correlates, score_threshold=0)
        if not trades_day:
            no_setup_days += 1
            continue

        # On prend le meilleur du jour
        def k(t):
            sa = t.setup_analysis or {}
            hf = sa.get("hard_filters", [])
            passes = sum(1 for f in hf if f.get("passed")) / (len(hf) or 1)
            return (passes, t.would_trade, t.score, min(t.risk_reward, 10))

        best = max(trades_day, key=k)
        all_trades.append(best)
        if best.would_trade:
            taken_trades.append(best)
            if best.status == "win":
                portfolio.apply_result(best.pnl, True)
            elif best.status == "loss":
                portfolio.apply_result(best.pnl, False)
        else:
            rejected_trades.append(best)

    wins = sum(1 for t in taken_trades if t.status == "win")
    losses = sum(1 for t in taken_trades if t.status == "loss")
    winrate = (wins / (wins + losses) * 100) if (wins + losses) else 0
    total_pnl = portfolio.balance - portfolio.initial_balance

    return {
        "instrument": instrument,
        "days_scanned": len(days),
        "no_setup_days": no_setup_days,
        "total_trades_detected": len(all_trades),
        "taken_trades": len(taken_trades),
        "wins": wins,
        "losses": losses,
        "pendings": len(taken_trades) - wins - losses,
        "winrate": winrate,
        "total_pnl": total_pnl,
        "pnl_pct": (total_pnl / INITIAL_BALANCE * 100),
        "final_balance": portfolio.balance,
        "taken_list": [(t.entry_time, t.direction, t.score, t.risk_reward, t.status, t.pnl) for t in taken_trades],
        "rejected_list": [(t.entry_time, t.direction, t.score, t.setup_analysis) for t in rejected_trades],
    }


def main() -> None:
    init_db()
    instruments = list(INSTRUMENTS.keys())

    print(f"\n=== Scan multi-actif : {instruments} ===\n")

    results = []
    for inst in instruments:
        print(f"\n--- {inst} ---")
        r = scan_one_instrument(inst)
        if "error" in r:
            print(f"  ERREUR: {r['error']}")
            continue
        results.append(r)
        print(f"  Jours: {r['days_scanned']} (vide: {r['no_setup_days']})")
        print(f"  Trades detectes: {r['total_trades_detected']}")
        print(f"  PRIS: {r['taken_trades']} (W:{r['wins']} L:{r['losses']}) "
              f"winrate {r['winrate']:.0f}% | PnL {r['total_pnl']:+.0f}$ ({r['pnl_pct']:+.1f}%)")

    # Rapport markdown
    lines = ["# Scan multi-actif - Resultats\n\n"]
    lines.append("| Actif | Jours | Setups | Pris | W | L | Winrate | PnL | % |\n")
    lines.append("|-------|-------|--------|------|---|---|---------|-----|---|\n")
    total_taken = total_w = total_l = 0
    total_pnl = 0
    for r in results:
        lines.append(
            f"| {r['instrument']} | {r['days_scanned']} | {r['total_trades_detected']} | "
            f"{r['taken_trades']} | {r['wins']} | {r['losses']} | "
            f"{r['winrate']:.0f}% | {r['total_pnl']:+.0f}$ | {r['pnl_pct']:+.1f}% |\n"
        )
        total_taken += r['taken_trades']
        total_w += r['wins']
        total_l += r['losses']
        total_pnl += r['total_pnl']

    overall_wr = (total_w / (total_w + total_l) * 100) if (total_w + total_l) else 0
    lines.append(f"\n**Total : {total_taken} trades pris, W:{total_w}/L:{total_l} "
                 f"= winrate {overall_wr:.1f}%, PnL {total_pnl:+.0f}$**\n\n")

    for r in results:
        lines.append(f"\n## {r['instrument']} - Trades pris\n\n")
        lines.append("| Date | Dir | Score | RR | Outcome | PnL |\n")
        lines.append("|------|-----|-------|-----|---------|-----|\n")
        for ts, dir, score, rr, status, pnl in r['taken_list']:
            lines.append(f"| {ts.strftime('%Y-%m-%d %H:%M')} | {dir} | {score} | {rr:.2f} | {status} | {pnl:+.0f}$ |\n")

        # Top 3 raisons de reject
        reasons_count: dict[str, int] = {}
        for ts, dir, score, sa in r['rejected_list']:
            failed = [hf.get("label") for hf in (sa or {}).get("hard_filters", []) if not hf.get("passed")]
            for f in failed:
                reasons_count[f] = reasons_count.get(f, 0) + 1
        top_reasons = sorted(reasons_count.items(), key=lambda x: -x[1])[:5]
        if top_reasons:
            lines.append(f"\n**Top raisons reject ({r['instrument']}) :**\n")
            for label, n in top_reasons:
                lines.append(f"- {label} : {n}x\n")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(lines), encoding="utf-8")
    print(f"\nRapport : {OUT}")
    print(f"\n=== TOTAL : {total_taken} trades pris, W:{total_w}/L:{total_l} = {overall_wr:.1f}% winrate, PnL {total_pnl:+.0f}$ ===")


if __name__ == "__main__":
    main()
