"""Analyse les wins vs losses pour identifier les facteurs predictifs."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.portfolio import FictivePortfolio
from bot.strategy import scan_all
from bot.training import load_all_dfs, load_correlates_m1, slice_day
from config import INITIAL_BALANCE, INSTRUMENTS
from db.models import init_db


def collect_taken_trades():
    """Reconstit la liste des trades pris sur tous les actifs."""
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
            if not trades:
                continue
            def k(t):
                sa = t.setup_analysis or {}
                hf = sa.get("hard_filters", [])
                passes = sum(1 for f in hf if f.get("passed")) / (len(hf) or 1)
                return (passes, t.would_trade, t.score, min(t.risk_reward, 10))
            best = max(trades, key=k)
            if best.would_trade and best.status in ("win", "loss"):
                all_taken.append(best)
    return all_taken


def main():
    trades = collect_taken_trades()
    wins = [t for t in trades if t.status == "win"]
    losses = [t for t in trades if t.status == "loss"]
    print(f"\nTotal: {len(trades)} pris (W:{len(wins)} L:{len(losses)})\n")

    # Pour chaque facteur, fraction de WIN vs LOSS quand il est present
    factor_stats: dict[str, dict] = {}
    for t in trades:
        is_win = t.status == "win"
        sa = t.setup_analysis or {}
        for f in sa.get("factors", []):
            name = f.get("name")
            if not name:
                continue
            status = f.get("status")
            d = factor_stats.setdefault(name, {
                "label": f.get("label", name),
                "present_w": 0, "present_l": 0,
                "warning_w": 0, "warning_l": 0,
                "absent_w": 0, "absent_l": 0,
            })
            if status == "present":
                d["present_w" if is_win else "present_l"] += 1
            elif status == "warning":
                d["warning_w" if is_win else "warning_l"] += 1
            else:
                d["absent_w" if is_win else "absent_l"] += 1

    print(f"{'Facteur':<35} | PRES W/L  | WARN W/L  | ABS W/L | win% si pres")
    print("-" * 100)
    for name, d in sorted(factor_stats.items()):
        pres_total = d["present_w"] + d["present_l"]
        warn_total = d["warning_w"] + d["warning_l"]
        abs_total = d["absent_w"] + d["absent_l"]
        win_pct_pres = (d["present_w"] / pres_total * 100) if pres_total else None
        win_pct_str = f"{win_pct_pres:.0f}%" if win_pct_pres is not None else "-"
        print(f"{name:<35} | {d['present_w']:2d}/{d['present_l']:2d}     | "
              f"{d['warning_w']:2d}/{d['warning_l']:2d}     | "
              f"{d['absent_w']:2d}/{d['absent_l']:2d}    | {win_pct_str}")

    print("\n=== Filtres durs : casses vs passes ===")
    hf_stats: dict[str, dict] = {}
    for t in trades:
        is_win = t.status == "win"
        for hf in (t.setup_analysis or {}).get("hard_filters", []):
            name = hf.get("name")
            d = hf_stats.setdefault(name, {"label": hf.get("label", name), "pass_w": 0, "pass_l": 0, "fail_w": 0, "fail_l": 0})
            if hf.get("passed"):
                d["pass_w" if is_win else "pass_l"] += 1
            else:
                d["fail_w" if is_win else "fail_l"] += 1

    print(f"{'Filtre dur':<40} | PASS W/L | FAIL W/L")
    print("-" * 80)
    for name, d in sorted(hf_stats.items()):
        print(f"{name:<40} | {d['pass_w']:2d}/{d['pass_l']:2d}    | {d['fail_w']:2d}/{d['fail_l']:2d}")


if __name__ == "__main__":
    main()
