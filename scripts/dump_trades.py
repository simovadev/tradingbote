"""Pioche N trades aleatoires et dump tout le contexte pour analyse humaine.

Sortie : un fichier markdown par trade dans scripts/output/, plus un summary.
"""
from __future__ import annotations

import json
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from bot.training import find_best_trade_of_day, load_all_dfs, pick_random_day, slice_day
from db.models import init_db


OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)


def candles_around(df: pd.DataFrame, ts: pd.Timestamp, before: int, after: int) -> pd.DataFrame:
    """Retourne les bougies autour d'un timestamp."""
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    try:
        loc = df.index.get_indexer([ts], method="nearest")[0]
    except Exception:
        return df.iloc[0:0]
    start = max(0, loc - before)
    end = min(len(df), loc + after)
    return df.iloc[start:end]


def format_candles_text(df: pd.DataFrame, max_rows: int = 30) -> str:
    """Compact OHLC pour lecture humaine."""
    if df.empty:
        return "(no candles)"
    df = df.tail(max_rows)
    lines = ["time                  | open    | high    | low     | close   | body"]
    lines.append("-" * 70)
    for ts, row in df.iterrows():
        body = row["close"] - row["open"]
        direction = "↑" if body > 0 else "↓"
        lines.append(
            f"{ts.strftime('%Y-%m-%d %H:%M')}    | "
            f"{row['open']:.2f} | {row['high']:.2f} | {row['low']:.2f} | "
            f"{row['close']:.2f} | {direction}{abs(body):+.2f}"
        )
    return "\n".join(lines)


def dump_trade(idx: int, dfs: dict, day: pd.Timestamp, trade) -> str:
    """Produit le rapport markdown d'un trade et retourne le path."""
    out_lines = [f"# Trade #{idx} — {trade.direction.upper()} {trade.instrument}"]
    out_lines.append(f"\n**Jour piocé** : {day.strftime('%Y-%m-%d (%A)')}")
    out_lines.append(f"**Entree** : {trade.entry_time} @ **{trade.entry_price:.2f}**")
    out_lines.append(f"**SL** : {trade.stop_loss:.2f} | **TP** : {trade.take_profit:.2f}")
    out_lines.append(f"**RR** : {trade.risk_reward:.2f} | **Outcome** : {trade.status} ({trade.pnl:+.2f}$)")
    out_lines.append(f"**Score bot** : {trade.score}/100 ({trade.verdict})")
    out_lines.append(f"**Bot l'aurait pris** : {'OUI' if trade.would_trade else 'NON'}")

    sa = trade.setup_analysis or {}

    # Filtres durs
    out_lines.append("\n## Filtres durs")
    for hf in sa.get("hard_filters", []):
        icon = "✓" if hf.get("passed") else "✗"
        out_lines.append(f"- {icon} **{hf.get('label')}** — _{hf.get('detail', '')}_")

    # Facteurs
    out_lines.append("\n## Facteurs vus par le bot")
    by_cat: dict[str, list] = {}
    for f in sa.get("factors", []):
        by_cat.setdefault(f.get("category", "misc"), []).append(f)
    for cat, fs in by_cat.items():
        out_lines.append(f"\n### {cat}")
        for f in fs:
            sign = "+" if f.get("applied_score", 0) >= 0 else ""
            icon = {"present": "✓", "warning": "⚠", "absent": "·", "neutral": "·"}.get(f.get("status"), "?")
            out_lines.append(f"- {icon} **{f.get('label')}** ({sign}{f.get('applied_score', 0)}) — _{f.get('detail', '')}_")

    # Zones tracees par le bot
    out_lines.append("\n## Zones tracees par le bot")
    ov = trade.chart_overlays or {}
    if ov.get("liquidity"):
        out_lines.append(f"- **Liquidite** : {ov['liquidity'].get('price'):.2f} ({ov['liquidity'].get('side')})")
    if ov.get("sweep"):
        out_lines.append(f"- **Sweep** : {ov['sweep'].get('time')} (wick @ {ov['sweep'].get('wick'):.2f})")
    if ov.get("structure_break"):
        out_lines.append(f"- **BOS** : {ov['structure_break'].get('time')} @ {ov['structure_break'].get('price'):.2f}")
    if ov.get("ob_zone"):
        ob = ov["ob_zone"]
        out_lines.append(f"- **OB** : [{ob.get('low'):.2f} → {ob.get('high'):.2f}], mid={ob.get('mid'):.2f}, type={ob.get('type')}")

    # OHLC contexte M1 autour de l'entree (les 30 bougies avant et 15 apres)
    df_m1 = dfs.get("M1")
    if df_m1 is not None:
        out_lines.append("\n## Bougies M1 (30 avant + 15 apres entree)")
        out_lines.append("```")
        snippet = candles_around(df_m1, trade.entry_time, before=30, after=15)
        out_lines.append(format_candles_text(snippet, max_rows=50))
        out_lines.append("```")

    # Contexte H1 (15 bougies avant l'entree)
    df_h1 = dfs.get("H1")
    if df_h1 is not None:
        out_lines.append("\n## Bougies H1 (15 avant entree)")
        out_lines.append("```")
        h1_snippet = candles_around(df_h1, trade.entry_time, before=15, after=3)
        out_lines.append(format_candles_text(h1_snippet, max_rows=20))
        out_lines.append("```")

    # H4 (5 bougies avant l'entree)
    df_h4 = dfs.get("H4")
    if df_h4 is not None:
        out_lines.append("\n## Bougies H4 (5 avant entree)")
        out_lines.append("```")
        h4_snippet = candles_around(df_h4, trade.entry_time, before=5, after=2)
        out_lines.append(format_candles_text(h4_snippet, max_rows=10))
        out_lines.append("```")

    path = OUT_DIR / f"trade_{idx:02d}_{trade.entry_time.strftime('%Y%m%d_%H%M')}.md"
    path.write_text("\n".join(out_lines), encoding="utf-8")
    return str(path)


def main(n: int = 10) -> None:
    init_db()
    # Purge les anciens dumps pour repartir propre a chaque batch
    for old in OUT_DIR.glob("trade_*.md"):
        old.unlink()
    for old in OUT_DIR.glob("SUMMARY.md"):
        old.unlink()

    dfs = load_all_dfs()
    if "M1" not in dfs:
        print("ERREUR: M1 manquant")
        return

    found = []
    attempts = 0
    max_attempts = n * 5

    print(f"\n=== Pioche de {n} trades random ===\n")

    seen_days: set = set()
    while len(found) < n and attempts < max_attempts:
        attempts += 1
        day = pick_random_day(dfs, exclude_seen=False)
        if day is None:
            break

        # Pas de doublons dans le batch
        if day in seen_days:
            continue
        seen_days.add(day)

        dfs_day = slice_day(dfs, day)
        trade = find_best_trade_of_day(dfs_day)

        if trade is None:
            print(f"  [{attempts:02d}] {day.date()} : aucun setup")
            continue

        found.append((day, trade))
        path = dump_trade(len(found), dfs, day, trade)
        print(f"  [{attempts:02d}] {day.date()} : {trade.direction.upper()} score={trade.score:3d} "
              f"verdict={trade.verdict:<9} outcome={trade.status:<7} -> {Path(path).name}")

    # Summary
    print(f"\n=== {len(found)} trades dumped ===")
    summary = OUT_DIR / "SUMMARY.md"
    lines = ["# Summary des trades pioces\n"]
    lines.append(f"_{len(found)} trades / {attempts} jours testes_\n\n")
    lines.append("| # | Date | Dir | Score | Verdict | RR | Outcome | PnL | KZ entry | KZ OB |")
    lines.append("|---|------|-----|-------|---------|-----|---------|-----|----------|-------|")
    for i, (day, t) in enumerate(found, 1):
        sa = t.setup_analysis or {}
        hard = {hf["name"]: hf for hf in sa.get("hard_filters", [])}
        kz_e = hard.get("entry_in_killzone", {}).get("detail", "?").split(" - ")[-1]
        kz_o = hard.get("ob_in_killzone", {}).get("detail", "?").split(" - ")[-1]
        lines.append(
            f"| {i} | {day.date()} | {t.direction} | {t.score} | {t.verdict} | "
            f"{t.risk_reward:.2f} | {t.status} | {t.pnl:+.2f} | {kz_e} | {kz_o} |"
        )
    summary.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSummary : {summary}")


if __name__ == "__main__":
    main(10)
