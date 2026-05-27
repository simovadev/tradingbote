"""V19 SHADOW DEMO : loggue toutes les predictions V19 sans envoyer d'ordres reels.

Usage :
    SHADOW_MODE=1 python -m bot_v2.live_runner_v2

Quand SHADOW_MODE=1 :
- Tous les setups valides sont logges dans shadow_predictions.jsonl
- Aucun ordre MT5 envoye
- On peut comparer apres 24-48h les probas V19 vs les vrais resultats

Output : shadow_predictions.jsonl (1 ligne JSON par prediction)
{
    "ts": "2026-05-27T14:30:00Z",
    "instrument": "EURUSD",
    "direction": "bullish",
    "entry": 1.0850,
    "sl": 1.0820,
    "tp": 1.0910,
    "rr": 2.0,
    "proba_v19": 0.73,
    "proba_v18": 0.68,
    "threshold": 0.65,
    "would_trade": true,
    "features_ict": {...},
    "outcome": "PENDING"  # rempli a posteriori via shadow_reconcile.py
}
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = "c:/Users/Shadow/TradingBot" if sys.platform == "win32" else "/workspace/TradingBot"
sys.path.insert(0, ROOT)


SHADOW_LOG = Path(f"{ROOT}/shadow_predictions.jsonl")


def is_shadow_mode() -> bool:
    return os.environ.get("SHADOW_MODE", "0") == "1"


def log_prediction(
    ts,
    instrument: str,
    direction: str,
    entry: float,
    sl: float,
    tp: float,
    rr: float,
    proba_v19: float | None,
    proba_v18: float | None,
    threshold: float,
    would_trade: bool,
    features_ict: dict | None = None,
    ob_validation_ts=None,
) -> None:
    """Log 1 prediction dans shadow_predictions.jsonl."""
    if hasattr(ts, "isoformat"):
        ts_str = ts.isoformat()
    else:
        ts_str = str(ts)
    ob_ts_str = ob_validation_ts.isoformat() if hasattr(ob_validation_ts, "isoformat") else None

    record = {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "ts": ts_str,
        "ob_validation_ts": ob_ts_str,
        "instrument": instrument,
        "direction": direction,
        "entry": float(entry),
        "sl": float(sl),
        "tp": float(tp),
        "rr": float(rr),
        "proba_v19": float(proba_v19) if proba_v19 is not None else None,
        "proba_v18": float(proba_v18) if proba_v18 is not None else None,
        "threshold": float(threshold),
        "would_trade": bool(would_trade),
        "features_ict": features_ict,
        "outcome": "PENDING",
    }
    SHADOW_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(SHADOW_LOG, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def load_predictions() -> list[dict]:
    """Charge toutes les predictions logguees."""
    if not SHADOW_LOG.exists():
        return []
    records = []
    with open(SHADOW_LOG) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    continue
    return records


def summary():
    """Resume des predictions shadow."""
    records = load_predictions()
    if not records:
        print("Aucune prediction shadow loggee.")
        return

    print(f"=== SHADOW V19 SUMMARY ({len(records)} predictions) ===\n")

    # Par instrument
    by_inst: dict[str, list] = {}
    for r in records:
        by_inst.setdefault(r["instrument"], []).append(r)

    print(f"{'Instrument':<12} {'Total':<7} {'V19>0.65':<10} {'V18>0.65':<10} {'Would trade':<12}")
    for inst, recs in sorted(by_inst.items()):
        n = len(recs)
        n_v19 = sum(1 for r in recs if r.get("proba_v19") and r["proba_v19"] >= 0.65)
        n_v18 = sum(1 for r in recs if r.get("proba_v18") and r["proba_v18"] >= 0.65)
        n_trade = sum(1 for r in recs if r.get("would_trade"))
        print(f"  {inst:<12} {n:<7} {n_v19:<10} {n_v18:<10} {n_trade:<12}")

    # Stats overall
    closed = [r for r in records if r.get("outcome") in ("WIN", "LOSS")]
    if closed:
        wins = sum(1 for r in closed if r["outcome"] == "WIN")
        wr = wins / len(closed) * 100
        print(f"\n=== OUTCOMES (sur {len(closed)} trades closes) ===")
        print(f"  WIN  : {wins}")
        print(f"  LOSS : {len(closed) - wins}")
        print(f"  WR   : {wr:.1f}%")


if __name__ == "__main__":
    summary()
