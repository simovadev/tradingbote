"""Logger spreads pour les 77 actifs.

A chaque cycle V21, on enregistre tick.ask-tick.bid pour chaque actif.
Apres 7-14 jours, on a un dataset avec spread par actif x heure UTC x weekday.

Usage dans live_runner_v21.py :
    from bot_v2.spread_logger import SpreadLogger
    spread_logger = SpreadLogger()  # au boot
    # Dans chaque cycle :
    spread_logger.log_tick(now_utc, asset, tick.ask, tick.bid, info.point)
    # Flush toutes les 10 min
    if cycle_n % 10 == 0:
        spread_logger.flush()
"""
from __future__ import annotations

import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


class SpreadLogger:
    """Logger thread-safe pour spreads MT5."""

    def __init__(self, output_path: Path | None = None, flush_every: int = 5000):
        self.output_path = output_path or (ROOT / "data" / "spreads_history.parquet")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.flush_every = flush_every
        self._buffer: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def log_tick(self, ts: datetime, asset: str, ask: float, bid: float, point: float):
        """Enregistre 1 tick. point = info.point (taille minimale tick)."""
        if ask <= 0 or bid <= 0 or ask < bid:
            return
        spread_abs = ask - bid
        spread_points = spread_abs / point if point > 0 else 0
        mid = (ask + bid) / 2
        spread_pct = spread_abs / mid * 100 if mid > 0 else 0
        row = {
            "ts": ts.isoformat(),
            "asset": asset,
            "hour_utc": ts.hour,
            "weekday": ts.weekday(),  # 0=Mon, 6=Sun
            "spread_points": float(spread_points),
            "spread_pct": float(spread_pct),
            "mid_price": float(mid),
        }
        with self._lock:
            self._buffer.append(row)
            if len(self._buffer) >= self.flush_every:
                self._flush_locked()

    def flush(self):
        """Force flush (a appeler regulierement)."""
        with self._lock:
            self._flush_locked()

    def _flush_locked(self):
        if not self._buffer:
            return
        df = pd.DataFrame(self._buffer)
        # Append au parquet (cree si n'existe pas)
        if self.output_path.exists():
            try:
                old = pd.read_parquet(self.output_path)
                df = pd.concat([old, df], ignore_index=True)
            except Exception:
                pass
        df.to_parquet(self.output_path, index=False)
        self._buffer = []

    def get_stats(self, asset: str, hour_utc: int | None = None) -> dict:
        """Stats spread pour un actif (optionnel par heure UTC)."""
        if not self.output_path.exists():
            return {}
        df = pd.read_parquet(self.output_path)
        sub = df[df["asset"] == asset]
        if hour_utc is not None:
            sub = sub[sub["hour_utc"] == hour_utc]
        if len(sub) == 0:
            return {}
        return {
            "n": len(sub),
            "median_pct": float(sub["spread_pct"].median()),
            "p95_pct": float(sub["spread_pct"].quantile(0.95)),
            "max_pct": float(sub["spread_pct"].max()),
            "median_points": float(sub["spread_points"].median()),
        }
