"""Audit logger detaille pour analyse forensic V21.

Enregistre TOUT ce qui se passe pour chaque OB detecte + tente de trade :
- Timing : OB validation_ts -> now -> exec_ts
- Prix : OB top/bottom, market_price, entry effectif, slippage
- ICT : sweep_strength, has_fvg, daily_bias, retest_count, etc.
- ML : proba V21, proba V20 (shadow)
- Decision : accepted / rejected / mt5_failed (avec raison)
- Outcome (a remplir par cron apres SL/TP) : WIN / LOSS / pnl_eur

Output : data/v21_audit_live.parquet (append-only)
"""
from __future__ import annotations

import threading
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


class TradeAuditLogger:
    """Logger thread-safe pour audit detaille V21 live."""

    def __init__(self, output_path: Path | None = None, flush_every: int = 50):
        self.output_path = output_path or (ROOT / "data" / "v21_audit_live.parquet")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.flush_every = flush_every
        self._buffer: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def log(self, **fields):
        """Enregistre 1 event audit."""
        fields.setdefault("ts_log", datetime.now(timezone.utc).isoformat())
        with self._lock:
            self._buffer.append(fields)
            if len(self._buffer) >= self.flush_every:
                self._flush_locked()

    def flush(self):
        with self._lock:
            self._flush_locked()

    def _flush_locked(self):
        if not self._buffer:
            return
        df = pd.DataFrame(self._buffer)
        # Conversion isoformat tous les Timestamp/datetime
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].apply(
                    lambda x: x.isoformat() if hasattr(x, "isoformat") else x
                )
        if self.output_path.exists():
            try:
                old = pd.read_parquet(self.output_path)
                df = pd.concat([old, df], ignore_index=True)
            except Exception:
                pass
        df.to_parquet(self.output_path, index=False)
        self._buffer = []
