"""Persistance SQLite pour le bot live.

Sauvegarde :
- Cooldowns par actif (last trade timestamp)
- Journal de tous les trades pris (audit + analyse post-mortem)
- Setups detectes mais rejetes (debug)

Decision user 2026-05-17 : tout sur SQLite local (pas de base distante).
Le bot reprend son etat exact au redemarrage.
"""
from __future__ import annotations

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


DB_PATH = Path("c:/Users/Shadow/TradingBot/db/live_state.sqlite")


class LiveState:
    """Wrapper SQLite simple pour le bot live."""

    def __init__(self, db_path: Path = DB_PATH):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        c = self.conn.cursor()
        c.executescript("""
        CREATE TABLE IF NOT EXISTS cooldowns (
            instrument TEXT PRIMARY KEY,
            last_trade_ts TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS trades_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket INTEGER UNIQUE,
            instrument TEXT NOT NULL,
            direction TEXT NOT NULL,
            opened_ts TEXT NOT NULL,
            closed_ts TEXT,
            entry REAL NOT NULL,
            sl REAL NOT NULL,
            tp REAL NOT NULL,
            rr REAL,
            volume REAL,
            score INTEGER,
            ml_proba REAL,
            killzone TEXT,
            outcome TEXT,            -- WIN | LOSS | PENDING | ERROR
            pnl_real REAL,           -- PnL reel reporte par MT5 (EUR)
            comment TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_trades_opened ON trades_log(opened_ts);
        CREATE INDEX IF NOT EXISTS idx_trades_instr  ON trades_log(instrument);

        CREATE TABLE IF NOT EXISTS rejected_setups (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            instrument TEXT NOT NULL,
            ts TEXT NOT NULL,
            direction TEXT NOT NULL,
            rejection_reason TEXT NOT NULL,
            ml_proba REAL,
            score INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS bot_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT DEFAULT CURRENT_TIMESTAMP,
            event_type TEXT NOT NULL,   -- START, STOP, ERROR, INFO
            message TEXT
        );
        """)
        self.conn.commit()

    # ========== COOLDOWNS ==========

    def get_last_trade_ts(self, instrument: str) -> pd.Timestamp | None:
        c = self.conn.cursor()
        c.execute("SELECT last_trade_ts FROM cooldowns WHERE instrument = ?", (instrument,))
        row = c.fetchone()
        if not row:
            return None
        return pd.Timestamp(row["last_trade_ts"])

    def set_last_trade_ts(self, instrument: str, ts: pd.Timestamp):
        c = self.conn.cursor()
        c.execute("""
            INSERT INTO cooldowns (instrument, last_trade_ts) VALUES (?, ?)
            ON CONFLICT(instrument) DO UPDATE SET last_trade_ts = excluded.last_trade_ts
        """, (instrument, ts.isoformat()))
        self.conn.commit()

    def is_in_cooldown(self, instrument: str, now: pd.Timestamp, cooldown_sec: int = 900) -> bool:
        last = self.get_last_trade_ts(instrument)
        if last is None:
            return False
        return (now - last).total_seconds() < cooldown_sec

    # ========== TRADES LOG ==========

    def log_trade_opened(self, *, ticket: int, instrument: str, direction: str,
                         opened_ts: pd.Timestamp, entry: float, sl: float, tp: float,
                         rr: float, volume: float, score: int, ml_proba: float,
                         killzone: str | None, comment: str = ""):
        c = self.conn.cursor()
        c.execute("""
            INSERT OR IGNORE INTO trades_log
            (ticket, instrument, direction, opened_ts, entry, sl, tp, rr, volume,
             score, ml_proba, killzone, outcome, comment)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', ?)
        """, (ticket, instrument, direction, opened_ts.isoformat(),
              entry, sl, tp, rr, volume, score, ml_proba, killzone, comment))
        self.conn.commit()

    def update_trade_closed(self, *, ticket: int, closed_ts: pd.Timestamp,
                            outcome: str, pnl_real: float):
        c = self.conn.cursor()
        c.execute("""
            UPDATE trades_log
            SET closed_ts = ?, outcome = ?, pnl_real = ?
            WHERE ticket = ?
        """, (closed_ts.isoformat(), outcome, pnl_real, ticket))
        self.conn.commit()

    def get_pending_tickets(self) -> list[int]:
        """Tickets ouverts (outcome PENDING) - pour reconcilier au demarrage."""
        c = self.conn.cursor()
        c.execute("SELECT ticket FROM trades_log WHERE outcome = 'PENDING'")
        return [r["ticket"] for r in c.fetchall()]

    def get_stats(self, since: pd.Timestamp | None = None) -> dict[str, Any]:
        """Stats globales (WR, PnL, etc.)."""
        c = self.conn.cursor()
        where = ""
        params: tuple = ()
        if since:
            where = "WHERE opened_ts >= ?"
            params = (since.isoformat(),)
        c.execute(f"""
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN outcome = 'WIN'  THEN 1 ELSE 0 END) AS wins,
                SUM(CASE WHEN outcome = 'LOSS' THEN 1 ELSE 0 END) AS losses,
                SUM(CASE WHEN outcome = 'PENDING' THEN 1 ELSE 0 END) AS pending,
                COALESCE(SUM(pnl_real), 0) AS pnl_total
            FROM trades_log {where}
        """, params)
        row = c.fetchone()
        total = row["total"] or 0
        wins = row["wins"] or 0
        losses = row["losses"] or 0
        closed = wins + losses
        return {
            "total": total,
            "wins": wins,
            "losses": losses,
            "pending": row["pending"] or 0,
            "wr": (wins / closed * 100) if closed > 0 else 0,
            "pnl_total": float(row["pnl_total"] or 0),
        }

    # ========== REJETS (DEBUG) ==========

    def log_rejected(self, instrument: str, ts: pd.Timestamp, direction: str,
                     reason: str, ml_proba: float | None = None, score: int | None = None):
        c = self.conn.cursor()
        c.execute("""
            INSERT INTO rejected_setups (instrument, ts, direction, rejection_reason, ml_proba, score)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (instrument, ts.isoformat(), direction, reason, ml_proba, score))
        self.conn.commit()

    # ========== EVENTS ==========

    def log_event(self, event_type: str, message: str):
        c = self.conn.cursor()
        c.execute(
            "INSERT INTO bot_events (event_type, message) VALUES (?, ?)",
            (event_type, message)
        )
        self.conn.commit()

    def close(self):
        self.conn.close()
