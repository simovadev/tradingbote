"""Push asynchrone des events du bot vers le dashboard Railway.

Architecture : queue + thread daemon = jamais bloquant pour le scan principal.
Si le dashboard est down, les events sont droppes silencieusement (au pire
la queue se vide d'elle-meme via le worker, ou on drop si maxsize atteint).

Usage :
    from bot_v2.push_dashboard import DashboardPusher
    pusher = DashboardPusher(url=os.getenv("DASHBOARD_URL"), session_id="xxxxx")
    pusher.push_setup(...)
    pusher.push_trade_executed(...)
    pusher.shutdown()  # dans le finally
"""
from __future__ import annotations

import logging
import os
import queue
import threading
from datetime import datetime, timezone
from typing import Any

try:
    import requests  # dependance optionnelle (pas requise si url=None)
except ImportError:
    requests = None  # type: ignore


log = logging.getLogger("push_dashboard")


class DashboardPusher:
    """Push non-bloquant vers le dashboard FastAPI (Railway).

    Si `url` est None ou vide, devient un no-op silencieux (dev local sans
    dashboard). Aucune exception remontee a l'appelant.
    """

    def __init__(
        self,
        url: str | None,
        session_id: str,
        timeout_sec: float = 2.0,
        max_queue_size: int = 500,
    ) -> None:
        self.url = (url or "").strip() or None
        self.session_id = session_id
        self.timeout_sec = timeout_sec
        self._queue: queue.Queue[dict] = queue.Queue(maxsize=max_queue_size)
        self._running = False
        self._worker: threading.Thread | None = None

        if self.url is None:
            log.info("DashboardPusher: URL vide -> mode no-op (pas de push)")
            return

        if requests is None:
            log.warning("DashboardPusher: module `requests` absent -> mode no-op")
            self.url = None
            return

        self._running = True
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()
        log.info(f"DashboardPusher: actif vers {self.url} (session={session_id})")

    def _worker_loop(self) -> None:
        while self._running:
            try:
                msg = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                requests.post(self.url, json=msg, timeout=self.timeout_sec)  # type: ignore[union-attr]
            except Exception as e:
                log.debug(f"push fail ({type(e).__name__}): {str(e)[:120]}")
            finally:
                self._queue.task_done()

    def _enqueue(self, msg: dict) -> None:
        """Ajoute un event. Si la queue est pleine -> drop silencieux."""
        if self.url is None:
            return
        msg.setdefault("session_id", self.session_id)
        msg.setdefault("ts", datetime.now(timezone.utc).isoformat())
        try:
            self._queue.put_nowait(msg)
        except queue.Full:
            log.warning("DashboardPusher: queue pleine, drop")

    # ============ API publique ============

    def push_event(self, type_: str, data: dict[str, Any] | None = None,
                   instrument: str | None = None) -> None:
        """Push generique."""
        self._enqueue({
            "type": type_,
            "instrument": instrument,
            "data": data or {},
        })

    def push_start(self, message: str = "") -> None:
        # session_id duplique dans data pour le dashboard (qui ne voit que data)
        self.push_event("START", {"message": message, "session_id": self.session_id})

    def push_stop(self, message: str = "") -> None:
        self.push_event("STOP", {"message": message})

    def push_error(self, error_msg: str) -> None:
        self.push_event("ERROR", {"error_msg": str(error_msg)[:1000]})

    def push_cycle(
        self,
        actifs_scanned: int,
        total_s: float,
        fetch_s: float,
        compute_s: float,
        latencies: dict[str, int],
        last_bar_ts: Any = None,
    ) -> None:
        """Push apres CYCLE scan (1 message qui contient les 14 latences).

        last_bar_ts : timestamp de la derniere bougie M1 (toute reference,
        ex. XAUUSD) pour afficher cote dashboard "derniere bougie".
        """
        data = {
            "actifs_scanned": actifs_scanned,
            "total_s": round(total_s, 2),
            "fetch_s": round(fetch_s, 2),
            "compute_s": round(compute_s, 2),
            "latencies": {k: int(v) for k, v in latencies.items()},
        }
        if last_bar_ts is not None:
            data["last_bar_ts"] = last_bar_ts.isoformat() if hasattr(last_bar_ts, "isoformat") else str(last_bar_ts)
        self.push_event("CYCLE", data)

    def push_diag(self, instrument: str, data: dict[str, Any]) -> None:
        self.push_event("DIAG", data, instrument=instrument)

    def push_setup(
        self,
        instrument: str,
        ts: Any,
        direction: str,
        entry_price: float,
        sl: float,
        tp: float,
        rr: float,
        score: int,
        ml_proba: float,
        killzone: str | None = None,
        candles: list[dict] | None = None,
    ) -> None:
        ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
        data = {
            "ts": ts_str,
            "direction": direction,
            "entry_price": float(entry_price),
            "sl": float(sl),
            "tp": float(tp),
            "rr": float(rr),
            "score": int(score),
            "ml_proba": float(ml_proba),
            "killzone": killzone,
        }
        if candles:
            data["candles"] = candles
        self.push_event("SETUP", data, instrument=instrument)

    def push_rejected_batch(self, items: list[dict[str, Any]],
                            candles_by_asset: dict[str, list[dict]] | None = None) -> None:
        """Batch des rejets ML d'un cycle (1 push contient tous les rejets).

        candles_by_asset : optionnel, {asset: [bougies M1]} pour que le dashboard
        affiche un chart Lightweight au clic. 1 copie par actif (partagee entre
        tous les rejets de cet actif), pas par item.
        """
        if not items:
            return
        payload = {"items": items}
        if candles_by_asset:
            payload["candles_by_asset"] = candles_by_asset
        self.push_event("REJECTED", payload)

    def push_trade_executed(
        self,
        instrument: str,
        ticket: int,
        direction: str,
        entry: float,
        sl: float,
        tp: float,
        volume: float,
        rr: float,
        ml_proba: float,
        score: int,
        killzone: str | None = None,
    ) -> None:
        self.push_event("TRADE_EXECUTED", {
            "ticket": int(ticket),
            "direction": direction,
            "entry": float(entry),
            "sl": float(sl),
            "tp": float(tp),
            "volume": float(volume),
            "rr": float(rr),
            "ml_proba": float(ml_proba),
            "score": int(score),
            "killzone": killzone,
            "opened_ts": datetime.now(timezone.utc).isoformat(),
        }, instrument=instrument)

    def push_trade_closed(
        self,
        ticket: int,
        outcome: str,
        pnl_real: float,
        duration_min: float,
        instrument: str | None = None,
    ) -> None:
        self.push_event("TRADE_CLOSED", {
            "ticket": int(ticket),
            "outcome": outcome,
            "pnl_real": float(pnl_real),
            "duration_min": round(float(duration_min), 1),
            "closed_ts": datetime.now(timezone.utc).isoformat(),
        }, instrument=instrument)

    def push_stats(
        self,
        total: int,
        wins: int,
        losses: int,
        wr_pct: float,
        pnl_total: float,
        balance: float,
        equity: float | None = None,
        positions_open: int = 0,
        pending_orders: int = 0,
    ) -> None:
        self.push_event("STATS", {
            "total_trades": int(total),
            "wins": int(wins),
            "losses": int(losses),
            "wr_pct": round(float(wr_pct), 1),
            "pnl_total": round(float(pnl_total), 2),
            "balance": round(float(balance), 2),
            "equity": round(float(equity), 2) if equity is not None else None,
            "positions_open": int(positions_open),
            "pending_orders": int(pending_orders),
        })

    def push_positions_sync(
        self,
        open_positions: list[dict[str, Any]],
        pending_tickets: list[int],
    ) -> None:
        """Envoie l'etat reel MT5 a chaque cycle.

        open_positions : [{"ticket": int, "pnl": float}] positions ouvertes.
        pending_tickets : [int] tickets des ordres LIMIT encore en attente.

        Le dashboard marque FILLED les positions ouvertes (avec pnl flottant)
        et CANCELLED les trades PENDING dont l'ordre a disparu de MT5.
        """
        self.push_event("POSITIONS_SYNC", {
            "open_positions": [
                {"ticket": int(p["ticket"]), "pnl": float(p.get("pnl", 0.0))}
                for p in open_positions
            ],
            "pending_tickets": [int(t) for t in pending_tickets],
        })

    def shutdown(self, wait_drain_sec: float = 2.0) -> None:
        """Arrete le worker proprement (a appeler dans le finally de main)."""
        if not self._running:
            return
        # Laisse le worker drainer la queue
        try:
            self._queue.join()
        except Exception:
            pass
        self._running = False
        if self._worker:
            self._worker.join(timeout=wait_drain_sec)
        log.info("DashboardPusher: shutdown")


def make_pusher_from_env(session_id: str) -> DashboardPusher:
    """Helper : cree un pusher depuis la variable d'env DASHBOARD_URL."""
    url = os.getenv("DASHBOARD_URL")
    return DashboardPusher(url=url, session_id=session_id)
