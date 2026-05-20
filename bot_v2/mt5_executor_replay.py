"""MT5 Executor REPLAY : mock qui sert les bougies Vantage parquet au lieu de MT5 live.

Implemente la meme API que MT5Executor mais :
- get_bars() retourne les N dernieres bougies du parquet JUSQU'A `self.now`
- place_limit_order() enregistre dans une liste virtuelle
- get_pending_orders() / get_closed_deals() renvoient les ordres virtuels
- broker_utc_offset_sec = 0 (Vantage parquets deja en UTC apres export)

Usage :
    from bot_v2.mt5_executor_replay import MT5ExecutorReplay
    mt5 = MT5ExecutorReplay(data_dir="data_vantage")
    mt5.initialize()
    mt5.set_now(pd.Timestamp("2025-11-01 10:00", tz="UTC"))
    df = mt5.get_bars("XAUUSD", "M1", 200)  # 200 dernieres bougies <= now
    mt5.advance(pd.Timedelta(minutes=1))  # avance d'1 min
"""
from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class _AccountInfo:
    login: int = 99999999
    server: str = "Vantage-Replay"
    currency: str = "EUR"
    balance: float = 154.96  # 104.96 + bonus 50


@dataclass
class _VirtualOrder:
    """Ordre LIMIT virtuel (pas encore fill)."""
    ticket: int
    symbol: str
    direction: str  # "bullish"|"bearish"
    type: str       # "BUY_LIMIT"|"SELL_LIMIT"
    entry: float
    sl: float
    tp: float
    lots: float
    magic: int
    comment: str
    time_setup: pd.Timestamp
    state: str = "pending"  # "pending"|"filled"|"cancelled"|"closed"
    fill_time: Optional[pd.Timestamp] = None
    fill_price: Optional[float] = None
    close_time: Optional[pd.Timestamp] = None
    close_price: Optional[float] = None
    outcome: Optional[str] = None  # "WIN"|"LOSS"
    pnl_usd: float = 0.0


class MT5ExecutorReplay:
    """Mock MT5 qui replay les donnees Vantage parquet."""

    def __init__(self, data_dir: str | Path = "data_vantage", root: Optional[str] = None):
        self.root = Path(root) if root else Path(os.path.dirname(os.path.abspath(__file__))).parent
        self.data_dir = self.root / data_dir
        self.now: pd.Timestamp = pd.Timestamp("1970-01-01", tz="UTC")
        self.broker_utc_offset_sec = 0  # parquets deja en UTC
        self.account_info = _AccountInfo()

        # Cache des dataframes complets
        self._data_cache: dict[tuple[str, str], pd.DataFrame] = {}

        # Ordres virtuels
        self._orders: list[_VirtualOrder] = []
        self._next_ticket: int = 1000000

    def initialize(self, *args, **kwargs) -> bool:
        return True

    def shutdown(self):
        pass

    def set_now(self, ts: pd.Timestamp):
        """Definit l'horloge simulee."""
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        self.now = ts

    def advance(self, delta: pd.Timedelta):
        self.now = self.now + delta

    def get_balance(self) -> float:
        return self.account_info.balance

    def _load_df(self, symbol: str, tf: str) -> Optional[pd.DataFrame]:
        key = (symbol, tf)
        if key in self._data_cache:
            return self._data_cache[key]
        path = self.data_dir / f"{symbol}_{tf}.parquet"
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        self._data_cache[key] = df
        return df

    def get_bars(self, symbol: str, tf: str, n: int = 500, force_sync: bool = False) -> Optional[pd.DataFrame]:
        """Retourne les N dernieres bougies <= self.now."""
        df = self._load_df(symbol, tf)
        if df is None:
            return None
        # On veut les bougies dont la close <= now (i.e. bougies completes)
        # La bougie M1 14:30 contient les ticks de 14:30:00 a 14:30:59,
        # elle est "close" a 14:31:00. Donc on peut la voir si now >= 14:31:00.
        # Approximation : on prend les bougies dont l'index <= now - 1min (pour M1).
        # Plus simple : index <= now (le bot fait deja df.iloc[:-1] pour virer la bougie en cours)
        df_view = df[df.index <= self.now]
        if len(df_view) == 0:
            return None
        return df_view.tail(n).copy()

    def get_bars_range(self, symbol: str, tf: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> Optional[pd.DataFrame]:
        df = self._load_df(symbol, tf)
        if df is None:
            return None
        mask = (df.index >= start_ts) & (df.index <= end_ts) & (df.index <= self.now)
        return df[mask].copy()

    # ============ ORDRES VIRTUELS ============

    def place_limit_order(self, symbol: str, order_type: str, entry: float, sl: float, tp: float,
                          lots: float, magic: int = 0, comment: str = "", **kwargs) -> dict:
        """Enregistre l'ordre dans la liste virtuelle."""
        direction = "bullish" if order_type == "BUY_LIMIT" else "bearish"
        ticket = self._next_ticket
        self._next_ticket += 1
        order = _VirtualOrder(
            ticket=ticket, symbol=symbol, direction=direction, type=order_type,
            entry=float(entry), sl=float(sl), tp=float(tp), lots=float(lots),
            magic=magic, comment=comment, time_setup=self.now,
        )
        self._orders.append(order)
        return {"retcode": 10009, "order": ticket, "comment": "OK_VIRTUAL"}

    def get_pending_orders(self, magic: int | None = None) -> list[dict]:
        out = []
        for o in self._orders:
            if o.state != "pending":
                continue
            if magic is not None and o.magic != magic:
                continue
            out.append({
                "ticket": o.ticket,
                "symbol": o.symbol,
                "type": o.type,
                "price_open": o.entry,
                "sl": o.sl,
                "tp": o.tp,
                "volume_initial": o.lots,
                "time_setup": int(o.time_setup.timestamp()),
                "magic": o.magic,
                "comment": o.comment,
            })
        return out

    def cancel_order(self, ticket: int) -> bool:
        for o in self._orders:
            if o.ticket == ticket and o.state == "pending":
                o.state = "cancelled"
                return True
        return False

    def get_open_positions(self, magic: int | None = None) -> list[dict]:
        out = []
        for o in self._orders:
            if o.state != "filled":
                continue
            if magic is not None and o.magic != magic:
                continue
            out.append({
                "ticket": o.ticket, "symbol": o.symbol, "type": o.type,
                "price_open": o.fill_price, "sl": o.sl, "tp": o.tp,
                "volume": o.lots, "magic": o.magic,
            })
        return out

    def get_closed_deals(self, from_ts: pd.Timestamp, to_ts: pd.Timestamp, magic: int | None = None) -> list[dict]:
        out = []
        for o in self._orders:
            if o.state != "closed":
                continue
            if magic is not None and o.magic != magic:
                continue
            if o.close_time is None or o.close_time < from_ts or o.close_time > to_ts:
                continue
            out.append({
                "ticket": o.ticket, "symbol": o.symbol, "type": o.type,
                "entry": o.fill_price, "close": o.close_price,
                "time": int(o.close_time.timestamp()),
                "profit": o.pnl_usd, "magic": o.magic, "outcome": o.outcome,
            })
        return out

    # ============ FILL / SL / TP SIMULATION ============

    def _tick_value(self, symbol: str) -> float:
        """tick_value approximatif pour calcul P&L."""
        from bot_v2.config import INSTRUMENTS
        return float(INSTRUMENTS.get(symbol, {}).get("tick_value", 1.0))

    def update_orders(self):
        """Avance l'etat des ordres : fill / SL / TP / expiration.

        A appeler apres chaque set_now(). Verifie pour chaque ordre pending
        si la bougie M1 a la timestamp current a touche l'entry/SL/TP.
        """
        for o in self._orders:
            if o.state in ("cancelled", "closed"):
                continue

            # Charge la bougie M1 a self.now (ou plus recente <= now)
            df_m1 = self._load_df(o.symbol, "M1")
            if df_m1 is None:
                continue
            df_view = df_m1[(df_m1.index > o.time_setup) & (df_m1.index <= self.now)]
            if len(df_view) == 0:
                continue

            for ts, bar in df_view.iterrows():
                high, low = float(bar["high"]), float(bar["low"])

                if o.state == "pending":
                    # Check fill
                    if o.type == "BUY_LIMIT" and low <= o.entry:
                        o.state = "filled"
                        o.fill_time = ts
                        o.fill_price = o.entry
                    elif o.type == "SELL_LIMIT" and high >= o.entry:
                        o.state = "filled"
                        o.fill_time = ts
                        o.fill_price = o.entry
                    # Expiration : 30 min sans fill
                    elif (ts - o.time_setup).total_seconds() > 30 * 60:
                        o.state = "cancelled"
                        break
                    if o.state != "filled":
                        continue

                # Si on est filled, on check SL/TP sur cette meme bougie (ou suivantes)
                if o.state == "filled" and ts >= o.fill_time:
                    tv = self._tick_value(o.symbol)
                    if o.direction == "bullish":
                        # SL touche en premier ? (low <= sl)
                        if low <= o.sl:
                            o.state = "closed"
                            o.close_time = ts
                            o.close_price = o.sl
                            o.outcome = "LOSS"
                            o.pnl_usd = (o.sl - o.fill_price) * o.lots * tv
                            break
                        elif high >= o.tp:
                            o.state = "closed"
                            o.close_time = ts
                            o.close_price = o.tp
                            o.outcome = "WIN"
                            o.pnl_usd = (o.tp - o.fill_price) * o.lots * tv
                            break
                    else:
                        # bearish
                        if high >= o.sl:
                            o.state = "closed"
                            o.close_time = ts
                            o.close_price = o.sl
                            o.outcome = "LOSS"
                            o.pnl_usd = (o.fill_price - o.sl) * o.lots * tv
                            break
                        elif low <= o.tp:
                            o.state = "closed"
                            o.close_time = ts
                            o.close_price = o.tp
                            o.outcome = "WIN"
                            o.pnl_usd = (o.fill_price - o.tp) * o.lots * tv
                            break

    # ============ STATS ============

    def stats_recap(self) -> dict:
        n = len(self._orders)
        filled = [o for o in self._orders if o.state == "filled"]
        closed = [o for o in self._orders if o.state == "closed"]
        cancelled = [o for o in self._orders if o.state == "cancelled"]
        wins = [o for o in closed if o.outcome == "WIN"]
        losses = [o for o in closed if o.outcome == "LOSS"]
        total_pnl = sum(o.pnl_usd for o in closed)
        wr = len(wins) / len(closed) * 100 if closed else 0
        return {
            "total": n,
            "filled": len(filled),
            "closed": len(closed),
            "cancelled": len(cancelled),
            "wins": len(wins),
            "losses": len(losses),
            "wr": wr,
            "pnl_usd": total_pnl,
        }
