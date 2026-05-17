"""Portefeuille fictif pour simuler les trades du bot.

Risk par trade : RISK_PER_TRADE * balance (= 1% par defaut).
Lot size = ajuste pour que la perte au SL = risk_amount.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import pandas as pd

from config import INITIAL_BALANCE, RISK_PER_TRADE

TradeStatus = Literal["pending", "open", "win", "loss", "cancelled"]
Direction = Literal["bullish", "bearish"]


@dataclass
class FictivePortfolio:
    balance: float = INITIAL_BALANCE
    initial_balance: float = INITIAL_BALANCE
    trades_count: int = 0
    wins: int = 0
    losses: int = 0

    def position_size(self, entry: float, stop_loss: float, tick_value: float = 100.0) -> float:
        """Calcule la taille de position pour risquer RISK_PER_TRADE * balance.

        Args:
            tick_value: $ par unite de mouvement par lot (XAU=100, NAS=1, OIL=1000, etc.)
        """
        risk_amount = self.balance * RISK_PER_TRADE
        stop_distance = abs(entry - stop_loss)
        if stop_distance <= 0:
            return 0.0
        lots = risk_amount / (stop_distance * tick_value)
        return round(lots, 2)

    def apply_result(self, pnl: float, is_win: bool) -> None:
        self.balance += pnl
        self.trades_count += 1
        if is_win:
            self.wins += 1
        else:
            self.losses += 1

    @property
    def winrate(self) -> float:
        return self.wins / self.trades_count if self.trades_count else 0.0

    @property
    def pnl_total(self) -> float:
        return self.balance - self.initial_balance

    @property
    def pnl_pct(self) -> float:
        return (self.balance / self.initial_balance - 1) * 100


@dataclass
class SimulatedTrade:
    """Resultat simule d'un trade en backtest."""
    id: str
    direction: Direction
    instrument: str
    entry_time: pd.Timestamp
    entry_price: float
    stop_loss: float
    take_profit: float
    lot_size: float
    risk_reward: float

    status: TradeStatus = "pending"
    exit_time: pd.Timestamp | None = None
    exit_price: float | None = None
    pnl: float = 0.0

    # contexte detection (pour la revue)
    htf_strength: str = "neutral"
    ob_type: str = "order_block"
    touch_count: int = 0
    notes: str = ""

    # zones pour le chart (serializable JSON)
    chart_overlays: dict = field(default_factory=dict)

    # Scoring V2
    score: int = 0
    verdict: str = "weak"
    setup_analysis: dict = field(default_factory=dict)
    would_trade: bool = False     # True si filtres durs OK + score >= threshold
    tick_value: float = 100.0     # $ par unite de mouvement par lot

    def simulate_outcome(self, df: pd.DataFrame) -> None:
        """Simule le resultat du trade en regardant les bougies post-entree.

        On regarde quelle limite (SL ou TP) est touchee en premier.
        """
        try:
            entry_loc = df.index.get_loc(self.entry_time)
        except KeyError:
            return

        sub = df.iloc[int(entry_loc) + 1:]
        for ts, row in sub.iterrows():
            if self.direction == "bullish":
                if row["low"] <= self.stop_loss:
                    self.exit_time = ts
                    self.exit_price = self.stop_loss
                    self.status = "loss"
                    break
                if row["high"] >= self.take_profit:
                    self.exit_time = ts
                    self.exit_price = self.take_profit
                    self.status = "win"
                    break
            else:
                if row["high"] >= self.stop_loss:
                    self.exit_time = ts
                    self.exit_price = self.stop_loss
                    self.status = "loss"
                    break
                if row["low"] <= self.take_profit:
                    self.exit_time = ts
                    self.exit_price = self.take_profit
                    self.status = "win"
                    break

        if self.exit_price is not None:
            move = (self.exit_price - self.entry_price) if self.direction == "bullish" else (self.entry_price - self.exit_price)
            self.pnl = move * self.lot_size * self.tick_value
