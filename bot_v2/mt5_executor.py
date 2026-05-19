"""Module d'execution MT5 : connexion + placement ordres + lecture bougies.

Utilise le package officiel `MetaTrader5` (Windows uniquement).
Necessite MT5 desktop ouvert et connecte au compte Vantage.

Decision user 2026-05-17 :
- Bot place SL/TP cote broker (jamais en memoire seule)
- Si bot crash : le broker continue de gerer la position via SL/TP
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Literal

import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None  # type: ignore


log = logging.getLogger(__name__)


# === MAPPING SYMBOLES BROKER (user 2026-05-17) ===
# Certains brokers ajoutent un suffixe (Vantage = "+" sur XAUUSD RAW ECN).
# Le bot utilise le nom standard (XAUUSD) en interne, on traduit a la sortie/entree.
BROKER_SYMBOL_MAP: dict[str, str] = {
    # === ACTIFS V1 (deja branchees) ===
    "XAUUSD": "XAUUSD+",   # Vantage RAW ECN : suffixe +
    "SPX500": "SP500",     # Vantage : sans le X (utilise pour SMT NAS/GER)
    "DXY":    "USDX",      # Vantage : Dollar Index CFD
    # === ACTIFS V2 ajoutes 2026-05-18 (user demande) ===
    # Indices : pas de suffixe Vantage
    "DJ30":     "DJ30",
    "UK100":    "UK100",
    "FRA40":    "FRA40",
    "JP225":    "Nikkei225",   # ATTENTION : nom different chez Vantage
    # Forex majeurs USD : suffixe + sur RAW ECN
    "USDCAD":   "USDCAD+",
    "USDCHF":   "USDCHF+",
    # Les autres symboles principaux restent identiques (NAS100, GER40, BTCUSD, EURUSD, etc.)
}


def to_broker_symbol(symbol: str) -> str:
    """Convertit un nom standard (XAUUSD) en nom broker (XAUUSD+)."""
    return BROKER_SYMBOL_MAP.get(symbol, symbol)


def from_broker_symbol(broker_symbol: str) -> str:
    """Inverse : convertit XAUUSD+ -> XAUUSD pour la coherence interne."""
    for std, brok in BROKER_SYMBOL_MAP.items():
        if brok == broker_symbol:
            return std
    return broker_symbol


# === MAPPING TF (MT5 utilise des enums) ===
TF_MAP = {
    "M1":  mt5.TIMEFRAME_M1 if mt5 else None,
    "M5":  mt5.TIMEFRAME_M5 if mt5 else None,
    "M15": mt5.TIMEFRAME_M15 if mt5 else None,
    "H1":  mt5.TIMEFRAME_H1 if mt5 else None,
    "H4":  mt5.TIMEFRAME_H4 if mt5 else None,
    "D1":  mt5.TIMEFRAME_D1 if mt5 else None,
} if mt5 else {}


class MT5Executor:
    """Wrapper MT5 : initialize, fetch bars, place orders, manage positions."""

    def __init__(self):
        if mt5 is None:
            raise ImportError(
                "Package MetaTrader5 non installe. Lance: pip install MetaTrader5"
            )
        self.connected = False
        self.account_info = None

    def initialize(self, login: int | None = None, password: str | None = None,
                   server: str | None = None, path: str | None = None) -> bool:
        """Connecte au terminal MT5. Si login/password/server fournis, login explicite,
        sinon utilise le terminal MT5 deja ouvert et connecte.
        """
        if login and password and server:
            ok = mt5.initialize(login=login, password=password, server=server, path=path)
        else:
            ok = mt5.initialize(path=path) if path else mt5.initialize()

        if not ok:
            err = mt5.last_error()
            log.error(f"MT5 initialize() FAILED: {err}")
            return False

        self.account_info = mt5.account_info()
        if self.account_info is None:
            log.error("MT5 connected mais account_info() = None (pas de compte logge ?)")
            mt5.shutdown()
            return False

        self.connected = True
        log.info(
            f"MT5 connecte | login={self.account_info.login} "
            f"balance={self.account_info.balance:.2f} {self.account_info.currency} "
            f"server={self.account_info.server}"
        )
        return True

    def shutdown(self):
        if mt5 and self.connected:
            mt5.shutdown()
            self.connected = False

    # ========== ACCOUNT ==========

    def get_balance(self) -> float:
        info = mt5.account_info()
        return float(info.balance) if info else 0.0

    def get_equity(self) -> float:
        info = mt5.account_info()
        return float(info.equity) if info else 0.0

    # ========== BOUGIES ==========

    def get_bars(self, symbol: str, tf: str, n: int = 500) -> pd.DataFrame | None:
        """Recupere les N dernieres bougies cloturees pour symbol+TF.

        Retourne un DataFrame compatible avec le pipeline existant
        (colonnes: open, high, low, close, volume + index timestamp UTC).
        """
        if tf not in TF_MAP:
            log.error(f"TF inconnu : {tf}")
            return None

        broker_sym = to_broker_symbol(symbol)
        rates = mt5.copy_rates_from_pos(broker_sym, TF_MAP[tf], 0, n)
        if rates is None or len(rates) == 0:
            log.debug(f"Pas de data pour {symbol} {tf} : {mt5.last_error()}")
            return None

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("time")
        df = df.rename(columns={"tick_volume": "volume"})
        # Garde seulement les cols standards
        return df[["open", "high", "low", "close", "volume"]].copy()

    def get_bars_range(self, symbol: str, tf: str,
                       start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame | None:
        """Bougies sur plage de dates. Utile pour reload historique."""
        if tf not in TF_MAP:
            return None
        broker_sym = to_broker_symbol(symbol)
        rates = mt5.copy_rates_range(
            broker_sym, TF_MAP[tf],
            start.to_pydatetime(), end.to_pydatetime(),
        )
        if rates is None or len(rates) == 0:
            return None
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("time")
        df = df.rename(columns={"tick_volume": "volume"})
        return df[["open", "high", "low", "close", "volume"]].copy()

    # ========== SYMBOLES ==========

    def symbol_info(self, symbol: str):
        """Info MT5 d'un symbole : volume_min, volume_step, point, digits, etc."""
        return mt5.symbol_info(to_broker_symbol(symbol))

    def ensure_symbol_active(self, symbol: str) -> bool:
        """Active le symbole dans Market Watch (necessaire pour fetch bars)."""
        broker_sym = to_broker_symbol(symbol)
        info = mt5.symbol_info(broker_sym)
        if info is None:
            log.error(f"Symbole inconnu : {symbol} (broker={broker_sym})")
            return False
        if not info.visible:
            ok = mt5.symbol_select(broker_sym, True)
            if not ok:
                log.error(f"Impossible d'activer {broker_sym}")
                return False
        return True

    def current_price(self, symbol: str, direction: str) -> float | None:
        """Prix actuel ASK (pour BUY) ou BID (pour SELL)."""
        tick = mt5.symbol_info_tick(to_broker_symbol(symbol))
        if tick is None:
            return None
        if direction == "bullish":
            return float(tick.ask)
        return float(tick.bid)

    # ========== POSITIONS ==========

    def get_positions(self, symbol: str | None = None) -> list[dict]:
        """Liste des positions ouvertes. Si symbol fourni, filtre."""
        if symbol:
            positions = mt5.positions_get(symbol=to_broker_symbol(symbol))
        else:
            positions = mt5.positions_get()
        if positions is None:
            return []
        return [
            {
                "ticket": p.ticket,
                "symbol": from_broker_symbol(p.symbol),  # standardise
                "broker_symbol": p.symbol,
                "type": "BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL",
                "volume": p.volume,
                "price_open": p.price_open,
                "sl": p.sl,
                "tp": p.tp,
                "profit": p.profit,
                "time": pd.Timestamp(p.time, unit="s", tz="UTC"),
                "comment": p.comment,
            }
            for p in positions
        ]

    def get_n_open_positions(self) -> int:
        """Nombre total de positions ouvertes (tous symboles)."""
        positions = mt5.positions_get()
        return len(positions) if positions else 0

    # ========== ORDRES ==========

    def place_market_order(
        self,
        symbol: str,
        direction: Literal["bullish", "bearish"],
        volume: float,
        sl: float,
        tp: float,
        comment: str = "Vizion",
        magic: int = 20260517,
    ) -> dict | None:
        """Place un ordre MARKET avec SL et TP cote broker.

        Returns:
            {"ticket": int, "price": float} si succes, None si echec.
        """
        if not self.ensure_symbol_active(symbol):
            return None

        broker_sym = to_broker_symbol(symbol)
        info = self.symbol_info(symbol)
        if info is None:
            log.error(f"symbol_info({symbol}) = None")
            return None

        # Ajuste volume au step / min du broker
        vol_min = info.volume_min
        vol_step = info.volume_step
        volume = max(vol_min, round(volume / vol_step) * vol_step)
        volume = round(volume, 2)

        # Recupere prix courant
        tick = mt5.symbol_info_tick(broker_sym)
        if tick is None:
            log.error(f"Pas de tick pour {broker_sym}")
            return None

        if direction == "bullish":
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": broker_sym,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,  # slippage max en points
            "magic": magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None:
            log.error(f"order_send None : {mt5.last_error()}")
            return None

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            log.error(
                f"Order REJECTED {symbol} {direction} vol={volume} "
                f"retcode={result.retcode} : {result.comment}"
            )
            return None

        log.info(
            f"ORDER OK {symbol} {direction} vol={volume} @ {result.price:.5f} "
            f"SL={sl:.5f} TP={tp:.5f} ticket={result.order}"
        )
        return {
            "ticket": result.order,
            "price": float(result.price),
            "volume": float(result.volume),
        }

    def place_limit_order(
        self,
        symbol: str,
        direction: Literal["bullish", "bearish"],
        volume: float,
        entry_price: float,
        sl: float,
        tp: float,
        expiration_minutes: int = 60,
        comment: str = "Vizion_LIMIT",
        magic: int = 20260517,
    ) -> dict | None:
        """Place un ordre LIMIT (pending) au prix entry_price avec SL/TP.

        BUY LIMIT  : ordre place SOUS le prix actuel (achat moins cher si pullback)
        SELL LIMIT : ordre place AU-DESSUS du prix actuel (vente plus haut si rebond)

        L'ordre attend que le prix touche entry_price puis s'execute automatiquement.
        Expire apres expiration_minutes si non-touche.

        Equivalent du `setup.entry_price` du backtest (fill au prix de l'OB).
        """
        if not self.ensure_symbol_active(symbol):
            return None

        broker_sym = to_broker_symbol(symbol)
        info = self.symbol_info(symbol)
        if info is None:
            log.error(f"symbol_info({symbol}) = None")
            return None

        vol_min = info.volume_min
        vol_step = info.volume_step
        volume = max(vol_min, round(volume / vol_step) * vol_step)
        volume = round(volume, 2)

        tick = mt5.symbol_info_tick(broker_sym)
        if tick is None:
            log.error(f"Pas de tick pour {broker_sym}")
            return None

        # Detection automatique du sens du LIMIT selon direction et prix actuel
        if direction == "bullish":
            # BUY : on veut acheter. Si entry < prix actuel -> BUY LIMIT (pullback)
            #                       Si entry > prix actuel -> BUY STOP (breakout)
            if entry_price < tick.ask:
                order_type = mt5.ORDER_TYPE_BUY_LIMIT
            else:
                order_type = mt5.ORDER_TYPE_BUY_STOP
        else:
            # SELL : on veut vendre. Si entry > prix actuel -> SELL LIMIT (rebond)
            #                        Si entry < prix actuel -> SELL STOP (breakdown)
            if entry_price > tick.bid:
                order_type = mt5.ORDER_TYPE_SELL_LIMIT
            else:
                order_type = mt5.ORDER_TYPE_SELL_STOP

        from datetime import datetime, timedelta, timezone
        expiration = datetime.now(timezone.utc) + timedelta(minutes=expiration_minutes)

        request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": broker_sym,
            "volume": volume,
            "type": order_type,
            "price": float(entry_price),
            "sl": float(sl),
            "tp": float(tp),
            "deviation": 20,
            "magic": magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_SPECIFIED,
            "expiration": int(expiration.timestamp()),
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None:
            log.error(f"order_send None : {mt5.last_error()}")
            return None

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            log.error(
                f"LIMIT REJECTED {symbol} {direction} vol={volume} entry={entry_price} "
                f"retcode={result.retcode} : {result.comment}"
            )
            return None

        log.info(
            f"LIMIT OK {symbol} {direction} vol={volume} entry={entry_price:.5f} "
            f"SL={sl:.5f} TP={tp:.5f} expire={expiration_minutes}min ticket={result.order}"
        )
        return {
            "ticket": result.order,
            "price": float(entry_price),
            "volume": float(volume),
        }

    def get_pending_orders(self, magic: int | None = None) -> list[dict]:
        """Liste des ordres pending (LIMIT non encore remplis)."""
        orders = mt5.orders_get()
        if orders is None:
            return []
        out = []
        for o in orders:
            if magic and o.magic != magic:
                continue
            out.append({
                "ticket": o.ticket,
                "symbol": from_broker_symbol(o.symbol),
                "type": o.type,
                "volume": o.volume_initial,
                "price_open": o.price_open,
                "sl": o.sl,
                "tp": o.tp,
                "time_setup": pd.Timestamp(o.time_setup, unit="s", tz="UTC"),
            })
        return out

    def cancel_pending_order(self, ticket: int) -> bool:
        """Annule un ordre pending."""
        request = {
            "action": mt5.TRADE_ACTION_REMOVE,
            "order": ticket,
        }
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            log.error(f"Cancel pending {ticket} fail : {result.comment if result else 'None'}")
            return False
        log.info(f"Pending order {ticket} annule")
        return True

    def close_position(self, ticket: int) -> bool:
        """Ferme une position au market. Rarement utilise (le broker gere via SL/TP)."""
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return False
        p = positions[0]
        tick = mt5.symbol_info_tick(p.symbol)
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": p.symbol,
            "volume": p.volume,
            "type": mt5.ORDER_TYPE_SELL if p.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
            "position": ticket,
            "price": tick.bid if p.type == mt5.ORDER_TYPE_BUY else tick.ask,
            "deviation": 20,
            "magic": p.magic,
            "comment": "manual_close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        return result is not None and result.retcode == mt5.TRADE_RETCODE_DONE

    # ========== HISTORIQUE DEALS (pour journal) ==========

    def get_closed_deals(self, from_ts: pd.Timestamp, to_ts: pd.Timestamp,
                        magic: int | None = None) -> list[dict]:
        """Recupere les deals fermes entre 2 timestamps."""
        deals = mt5.history_deals_get(from_ts.to_pydatetime(), to_ts.to_pydatetime())
        if deals is None:
            return []
        out = []
        for d in deals:
            if magic and d.magic != magic:
                continue
            out.append({
                "ticket": d.ticket,
                "position_id": d.position_id,
                "order": d.order,  # ID ordre (peut differer du ticket sur certains brokers)
                "symbol": d.symbol,
                "type": d.type,
                "volume": d.volume,
                "price": d.price,
                "profit": d.profit,
                "time": pd.Timestamp(d.time, unit="s", tz="UTC"),
                "comment": d.comment,
            })
        return out
