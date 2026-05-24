"""Micro-service HTTP qui sert les bougies M1 + tick courant pour le dashboard.

Tourne sur le VPS Contabo, lit directement MT5 (qui est deja initialise par
le bot principal - MT5 partage la session via IPC entre processes Python).

Endpoint :
    GET /candles?asset=XAUUSD&count=60&token=xxx
    -> {"asset": "XAUUSD", "candles": [{time, open, high, low, close}, ...],
        "tick": {time, bid, ask}, "broker_offset_min": -2504.2}

Securite : token simple en query string (defini via env CANDLE_TOKEN).
Si pas de token configure, l'endpoint est ouvert (dev only).

Usage :
    set CANDLE_TOKEN=mon_secret_random
    python candle_service.py --port 8080
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import MetaTrader5 as mt5
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from bot_v2.mt5_executor import MT5Executor, to_broker_symbol

# Executor partage (singleton) qui detecte et compense le broker UTC offset
_MT5_EXEC = MT5Executor()

# Logging vers fichier + console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(ROOT, "candle_service.log")),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("candle_service")

CANDLE_TOKEN = os.environ.get("CANDLE_TOKEN", "").strip()
if not CANDLE_TOKEN:
    log.warning("CANDLE_TOKEN env var vide -> endpoint ouvert (dev only)")

# 14 actifs live + 3 SMT
ALLOWED_ASSETS = {
    "XAUUSD", "NAS100", "GER40", "BTCUSD",
    "EURUSD", "GBPUSD", "AUDUSD", "USDJPY",
    "SP500", "DJ30", "UK100", "FRA40",
    "USDCAD", "USDCHF",
    "XAGUSD", "DXY", "SPX500",
}

app = FastAPI(title="TradingBote candle service", version="1.0")

# CORS : dashboard tourne sur Railway, doit pouvoir appeler ce service
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _check_token(token: str | None) -> None:
    if CANDLE_TOKEN and token != CANDLE_TOKEN:
        raise HTTPException(status_code=401, detail="Bad token")


@app.on_event("startup")
def _on_startup():
    """Initialise MT5 au demarrage (partage la session avec le bot principal).

    Utilise MT5Executor pour beneficier de la detection automatique du broker
    UTC offset (Vantage = GMT+3 traite comme UTC par MT5). Sans cette compensation,
    les timestamps des bougies sont en time broker, decalees de 3h vs UTC reel.
    """
    ok = _MT5_EXEC.initialize()
    if not ok:
        log.error(f"MT5Executor initialize() FAILED")
    else:
        log.info(f"MT5 connecte | broker_offset_sec={_MT5_EXEC.broker_utc_offset_sec}")


@app.get("/health")
def health():
    return {"status": "ok", "mt5_initialized": mt5.terminal_info() is not None}


@app.get("/candles")
def candles(
    asset: str = Query(..., min_length=2, max_length=12),
    count: int = Query(60, ge=1, le=500),
    token: str | None = Query(None),
):
    """Retourne les `count` dernieres bougies M1 fermees + tick courant.

    Format candles : [{time: unix_ts_sec, open, high, low, close}, ...]
    Tries par time croissant. time est en UTC seconds (compatible Lightweight Charts).
    """
    _check_token(token)
    if asset not in ALLOWED_ASSETS:
        raise HTTPException(status_code=400, detail=f"Asset not allowed: {asset}")

    broker_sym = to_broker_symbol(asset)
    # copy_rates_from_pos : recupere les `count` dernieres bougies fermees
    # start_pos=1 pour exclure la bougie en cours (non close)
    rates = mt5.copy_rates_from_pos(broker_sym, mt5.TIMEFRAME_M1, 1, count)
    if rates is None or len(rates) == 0:
        # Fallback : essayer le symbol sans suffix +
        if broker_sym != asset:
            rates = mt5.copy_rates_from_pos(asset, mt5.TIMEFRAME_M1, 1, count)
        if rates is None or len(rates) == 0:
            raise HTTPException(status_code=503, detail=f"MT5 returned no bars for {asset}")

    # Compense l'offset broker (Vantage +3h traite comme UTC) pour avoir les
    # timestamps en UTC reel (= ce que le bot envoie via push_dashboard, qui
    # passe par MT5Executor.get_bars qui fait deja cette compensation).
    offset_sec = _MT5_EXEC.broker_utc_offset_sec
    candles_list = [
        {
            "time": int(r["time"]) - offset_sec,
            "open": float(r["open"]),
            "high": float(r["high"]),
            "low": float(r["low"]),
            "close": float(r["close"]),
        }
        for r in rates
    ]

    # Tick courant pour la bougie en formation (meme compensation)
    tick = mt5.symbol_info_tick(broker_sym) or mt5.symbol_info_tick(asset)
    tick_data = None
    if tick is not None:
        tick_data = {
            "time": int(tick.time) - offset_sec,
            "bid": float(tick.bid),
            "ask": float(tick.ask),
            "last": float(tick.last) if tick.last else (float(tick.bid) + float(tick.ask)) / 2,
        }

    # Broker offset rapporte (info pour debug, le service compense deja en interne)
    utc_now = datetime.now(timezone.utc)

    return {
        "asset": asset,
        "broker_symbol": broker_sym,
        "candles": candles_list,
        "tick": tick_data,
        "broker_offset_min": round(offset_sec / 60, 1),
        "server_utc": int(utc_now.timestamp()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--host", default="0.0.0.0")
    args = p.parse_args()
    log.info(f"Starting candle_service on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
