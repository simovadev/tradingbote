"""LIVE RUNNER V21 MONSTER - bot avec Transformer profond + cross-asset.

Difference vs V20 :
- Inference V21MonsterNet (9.2M params)
- Fetch 5 actifs ref M15 (XAUUSD, USDJPY, SP500, BTCUSD, USDCHF) pour cross-attention
- Bougies brutes en input (pas de features ICT pre-calculees pour le model)
- Magic 21000 (vs 20000 V20)

Pipeline ICT toujours utilise en amont (filter qualite OBs) - V21 predict sur
les setups ICT valides.

Usage : python -m bot_v2.live_runner_v21
"""
from __future__ import annotations

import torch as _TORCH_EAGER

import os
import sys
import time
import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import MetaTrader5 as mt5

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("DASHBOARD_URL", "https://tradingbote-production.up.railway.app/api/ingest")
os.environ.setdefault("BOT_THRESHOLD", "0.65")
os.environ.setdefault("RECENT_CUTOFF_MIN", "60")

# Logging
log = logging.getLogger("live_v21")
log.setLevel(logging.INFO)
_fh = logging.FileHandler(ROOT / "live_v21.log", mode="a", encoding="utf-8")
_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
log.addHandler(_fh)
_sh = logging.StreamHandler()
_sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
log.addHandler(_sh)

from bot_v2.config import INSTRUMENTS, SMT_PAIRS, get_param
from bot_v2.concepts.order_block import detect_order_blocks
from bot_v2.concepts.killzones import killzone_at
from bot_v2.concepts.fvg import detect_fvg
from bot_v2.concepts.structure import detect_structure_breaks
from bot_v2.concepts.liquidity import find_swings
from bot_v2.concepts.mss_setup import detect_mss_setups
from bot_v2.concepts.daily_bias import build_d1_from_h1
from bot_v2.pipeline import evaluate_ob
from bot_v2.mt5_executor import MT5Executor
from bot_v2.v21_inference import V21Predictor, REF_ASSETS, M15_LEN
from bot_v2.v20_inference import V20Predictor  # shadow: affiche aussi proba V20 pour comparaison
from bot_v2.push_dashboard import DashboardPusher
from bot_v2.spread_logger import SpreadLogger
from bot_v2.trade_audit_logger import TradeAuditLogger

# Mapping nom V20 -> nom broker MT5 (meme que V20)
BROKER_MAP = {
    "XAUUSD":    "XAUUSD+",
    "EURUSD":    "EURUSD+",
    "GBPUSD":    "GBPUSD+",
    "USDJPY":    "USDJPY+",
    "USDCHF":    "USDCHF+",
    "USDCAD":    "USDCAD+",
    "AUDUSD":    "AUDUSD+",
    "NZDUSD":    "NZDUSD+",
    "USDMXN":    "USDMXN+",
    "BTCUSD":    "BTCUSD",
    # ETHUSD : retire (spread trop large -> setups annules systematiquement)
    "CL-OIL":    "CL-OIL",
    "NAS100":    "NAS100",
    "XAGUSD":    "XAGUSD",
    "DJ30":      "DJ30",
    "GAS-C":     "GAS-C",
    "HK50":      "HK50",
    "GER40":     "GER40",
    "FRA40":     "FRA40",
    "UK100":     "UK100",
    "Nikkei225": "Nikkei225",
    "BVSPX":     "BVSPX",
    "SP500":     "SP500",
    "Coffee-C":  "Coffee-C",
    "Cocoa-C":   "Cocoa-C",
    "Sugar-C":   "Sugar-C",
    "EURJPY":   "EURJPY+", "EURGBP":   "EURGBP+", "EURCHF":   "EURCHF+",
    "EURAUD":   "EURAUD+", "EURCAD":   "EURCAD+", "EURNZD":   "EURNZD+",
    "GBPJPY":   "GBPJPY+", "GBPCHF":   "GBPCHF+", "GBPAUD":   "GBPAUD+",
    "GBPCAD":   "GBPCAD+", "GBPNZD":   "GBPNZD+", "AUDJPY":   "AUDJPY+",
    "AUDCHF":   "AUDCHF+", "AUDCAD":   "AUDCAD+", "AUDNZD":   "AUDNZD+",
    "NZDJPY":   "NZDJPY+", "NZDCHF":   "NZDCHF+", "NZDCAD":   "NZDCAD+",
    "CADJPY":   "CADJPY+", "CADCHF":   "CADCHF+", "CHFJPY":   "CHFJPY+",
    "USDZAR":   "USDZAR+", "USDTRY":   "USDTRY+", "USDSGD":   "USDSGD+",
    "USDNOK":   "USDNOK+", "USDSEK":   "USDSEK+", "USDDKK":   "USDDKK+",
    "USDPLN":   "USDPLN+", "USDCNH":   "USDCNH+", "EURPLN":   "EURPLN+",
    "EURNOK":   "EURNOK+", "EURSEK":   "EURSEK+", "EURHUF":   "EURHUF+",
    "EURCZK":   "EURCZK+",
    "XAUEUR":   "XAUEUR+", "XAUAUD":   "XAUAUD+", "XAUJPY":   "XAUJPY+",
    "XPDUSD":   "XPDUSD",  "XPTUSD":   "XPTUSD",
    "CHINA50":  "CHINA50",
    "LTCUSD":   "LTCUSD",  "XRPUSD":   "XRPUSD",  "ADAUSD":   "ADAUSD",
    "BCHUSD":   "BCHUSD",  "DOTUSD":   "DOTUSD",  "LNKUSD":   "LNKUSD",
    "SOLUSD":   "SOLUSD",  "UKOUSD":   "UKOUSD",
    "Wheat-C":  "Wheat-C", "Cotton-C": "Cotton-C", "Soybean-C": "Soybean-C",
}

MIN_SL_PCT = {
    # Crypto (broker exige beaucoup)
    "BTCUSD":  0.0015, "LTCUSD":  0.025,
    "XRPUSD":  0.015,  "ADAUSD":  0.04,   "BCHUSD":  0.015,
    "DOTUSD":  0.05,   "LNKUSD":  0.02,   "SOLUSD":  0.02,
    # Metaux
    "XPDUSD":  0.015,  "XPTUSD":  0.015,
    "XAUJPY":  0.001,  "XAGUSD":  0.002,
    # Energy
    "CL-OIL":  0.003,  "UKOUSD":  0.002,  "GAS-C":   0.002,
    # Forex exotiques (besoin plus large)
    "USDZAR":  0.002,  "USDTRY":  0.005,  "USDMXN":  0.002,
    "USDNOK":  0.006,  "USDSEK":  0.002,  "USDDKK":  0.001,
    "USDPLN":  0.002,
    "EURHUF":  0.002,  "EURPLN":  0.002,
    "EURNOK":  0.006,  "EURSEK":  0.002,  "EURCZK":  0.002,
    # Indices europe/asia
    "GER40":   0.001,  "FRA40":   0.001,  "UK100":   0.001,
    "Nikkei225": 0.001,
    # Softs
    "Cotton-C": 0.005, "Wheat-C": 0.003,  "Soybean-C": 0.003,
    "Coffee-C": 0.003, "Cocoa-C": 0.003,  "Sugar-C":  0.003,
}

# Spread max accepte par actif (% du prix). Au-dela, setup ANNULE
# (le spread bouffe le RR -> trade impossible meme si setup ICT valide).
# Default 0.05% (5bp) pour majeurs. Plus large pour exotiques connus.
MAX_SPREAD_PCT_DEFAULT = float(os.getenv("MAX_SPREAD_PCT_DEFAULT", "0.05"))
MAX_SPREAD_PCT = {
    # Forex exotiques (spreads broker tres larges)
    "USDZAR":  0.15, "USDTRY":  0.30, "USDMXN":  0.10,
    "USDNOK":  0.10, "USDSEK":  0.10, "USDDKK":  0.05,
    "USDPLN":  0.10, "USDCNH":  0.05, "USDSGD":  0.03,
    "EURHUF":  0.15, "EURPLN":  0.10, "EURNOK":  0.10,
    "EURSEK":  0.10, "EURCZK":  0.15,
    # Metaux exotiques
    "XPDUSD":  0.40, "XPTUSD":  0.20, "XAGUSD":  0.10,
    "XAUEUR":  0.05, "XAUAUD":  0.05, "XAUJPY":  0.05,
    # Crypto alts
    "LTCUSD":  0.25, "XRPUSD":  0.30, "ADAUSD":  0.50,
    "BCHUSD":  0.30, "DOTUSD":  0.50, "LNKUSD":  0.30, "SOLUSD":  0.20,
    # Indices europe/asia
    "CHINA50": 0.10, "Nikkei225": 0.05,
    # Energy
    "UKOUSD":  0.05, "GAS-C": 0.10,
    # Softs
    "Cotton-C": 0.10, "Wheat-C": 0.10, "Soybean-C": 0.10,
    "Coffee-C": 0.15, "Cocoa-C": 0.15, "Sugar-C": 0.15,
}

ASSETS = list(BROKER_MAP.keys())
THRESHOLD = float(os.environ["BOT_THRESHOLD"])
RECENT_CUTOFF_MIN = int(os.environ["RECENT_CUTOFF_MIN"])
BOT_MAGIC = 21000  # V21
MAX_CONCURRENT_TRADES = 5

_seen_obs: set[tuple] = set()
_boot_ts: pd.Timestamp | None = None
_ref_cache: dict[str, pd.DataFrame] = {}  # cache des df_m15 ref par cycle
_spread_logger = SpreadLogger()  # log spreads pour stats futures
_cycle_spreads: dict[str, dict] = {}  # spread_pct + spread_points du tick courant par actif
_audit = TradeAuditLogger()  # log forensic V21 (chaque OB detecte + decision)

# Offset broker : Vantage RAW ECN = UTC+3 (GMT+3 MSK). Detecte au boot via tick BTCUSD 24/7.
BROKER_OFFSET_SEC: int = 0  # set par detect_broker_offset() au boot


def detect_broker_offset() -> int:
    """Detecte l'offset broker (sec) en comparant le timestamp d'un candle M1 frais a UTC.
    Utilise BTCUSD (24/7) en priorite, fallback XAUUSD. Retourne offset entier en heures *3600.
    """
    for probe_asset in ("BTCUSD", "XAUUSD", "EURUSD"):
        bsym = BROKER_MAP.get(probe_asset, probe_asset)
        rates = mt5.copy_rates_from_pos(bsym, mt5.TIMEFRAME_M1, 0, 1)
        if rates is None or len(rates) == 0:
            continue
        broker_ts = int(rates[0]["time"])  # unix seconds dans le TZ broker
        utc_now = datetime.now(timezone.utc).timestamp()
        # Le dernier candle M1 a forcement entre 0s et 120s
        # offset = broker_ts - utc_now (a la minute la plus proche)
        diff = broker_ts - utc_now
        # arrondi a l'heure entiere
        offset_hours = round(diff / 3600)
        offset_sec = offset_hours * 3600
        log.info(
            f"BROKER OFFSET detecte via {probe_asset} : "
            f"broker_ts={broker_ts} utc_now={utc_now:.0f} "
            f"diff={diff:+.0f}s = UTC{'+' if offset_hours>=0 else ''}{offset_hours}h"
        )
        return offset_sec
    log.warning("BROKER OFFSET : detection echouee, on assume 0 (UTC)")
    return 0


def broker_sym(asset: str) -> str:
    return BROKER_MAP.get(asset, asset)


def family_of(asset: str) -> str:
    """Regroupe les actifs par 'famille' pour le cooldown anti-correlation.
    Ex: EURUSD/EURJPY/EURGBP -> 'EUR' (toutes correlees a EUR).
    SP500/NAS100/DJ30 -> 'IDX_US'.
    XAUUSD/XAUEUR/XAUJPY -> 'XAU'.
    """
    a = asset.upper()
    # Metaux precieux
    if a.startswith("XAU"): return "XAU"
    if a.startswith("XAG"): return "XAG"
    if a.startswith("XPD") or a.startswith("XPT"): return "PGM"
    # Crypto
    if a.endswith("USD") and a[:3] in ("BTC", "ETH", "LTC", "XRP", "ADA", "BCH", "DOT", "LNK", "SOL"):
        return "CRYPTO"
    # Indices US
    if a in ("SP500", "NAS100", "DJ30", "US30"): return "IDX_US"
    if a in ("GER40", "FRA40", "UK100", "EU50"): return "IDX_EU"
    if a in ("Nikkei225", "HK50", "CHINA50"): return "IDX_ASIA"
    # Energie
    if a in ("CL-OIL", "UKOUSD", "GAS-C"): return "ENERGY"
    # Softs
    if a.endswith("-C"): return "SOFT"
    # Forex : on prend la 1ere devise comme famille (EUR pour EURUSD/EURJPY/EURGBP)
    if len(a) == 6 and a.isalpha():
        return a[:3]
    return a


def fetch_ohlcv(asset: str, tf, n: int = 500) -> pd.DataFrame | None:
    rates = mt5.copy_rates_from_pos(broker_sym(asset), tf, 0, n)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    # FIX OFFSET : MT5 retourne le timestamp en TZ broker (Vantage = UTC+3).
    # On retranche l'offset detecte au boot pour avoir des timestamps UTC vrais.
    df["time"] = pd.to_datetime(df["time"] - BROKER_OFFSET_SEC, unit="s", utc=True)
    df = df.rename(columns={"tick_volume": "volume"})
    return df.set_index("time").sort_index()


def refresh_ref_cache():
    """Refresh des 5 actifs reference M15 au debut de chaque cycle."""
    global _ref_cache
    _ref_cache = {}
    for ref in REF_ASSETS:
        df = fetch_ohlcv(ref, mt5.TIMEFRAME_M15, M15_LEN + 10)
        if df is None or len(df) < M15_LEN:
            log.warning(f"  ref {ref} : data insuffisante (M15)")
        _ref_cache[ref] = df


def process_asset(asset: str, predictor: V21Predictor,
                   pusher: DashboardPusher, mt5_exec: MT5Executor,
                   balance: float) -> dict:
    res = {"asset": asset, "n_obs": 0, "rejets": [], "setups": [],
           "trades_taken": 0, "errors": [], "candles": None,
           "candidates": []}  # setups acceptes ML (proba>=THRESHOLD), a trier par la main loop
    # Capture spread courant pour cet actif
    bsym = broker_sym(asset)
    info = mt5.symbol_info(bsym)
    tick = mt5.symbol_info_tick(bsym)
    spread_pct = None
    spread_points = None
    if info and tick and tick.bid > 0 and tick.ask > tick.bid:
        spread_abs = tick.ask - tick.bid
        mid = (tick.ask + tick.bid) / 2
        spread_pct = round(spread_abs / mid * 100, 4)
        spread_points = round(spread_abs / info.point if info.point > 0 else 0, 1)
        _cycle_spreads[asset] = {"spread_pct": spread_pct, "spread_points": spread_points}
        # Log pour stats futures
        try:
            _spread_logger.log_tick(datetime.now(timezone.utc), asset, tick.ask, tick.bid, info.point)
        except Exception:
            pass

    try:
        df_m1 = fetch_ohlcv(asset, mt5.TIMEFRAME_M1, 500)
        # Capture 60 dernieres M1 pour graphique dashboard (uniquement si setup pertinent)
        if df_m1 is not None and len(df_m1) >= 60:
            _tail = df_m1.tail(60)
            res["candles"] = [
                {"t": int(ts.timestamp()), "o": float(r["open"]), "h": float(r["high"]),
                 "l": float(r["low"]), "c": float(r["close"])}
                for ts, r in _tail.iterrows()
            ]
        df_m15 = fetch_ohlcv(asset, mt5.TIMEFRAME_M15, 200)
        df_h1 = fetch_ohlcv(asset, mt5.TIMEFRAME_H1, 200)
        df_d1 = fetch_ohlcv(asset, mt5.TIMEFRAME_D1, 60)
        if df_m1 is None or len(df_m1) < 100:
            res["errors"].append("M1 insuffisant")
            return res

        swing_strength = get_param(asset, "swing_strength_m1", 2)
        obs = detect_order_blocks(df_m1, swing_strength=swing_strength)
        last_bar = df_m1.index[-1]
        cutoff_age = last_bar - pd.Timedelta(minutes=RECENT_CUTOFF_MIN)
        cutoff = cutoff_age
        if _boot_ts is not None and _boot_ts > cutoff_age:
            cutoff = _boot_ts
        obs_recent = [
            ob for ob in obs
            if ob.validation_ts is not None
            and ob.validation_ts >= cutoff
            and ob.validation_ts <= last_bar
        ]
        res["n_obs"] = len(obs_recent)
        if not obs_recent:
            return res

        # Now reference pour mesurer retards
        now_utc = pd.Timestamp.utcnow()
        if now_utc.tz is None:
            now_utc = now_utc.tz_localize("UTC")

        for ob in obs_recent:
            ob_key = (asset, ob.validation_ts.isoformat(), ob.direction)
            if ob_key in _seen_obs:
                continue

            # Audit timing
            ob_age_s = (now_utc - ob.validation_ts).total_seconds() if hasattr(ob.validation_ts, 'tz_localize') else 0
            ob_top = getattr(ob, "ob_high", None) or getattr(ob, "top", None)
            ob_bottom = getattr(ob, "ob_low", None) or getattr(ob, "bottom", None)

            try:
                r = evaluate_ob(
                    ob, df_m1, df_m15, df_d1,
                    instrument=asset, df_htf2=df_h1, htf2_name="H1",
                )
                if r is None or r.verdict != "TRADE" or r.trade_setup is None:
                    reason = r.rejection_reason if r else "no_result"
                    # Audit log : OB rejete par pipeline ICT
                    _audit.log(
                        asset=asset, direction=ob.direction,
                        ob_validation_ts=str(ob.validation_ts),
                        ob_age_s=ob_age_s,
                        ob_top=ob_top, ob_bottom=ob_bottom,
                        decision="ICT_REJECT", ict_reason=str(reason)[:200],
                    )
                    res["rejets"].append({"asset": asset, "ts": ob.validation_ts.isoformat(),
                                           "direction": ob.direction, "reason": reason or "no_trade"})
                    # REGLE STRICTE : 1 OB = 1 evaluation max. Si ICT_REJECT, on oublie cet OB
                    # a vie. Sinon le pipeline ICT re-evaluerait avec des bougies plus
                    # recentes -> trade en retard quand les conditions changent.
                    _seen_obs.add(ob_key)
                    continue
                # V21 predict (decideur, cross-asset)
                try:
                    proba = predictor.predict_one(
                        ts=ob.validation_ts,
                        df_m1=df_m1, df_m15=df_m15,
                        df_h1=df_h1, df_d1=df_d1,
                        ref_m15_dict=_ref_cache,
                    )
                except Exception as e:
                    res["errors"].append(f"V21 predict {asset}: {e}")
                    proba = None
                if proba is None:
                    _seen_obs.add(ob_key)
                    res["rejets"].append({"asset": asset, "ts": ob.validation_ts.isoformat(),
                                           "direction": ob.direction, "reason": "V21_unavailable"})
                    continue

                # V20 predict SHADOW (display only, comparaison)
                proba_v20 = None
                v20 = V20Predictor.get_instance()
                if v20 is not None:
                    try:
                        from bot_v2 import ml_filter
                        feats = ml_filter._features_from_result(
                            r, ob, asset,
                            df_ltf=df_m1, df_d1=df_d1, mss_setups=None, df_htf=df_m15,
                        )
                        proba_v20 = v20.predict_one(feats, asset, df_m1, df_m15, df_h1, ob.validation_ts)
                    except Exception as e:
                        log.warning(f"  V20 shadow predict {asset} fail : {type(e).__name__}: {str(e)[:120]}")
                        proba_v20 = None
                else:
                    if not hasattr(log, "_v20_warned"):
                        log.warning("V20Predictor.get_instance() = None (shadow disabled)")
                        log._v20_warned = True

                _seen_obs.add(ob_key)
                kz = killzone_at(ob.validation_ts) or "None"
                setup = r.trade_setup

                # Market price actuel pour calcul slippage entry vs OB
                cur_tick = mt5.symbol_info_tick(broker_sym(asset))
                cur_price = cur_tick.ask if cur_tick and ob.direction == "bullish" else (cur_tick.bid if cur_tick else None)
                sp_info = _cycle_spreads.get(asset, {})

                # Audit log : decision V21
                _audit.log(
                    asset=asset, direction=ob.direction,
                    ob_validation_ts=str(ob.validation_ts),
                    ob_age_s=ob_age_s,
                    ob_top=ob_top, ob_bottom=ob_bottom,
                    setup_entry=float(setup.entry_price), setup_sl=float(setup.stop_loss),
                    setup_tp=float(setup.take_profit), setup_rr=float(setup.rr),
                    market_price=cur_price,
                    spread_pct=sp_info.get("spread_pct"),
                    spread_points=sp_info.get("spread_points"),
                    score=r.score or 0, killzone=kz,
                    proba_v21=float(proba),
                    proba_v20=float(proba_v20) if proba_v20 is not None else None,
                    threshold=THRESHOLD,
                    decision="V21_TRADE" if proba >= THRESHOLD else "V21_REJECT",
                )

                if proba >= THRESHOLD:
                    # FIX RAFALES : on ne trade plus directement, on collecte le candidat.
                    # La boucle main triera par proba et appliquera top-N + cooldown devise.
                    res["candidates"].append({
                        "asset": asset, "direction": ob.direction,
                        "setup": setup, "ob": ob, "proba": proba,
                        "proba_v20": proba_v20, "killzone": kz,
                        "score": r.score or 0,
                        "validation_ts": ob.validation_ts,
                    })
                else:
                    res["rejets"].append({
                        "asset": asset, "ts": ob.validation_ts.isoformat(),
                        "direction": ob.direction,
                        "reason": f"ml_below_thr_{proba:.3f}",
                        "proba": proba, "proba_v20": proba_v20,
                        "threshold": THRESHOLD,
                        "entry": setup.entry_price,
                        "sl": setup.stop_loss, "tp": setup.take_profit, "rr": setup.rr,
                    })
            except Exception as e:
                res["errors"].append(f"OB process fail {asset}: {type(e).__name__}: {str(e)[:100]}")
    except Exception as e:
        res["errors"].append(f"asset {asset} fail: {type(e).__name__}: {str(e)[:100]}")
    return res


def execute_trade(mt5_exec: MT5Executor, asset: str, setup, proba: float,
                   balance: float, ob, killzone: str,
                   pusher: DashboardPusher) -> dict:
    """Tente d'envoyer un ordre MT5.

    Returns dict avec status :
      - "OK"               : trade execute
      - "SPREAD_TOO_HIGH"  : spread broker trop large -> setup annule
      - "SL_TOO_TIGHT"     : SL ICT < min broker -> setup annule
      - "MARGIN"           : marge insuffisante -> setup annule
      - "MT5_REJECT"       : ordre rejete par MT5 -> setup annule
      - "FAIL"             : erreur interne
    + 'detail' optionnel (str humain) + 'spread_pct' / 'min_required' selon cas.
    """
    try:
        direction = setup.direction
        sl = float(setup.stop_loss)
        tp = float(setup.take_profit)
        entry = float(setup.entry_price)
        rr = float(setup.rr)
        risk_pct = float(os.getenv("RISK_PCT", "0.05"))
        risk_eur = balance * risk_pct
        bsym = broker_sym(asset)
        info = mt5.symbol_info(bsym)
        if info is None:
            log.error(f"{asset} ({bsym}) : symbol_info None")
            return {"status": "FAIL", "detail": "symbol_info None"}
        tick_value = getattr(info, "trade_tick_value", 1.0)
        tick_size = getattr(info, "trade_tick_size", 0.01)
        tick = mt5.symbol_info_tick(bsym)
        if tick is None:
            return {"status": "FAIL", "detail": "tick None"}
        market_price = tick.ask if direction == "bullish" else tick.bid

        # FIX SL vs SPREAD : MT5 ferme un BUY au bid (bid touche SL = SL hit) et
        # un SELL au ask (ask touche SL = SL hit). Le SL ICT est sur le prix de l'OB.
        # Pour eviter d'etre stoppe avant que le prix touche reellement l'OB en bid/bull
        # ou en ask/bear, on retire (BUY) / ajoute (SELL) le spread courant au SL.
        spread_abs = tick.ask - tick.bid
        sl_original = sl
        if direction == "bullish":
            sl = sl - spread_abs  # SL plus bas : tolerance le spread
        else:
            sl = sl + spread_abs  # SL plus haut : tolerance le spread
        sl_distance = abs(market_price - sl)
        if sl_distance <= 0:
            return {"status": "FAIL", "detail": "sl_distance <= 0"}

        # SPREAD CHECK retire : trop de bons setups bloques. MT5 decidera si l'ordre passe.
        mid = (tick.ask + tick.bid) / 2 if tick.bid > 0 else market_price
        spread_pct = (spread_abs / mid * 100) if mid > 0 else 0

        # SL/TP ICT STRICT : NE JAMAIS modifier le SL/TP de l'OB.
        # Si le SL est sous le minimum broker -> SKIP le trade (pas elargir).
        stops_level = getattr(info, "trade_stops_level", 0) * info.point
        min_pct = MIN_SL_PCT.get(asset)
        if min_pct:
            min_required = max(stops_level, market_price * min_pct)
        else:
            min_required = stops_level
        if min_required > 0 and sl_distance < min_required:
            log.info(f"{asset} : SL ICT trop serre ({sl_distance:.5f} < min broker {min_required:.5f}) -> SKIP")
            return {
                "status": "SL_TOO_TIGHT",
                "detail": f"sl_distance={sl_distance:.5f} < min_broker={min_required:.5f}",
                "sl_distance": float(sl_distance),
                "min_required": float(min_required),
            }
        n_ticks = sl_distance / tick_size
        risk_per_lot = n_ticks * tick_value
        if risk_per_lot <= 0:
            return {"status": "FAIL", "detail": "risk_per_lot <= 0"}
        lots = risk_eur / risk_per_lot
        lots = max(info.volume_min, round(lots / info.volume_step) * info.volume_step)
        lots = min(lots, info.volume_max)

        # MARGIN CHECK retire : trop de bons setups bloques quand le compte est charge.
        # On laisse MT5 decider, il rejettera l'ordre si vraiment plus de marge (-> MT5_REJECT).
        comment = f"V21-{direction[0].upper()} ml={proba:.2f}"
        # Capture market price avant envoi pour calcul slippage
        pre_tick = mt5.symbol_info_tick(bsym)
        pre_price = pre_tick.ask if direction == "bullish" else (pre_tick.bid if pre_tick else market_price)

        # Ordre MARKET (entry immediate au prix courant).
        # Le fix critique '1 OB = 1 evaluation' garantit que V21 dit oui DES la validation
        # de l'OB, donc le price est encore dans (ou tres proche de) la zone OB.
        result = mt5_exec.place_market_order(
            symbol=bsym, direction=direction, volume=lots,
            sl=sl, tp=tp, comment=comment, magic=BOT_MAGIC,
        )
        if result is None:
            log.warning(f"{asset} : ordre rejete par MT5")
            _audit.log(
                asset=asset, direction=direction,
                ob_setup_entry=float(entry), ob_setup_sl=float(sl), ob_setup_tp=float(tp),
                lots=float(lots), market_price=float(pre_price),
                proba_v21=float(proba),
                decision="MT5_REJECT", mt5_error="place_market_order returned None",
            )
            return {"status": "MT5_REJECT", "detail": "place_market_order returned None"}
        slippage = result["price"] - pre_price
        log.info(
            f"TRADE OK {asset} {direction} entry={result['price']:.5f} "
            f"SL={sl:.5f} (OB={sl_original:.5f} +spread {spread_abs:.5f}) "
            f"TP={tp:.5f} vol={result['volume']:.2f} "
            f"rr={rr:.2f} ml={proba:.3f} slippage={slippage:+.5f}"
        )
        _audit.log(
            asset=asset, direction=direction, ticket=int(result["ticket"]),
            ob_setup_entry=float(entry),
            ob_setup_sl=float(sl_original),  # SL ICT vrai (top/bottom OB)
            sl_adjusted=float(sl),  # SL envoye a MT5 (= SL ICT +/- spread)
            spread_at_send=float(spread_abs),
            ob_setup_tp=float(tp),
            actual_entry=float(result["price"]),
            pre_market_price=float(pre_price),
            slippage_abs=float(slippage),
            slippage_pct=float(slippage / pre_price * 100) if pre_price else 0,
            lots=float(result["volume"]),
            proba_v21=float(proba),
            rr=float(rr), score=0,
            decision="MT5_OK",
        )
        pusher.push_trade_executed(
            instrument=asset, ticket=result["ticket"],
            direction=direction, entry=result["price"],
            sl=sl, tp=tp, volume=result["volume"],
            rr=rr, ml_proba=proba, score=0, killzone=killzone,
        )
        return {"status": "OK", "ticket": int(result["ticket"]),
                "entry": float(result["price"]), "volume": float(result["volume"])}
    except Exception as e:
        log.exception(f"execute_trade fail {asset}: {e}")
        return {"status": "FAIL", "detail": f"{type(e).__name__}: {str(e)[:120]}"}


def main():
    log.info("=" * 60)
    log.info("BOT V21 MONSTER LIVE - Deep Transformer + Cross-Asset")
    log.info("=" * 60)
    log.info(f"THRESHOLD = {THRESHOLD}")
    log.info(f"RECENT_CUTOFF_MIN = {RECENT_CUTOFF_MIN}")
    log.info(f"ASSETS = {len(ASSETS)} actifs")
    log.info(f"REF_ASSETS = {REF_ASSETS}")

    log.info("Chargement V21 MONSTER model...")
    predictor = V21Predictor.get_instance()
    if predictor is None:
        log.error("V21 model INDISPONIBLE - bot ne demarre pas")
        return
    log.info(f"V21 OK : device={predictor.device}, d_model={predictor.d_model}")

    mt5_exec = MT5Executor()
    if not mt5_exec.initialize():
        log.error(f"MT5 init failed : {mt5.last_error()}")
        return
    log.info(f"MT5 connecte : login={mt5_exec.account_info.login} server={mt5_exec.account_info.server}")
    log.info(f"  Balance : {mt5_exec.account_info.balance} {mt5_exec.account_info.currency}")

    activated, skipped = [], []
    for a in ASSETS:
        bsym = broker_sym(a)
        if mt5.symbol_select(bsym, True):
            activated.append(bsym)
        else:
            skipped.append((a, bsym))
    # Active aussi les REF assets
    for ref in REF_ASSETS:
        mt5.symbol_select(broker_sym(ref), True)
    log.info(f"  Symbols actives : {len(activated)}/{len(ASSETS)}")
    if skipped:
        log.warning(f"  Symbols KO : {skipped}")

    session_id = uuid.uuid4().hex[:8]
    pusher = DashboardPusher(
        url=os.getenv("DASHBOARD_URL"),
        session_id=session_id,
    )
    log.info(f"DashboardPusher : url={pusher.url}, session={session_id}")
    pusher.push_start(f"Bot V21 MONSTER demarre (threshold={THRESHOLD})")

    # Detecter offset broker AVANT tout fetch (sinon timestamps fausses)
    global BROKER_OFFSET_SEC
    BROKER_OFFSET_SEC = detect_broker_offset()
    log.info(f"BROKER_OFFSET_SEC = {BROKER_OFFSET_SEC} ({BROKER_OFFSET_SEC/3600:+.1f}h)")

    global _boot_ts, _seen_obs
    _boot_ts = pd.Timestamp.now(tz="UTC")
    log.info(f"BOOT_TS = {_boot_ts} (UTC vrai)")

    log.info("=" * 60)
    log.info("DEMARRAGE BOUCLE PRINCIPALE - 1 cycle/min")
    log.info("=" * 60)

    cycle_n = 0
    try:
        while True:
            cycle_n += 1
            t0 = time.time()
            now = datetime.now(timezone.utc)
            positions = mt5.positions_get() or []
            n_total_open = len(positions)
            account = mt5.account_info()
            balance = account.balance if account else 0

            log.info(f"--- CYCLE {cycle_n} | T={now.strftime('%H:%M:%S')} UTC | balance={balance:.2f} | open={n_total_open} ---")

            # Refresh cache des 5 refs
            refresh_ref_cache()

            all_rejets = []
            all_setups = []
            all_candidates = []
            total_obs = 0
            total_trades = 0
            candles_by_asset = {}
            for asset in ASSETS:
                res = process_asset(asset, predictor, pusher, mt5_exec, balance)
                total_obs += res["n_obs"]
                all_rejets.extend(res["rejets"])
                all_setups.extend(res["setups"])
                # Capture candles seulement si pertinent pour dashboard
                if (res["rejets"] or res["setups"] or res["candidates"]) and res.get("candles"):
                    candles_by_asset[asset] = res["candles"]
                # Candidats ML accepte (proba >= 0.80) : trade immediat sans tri ni cooldown
                for c in res["candidates"]:
                    c["candles"] = res.get("candles")
                    all_candidates.append(c)
                if res["errors"]:
                    for e in res["errors"]:
                        log.warning(f"  {asset} : {e}")

            # Chaque candidat valide ML = trade immediat (pas de limite cycle, pas de cooldown).
            # La protection contre les pertes vient du seuil V21 >= 0.80 (WR 87% attendu).
            for c in all_candidates:
                exec_result = execute_trade(mt5_exec, c["asset"], c["setup"], c["proba"], balance, c["ob"], c["killzone"], pusher)
                status = exec_result.get("status", "FAIL")
                if status == "OK":
                    pusher.push_setup(
                        instrument=c["asset"], ts=c["validation_ts"],
                        direction=c["direction"], entry_price=c["setup"].entry_price,
                        sl=c["setup"].stop_loss, tp=c["setup"].take_profit,
                        rr=c["setup"].rr, score=c["score"],
                        ml_proba=c["proba"], killzone=c["killzone"],
                        candles=c.get("candles"),
                    )
                    total_trades += 1
                    all_setups.append({"asset": c["asset"], "proba": c["proba"], "proba_v20": c["proba_v20"]})
                else:
                    # SETUP_CANCELLED -> push dashboard + audit
                    try:
                        pusher.push_setup_cancelled(
                            instrument=c["asset"], ts=c["validation_ts"],
                            direction=c["direction"], entry_price=c["setup"].entry_price,
                            sl=c["setup"].stop_loss, tp=c["setup"].take_profit,
                            rr=c["setup"].rr, score=c["score"], ml_proba=c["proba"],
                            cancel_reason=status, cancel_detail=exec_result.get("detail"),
                            spread_pct=exec_result.get("spread_pct"),
                            max_spread_pct=exec_result.get("max_spread_pct"),
                            killzone=c["killzone"], candles=c.get("candles"),
                        )
                    except Exception as e:
                        log.warning(f"push_setup_cancelled fail {c['asset']}: {e}")
                    _audit.log(
                        asset=c["asset"], direction=c["direction"],
                        ob_validation_ts=str(c["validation_ts"]),
                        setup_entry=float(c["setup"].entry_price),
                        setup_sl=float(c["setup"].stop_loss),
                        setup_tp=float(c["setup"].take_profit),
                        proba_v21=float(c["proba"]),
                        decision="SETUP_CANCELLED",
                        cancel_reason=status,
                        cancel_detail=exec_result.get("detail"),
                        spread_pct=exec_result.get("spread_pct"),
                    )
                    all_setups.append({"asset": c["asset"], "proba": c["proba"], "cancelled": status})

            elapsed = time.time() - t0
            log.info(f"  Total : {total_obs} OBs | {len(all_candidates)} candidats | {total_trades} trades passes | {len(all_rejets)} rejets ML (cycle {elapsed:.1f}s)")

            pusher.push_cycle(
                actifs_scanned=len(ASSETS),
                total_s=elapsed,
                fetch_s=0, compute_s=elapsed,
                latencies={a: 0 for a in ASSETS},
            )
            pusher.push_stats(
                total=cycle_n, wins=0, losses=0, wr_pct=0,
                pnl_total=0, balance=balance, equity=account.equity if account else 0,
                positions_open=n_total_open, pending_orders=0,
            )
            # Sync positions reelles MT5 -> dashboard marquera CANCELLED les pending fantomes
            open_pos = [{"ticket": p.ticket, "pnl": p.profit} for p in positions]
            pending_orders = mt5.orders_get() or []
            pending_tickets = [o.ticket for o in pending_orders]
            pusher.push_positions_sync(open_positions=open_pos, pending_tickets=pending_tickets)

            if all_rejets:
                # Convertir 'asset' -> 'instrument' pour compat dashboard
                rejets_fmt = []
                for r in all_rejets[:50]:
                    sp = _cycle_spreads.get(r.get("asset", "?"), {})
                    rejets_fmt.append({
                        "instrument": r.get("asset", "?"),
                        "ts": r.get("ts", ""),
                        "direction": r.get("direction", "?"),
                        "reason": r.get("reason", "?"),
                        "ml_proba": r.get("proba"),
                        "ml_proba_v20": r.get("proba_v20"),  # shadow
                        "threshold": r.get("threshold"),
                        "entry": r.get("entry"),
                        "sl": r.get("sl"),
                        "tp": r.get("tp"),
                        "rr": r.get("rr"),
                        "spread_pct": sp.get("spread_pct"),
                        "spread_points": sp.get("spread_points"),
                    })
                pusher.push_rejected_batch(rejets_fmt,
                                            candles_by_asset=candles_by_asset if candles_by_asset else None)
                log.info(f"  Pushed {len(rejets_fmt)} rejets ({len(candles_by_asset)} charts) au dashboard")

            # PAS de cleanup _seen_obs : un OB evalue une seule fois, point.
            # Si on cleanup, V21 peut dire NON puis OUI sur le meme OB plus tard
            # (apres que le price ait deja bouge) -> trade en retard, RR cassé.
            # Cap memoire : garde les 5000 dernieres entrees (au-dela, on supprime
            # les plus vieilles via FIFO).
            if cycle_n % 240 == 0 and len(_seen_obs) > 10000:
                log.info(f"  _seen_obs cap : {len(_seen_obs)} entries, garde les 5000 plus recentes")
                _seen_obs = set(list(_seen_obs)[-5000:])

            # Flush spreads + audit toutes les 10 cycles (~2.5 min en cycle 15s)
            if cycle_n % 10 == 0:
                try:
                    _spread_logger.flush()
                    _audit.flush()
                except Exception as e:
                    log.warning(f"loggers flush fail : {e}")

            # Sleep jusqu'a la prochaine minute boundary
            # Cycle plus rapide : 15s au lieu de 60s (reduit le retard sur OB validation)
            CYCLE_SEC = int(os.getenv("CYCLE_SEC", "15"))
            elapsed_cycle = time.time() - t0
            sleep_s = max(0.1, CYCLE_SEC - elapsed_cycle)
            time.sleep(sleep_s)
    except KeyboardInterrupt:
        log.info("Interruption clavier")
    except Exception as e:
        log.exception(f"Crash : {e}")
        pusher.push_error(f"Bot V21 crash: {e}")
    finally:
        pusher.shutdown()
        mt5.shutdown()


if __name__ == "__main__":
    main()
