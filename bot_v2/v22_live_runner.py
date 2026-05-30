"""V22 LIVE RUNNER - Bot ICT structurel multi-actifs sur demo Vantage.

Implementation 1:1 de la strategie V22 validee par 5 tests critiques.
Voir V22_COMPLET.md pour la doc complete.

Architecture :
- 22 actifs avec leur config dediee (PER_ASSET_CONFIG)
- Regle ICT D1 + H1 + PD daily + Option A horaire
- Scoring tier S/A/B/C/D avec sizing conservateur
- 10 garde-fous live (CB -20%, plafond 100 lots, news blackout, etc.)
- 1 cycle = 1 scan / minute
- MT5 demo Vantage + DashboardPusher (Railway) + Telegram

Usage : python -m bot_v2.v22_live_runner

Env vars (optionnel, ont des defauts) :
- DASHBOARD_URL    : default Railway
- VANTAGE_LOGIN    : 25420721 (demo)
- VANTAGE_PASSWORD : (defini hors Git)
- VANTAGE_SERVER   : VantageInternational-Demo
- LIQUIDATION_PCT  : default 0.25 (stop si capital < 25% initial)
- DAILY_STOP_PCT   : default 0.20 (circuit breaker journalier)
- LOT_MAX          : default 100
- MAX_CONCURRENT   : default 5
- INITIAL_BALANCE  : capital "de reference" pour liquidation (sinon = balance demarrage)
"""
from __future__ import annotations

import os
import sys
import time
import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
import MetaTrader5 as mt5

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Defauts env
os.environ.setdefault("DASHBOARD_URL", "https://tradingbote-production.up.railway.app/api/ingest")
os.environ.setdefault("DAILY_STOP_PCT", "0.20")
os.environ.setdefault("LIQUIDATION_PCT", "0.25")
os.environ.setdefault("LOT_MAX", "100")
os.environ.setdefault("MAX_CONCURRENT", "5")

# ============ Logging ============
log = logging.getLogger("v22_live")
log.setLevel(logging.INFO)
_fh = logging.FileHandler(ROOT / "v22_live.log", mode="a", encoding="utf-8")
_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
log.addHandler(_fh)
_sh = logging.StreamHandler()
_sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
log.addHandler(_sh)

# ============ Imports bot_v2 (apres env) ============
from bot_v2.concepts.safe import (
    find_obs_simple, daily_bias_safe, htf_value_at_t, atr_safe,
)
from bot_v2.mt5_executor import MT5Executor
from bot_v2.push_dashboard import DashboardPusher


# =====================================================================
# CONFIGURATION V22 - 22 actifs valides + leurs configs dediees
# =====================================================================

# Mapping nom V22 -> nom MT5 broker (avec suffixe + pour RAW ECN)
BROKER_MAP = {
    "AUDUSD":    "AUDUSD+",
    "BTCUSD":    "BTCUSD",
    "BVSPX":     "BVSPX",
    "CL-OIL":    "USOUSD",   # CL-OIL chez nous = USOUSD chez Vantage
    "Cocoa-C":   "Cocoa-C",
    "DJ30":      "DJ30",
    "ETHUSD":    "ETHUSD",
    "FRA40":     "FRA40",
    "GAS-C":     "NG-C",
    "GBPUSD":    "GBPUSD+",
    "GER40":     "GER40",
    "HK50":      "HK50",
    "NAS100":    "NAS100",
    "Nikkei225": "Nikkei225",
    "SP500":     "SP500",
    "UK100":     "UK100",
    "USDCAD":    "USDCAD+",
    "USDCHF":    "USDCHF+",
    "USDJPY":    "USDJPY+",
    "USDMXN":    "USDMXN+",
    "USDZAR":    "USDZAR+",
    "XAUUSD":    "XAUUSD+",
}

# Config dediee par actif (TOP 1 du grid)
PER_ASSET_CONFIG = {
    "AUDUSD":    {"h1m":0.10, "atr_max":2.0,  "h_min":9, "h_max":22, "disp":0.3, "sl_buf":0.20, "cons":2},
    "BTCUSD":    {"h1m":1.00, "atr_max":2.5,  "h_min":8, "h_max":20, "disp":0.0, "sl_buf":0.00, "cons":2},
    "BVSPX":     {"h1m":0.00, "atr_max":None, "h_min":9, "h_max":22, "disp":0.5, "sl_buf":0.15, "cons":2},
    "CL-OIL":    {"h1m":0.50, "atr_max":2.5,  "h_min":8, "h_max":19, "disp":0.3, "sl_buf":0.20, "cons":2},
    "Cocoa-C":   {"h1m":0.70, "atr_max":None, "h_min":0, "h_max":17, "disp":0.3, "sl_buf":0.15, "cons":2},
    "DJ30":      {"h1m":0.30, "atr_max":2.5,  "h_min":9, "h_max":22, "disp":0.0, "sl_buf":0.20, "cons":2},
    "ETHUSD":    {"h1m":1.00, "atr_max":2.0,  "h_min":9, "h_max":22, "disp":0.3, "sl_buf":0.15, "cons":2},
    "FRA40":     {"h1m":0.30, "atr_max":2.5,  "h_min":8, "h_max":22, "disp":0.0, "sl_buf":0.20, "cons":2},
    "GAS-C":     {"h1m":0.70, "atr_max":2.5,  "h_min":9, "h_max":19, "disp":0.0, "sl_buf":0.15, "cons":2},
    "GBPUSD":    {"h1m":0.20, "atr_max":3.0,  "h_min":6, "h_max":21, "disp":0.0, "sl_buf":0.20, "cons":2},
    "GER40":     {"h1m":0.40, "atr_max":None, "h_min":7, "h_max":22, "disp":0.0, "sl_buf":0.15, "cons":2},
    "HK50":      {"h1m":0.40, "atr_max":None, "h_min":0, "h_max":19, "disp":0.3, "sl_buf":0.20, "cons":2},
    "NAS100":    {"h1m":0.30, "atr_max":3.0,  "h_min":9, "h_max":21, "disp":0.3, "sl_buf":0.05, "cons":2},
    "Nikkei225": {"h1m":0.30, "atr_max":2.0,  "h_min":0, "h_max":21, "disp":0.5, "sl_buf":0.10, "cons":2},
    "SP500":     {"h1m":0.20, "atr_max":None, "h_min":9, "h_max":24, "disp":0.3, "sl_buf":0.00, "cons":2},
    "UK100":     {"h1m":0.10, "atr_max":2.5,  "h_min":7, "h_max":24, "disp":0.5, "sl_buf":0.20, "cons":2},
    "USDCAD":    {"h1m":0.10, "atr_max":None, "h_min":9, "h_max":17, "disp":0.0, "sl_buf":0.20, "cons":2},
    "USDCHF":    {"h1m":0.20, "atr_max":2.0,  "h_min":6, "h_max":22, "disp":0.0, "sl_buf":0.20, "cons":2},
    "USDJPY":    {"h1m":0.20, "atr_max":2.5,  "h_min":7, "h_max":22, "disp":0.0, "sl_buf":0.00, "cons":2},
    "USDMXN":    {"h1m":0.30, "atr_max":None, "h_min":6, "h_max":20, "disp":0.0, "sl_buf":0.20, "cons":2},
    "USDZAR":    {"h1m":0.30, "atr_max":3.0,  "h_min":7, "h_max":21, "disp":0.0, "sl_buf":0.00, "cons":2},
    "XAUUSD":    {"h1m":0.40, "atr_max":3.0,  "h_min":6, "h_max":21, "disp":0.0, "sl_buf":0.00, "cons":2},
}

ASSETS = list(PER_ASSET_CONFIG.keys())

# Sizing CONSERVATEUR (NE PAS toucher sans re-validation)
TIER_RISK = {"S": 0.020, "A": 0.015, "B": 0.010, "C": 0.005, "D": 0.0}

# Plan trade
TP_RR = 2.0
MAX_FILL_BARS = 20    # 20 bougies M5 = 1h40 max pour fill
MAX_HOLD_BARS = 50    # 50 bougies M5 = 4h max en position

# Garde-fous globaux
BOT_MAGIC = 22260530   # magic V22
DAILY_STOP_PCT = float(os.environ["DAILY_STOP_PCT"])
LIQUIDATION_PCT = float(os.environ["LIQUIDATION_PCT"])
LOT_MAX = float(os.environ["LOT_MAX"])
MAX_CONCURRENT = int(os.environ["MAX_CONCURRENT"])

# News blackout (UTC, hardcoded - a maintenir)
# Format : (year, month, day, hour, minute) UTC
NEWS_BLACKOUT_UTC = [
    # NFP (1er vendredi du mois 12h30 UTC) - juin a sept 2026 (a etendre)
    (2026, 6, 5, 12, 30),   (2026, 7, 3, 12, 30),
    (2026, 8, 7, 12, 30),   (2026, 9, 4, 12, 30),
    # FOMC (a maintenir manuellement, 18h UTC)
    (2026, 6, 17, 18, 0),   (2026, 7, 29, 18, 0),
    (2026, 9, 16, 18, 0),   (2026, 11, 4, 18, 0),
    # CPI US (mid-month 12h30 UTC)
    (2026, 6, 11, 12, 30),  (2026, 7, 15, 12, 30),
    (2026, 8, 12, 12, 30),  (2026, 9, 11, 12, 30),
]
NEWS_BLACKOUT_MINUTES = 15   # +- minutes autour de chaque news


# =====================================================================
# State global du bot
# =====================================================================

_seen_obs: set[tuple] = set()                      # OBs deja traites (anti re-trade)
_boot_ts: pd.Timestamp | None = None               # anti-rafale au demarrage
_daily_open_balance: dict[pd.Timestamp, float] = {}  # snapshot balance debut de jour
_daily_blocked: set[pd.Timestamp] = set()          # jours ou CB s'est declenche
_initial_balance: float | None = None              # capital "de reference" pour liquidation
_cycle_count = 0
_liquidated = False
# Paires partial+runner (Vantage pas de fermeture partielle):
# {partial_ticket: {asset, direction, entry, sl, tp_1r, runner_ticket, be_moved}}
_open_pairs: dict[int, dict] = {}
# Trades ouverts tracks par ticket (pour detect close + push TRADE_CLOSED) :
# {ticket: {asset, direction, entry, sl, tp, volume, opened_ts, opened_balance, magic}}
_open_trades: dict[int, dict] = {}
# Stats live cumulees
_live_stats = {"total": 0, "wins": 0, "losses": 0, "pnl_total": 0.0}
# Latencies par actif pour le dashboard (ms)
_last_latencies: dict[str, int] = {}


# =====================================================================
# Fetch & helpers
# =====================================================================

def broker_sym(asset: str) -> str:
    return BROKER_MAP.get(asset, asset)


def fetch_ohlcv(asset: str, tf, n: int = 500) -> pd.DataFrame | None:
    rates = mt5.copy_rates_from_pos(broker_sym(asset), tf, 0, n)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.rename(columns={"tick_volume": "volume"})
    return df.set_index("time").sort_index()[["open", "high", "low", "close"]]


def is_news_blackout(now_utc: pd.Timestamp) -> bool:
    """True si on est dans une fenetre news ±15min."""
    for (y, mo, d, h, mn) in NEWS_BLACKOUT_UTC:
        news_ts = pd.Timestamp(year=y, month=mo, day=d, hour=h, minute=mn, tz="UTC")
        delta_min = abs((now_utc - news_ts).total_seconds()) / 60
        if delta_min <= NEWS_BLACKOUT_MINUTES:
            return True
    return False


# =====================================================================
# Filtre USER V22 (D1 + H1 + PD daily)
# =====================================================================

def passes_user_rule(df_m5, df_h1, df_d1, ob) -> tuple[bool, dict]:
    """Renvoie (passes, details_dict) pour audit/log."""
    val_ts = ob.validation_ts
    val_idx = ob.validation_index
    out = {"d1_a": False, "h1_a": False, "pd_a": False, "h1_mom": 0.0,
           "hour_fr": 0.0, "atr_ratio": 1.0, "disp": 0.0}

    # D1 bias
    d1 = daily_bias_safe(df_d1, val_ts)
    if not d1["ok"]:
        return False, {**out, "reason": "no_d1"}
    d1_h = d1["bias"] == "haussier"
    out["d1_a"] = (ob.direction == "bullish" and d1_h) or (ob.direction == "bearish" and not d1_h)
    if not out["d1_a"]:
        return False, {**out, "reason": "d1_misaligned"}

    # H1 trend
    h1_close = htf_value_at_t(df_h1, val_ts, "close")
    if h1_close is None:
        return False, {**out, "reason": "no_h1"}
    pos = df_h1.index.searchsorted(val_ts, side="left") - 1 - 10
    if pos < 0:
        return False, {**out, "reason": "h1_too_recent"}
    h1_old = float(df_h1["close"].iloc[pos])
    if h1_old > 0:
        out["h1_mom"] = abs((h1_close - h1_old) / h1_old * 100)
    h1_h = h1_close > h1_old
    out["h1_a"] = (ob.direction == "bullish" and h1_h) or (ob.direction == "bearish" and not h1_h)
    if not out["h1_a"]:
        return False, {**out, "reason": "h1_misaligned"}

    # PD daily (J-1)
    target_day = pd.Timestamp(val_ts).normalize()
    df_d1_past = df_d1[df_d1.index < target_day]
    if len(df_d1_past) < 1:
        return False, {**out, "reason": "no_pd"}
    pdh = float(df_d1_past["high"].iloc[-1])
    pdl = float(df_d1_past["low"].iloc[-1])
    mid = (pdh + pdl) / 2
    price = float(df_m5["close"].iloc[val_idx])
    out["pd_a"] = (ob.direction == "bullish" and price < mid) or \
                  (ob.direction == "bearish" and price > mid)
    if not out["pd_a"]:
        return False, {**out, "reason": "pd_misaligned"}

    # ATR ratio + hour_fr + disp pour scoring tier
    atr = atr_safe(df_m5, val_idx)
    atr_50 = atr_safe(df_m5, val_idx, period=50)
    out["atr_ratio"] = atr / atr_50 if atr_50 > 0 else 1.0
    out["hour_fr"] = val_ts.tz_convert("Europe/Paris").hour + val_ts.tz_convert("Europe/Paris").minute / 60
    out["disp"] = abs(ob.displacement_atr)
    out["atr"] = atr

    return True, out


# =====================================================================
# Scoring tier S/A/B/C/D
# =====================================================================

def score_setup(hour_fr: float, h1_mom: float, disp: float, ob_size_atr: float) -> tuple[str, int]:
    s = 0
    if 14 <= hour_fr < 17: s += 3
    elif 17 <= hour_fr < 21: s += 2
    elif 8 <= hour_fr < 11: s += 2
    elif 11 <= hour_fr < 14: s += 1
    elif 5 <= hour_fr < 8: s += 1
    else: s -= 2
    if h1_mom > 1.0: s += 3
    elif h1_mom > 0.5: s += 2
    elif h1_mom > 0.3: s += 1
    if disp > 1.0: s += 2
    elif disp > 0.5: s += 1
    if 0.5 <= ob_size_atr <= 3.0: s += 1
    if s >= 7: return "S", s
    if s >= 5: return "A", s
    if s >= 3: return "B", s
    if s >= 1: return "C", s
    return "D", s


# =====================================================================
# Sizing : convertit risk_amount EUR + risk_pts en lots, plafonne a LOT_MAX
# =====================================================================

def calc_lots(asset: str, risk_amount_eur: float, risk_pts: float) -> float | None:
    """Renvoie nb lots a trader, ou None si pas tradeable (min_lot pas atteint)."""
    info = mt5.symbol_info(broker_sym(asset))
    if info is None:
        return None
    cs = info.trade_contract_size
    if cs <= 0 or risk_pts <= 0:
        return None
    lots_target = risk_amount_eur / (risk_pts * cs)
    # Plafond absolu
    lots = min(lots_target, LOT_MAX)
    # Arrondi au step et au min broker
    vol_step = info.volume_step
    vol_min = info.volume_min
    lots = round(lots / vol_step) * vol_step
    lots = round(lots, 2)
    if lots < vol_min:
        return None
    return lots


# =====================================================================
# Garde-fous : check avant chaque trade
# =====================================================================

def check_guards(mt5_exec: MT5Executor, asset: str, now_utc: pd.Timestamp) -> tuple[bool, str]:
    """Renvoie (ok, raison_si_refuse)."""
    global _liquidated

    if _liquidated:
        return False, "LIQUIDATED"

    # News blackout
    if is_news_blackout(now_utc):
        return False, "news_blackout"

    # Max positions
    n_open = len(mt5_exec.get_open_positions(magic=BOT_MAGIC))
    if n_open >= MAX_CONCURRENT:
        return False, f"max_positions_{n_open}"

    # Circuit breaker journalier
    today = now_utc.normalize()
    if today in _daily_blocked:
        return False, "cb_day_blocked"

    # Liquidation
    balance = mt5_exec.get_balance()
    if _initial_balance and balance < _initial_balance * (1 - LIQUIDATION_PCT):
        _liquidated = True
        log.error(f"LIQUIDATION : balance={balance:.2f} < {_initial_balance * (1-LIQUIDATION_PCT):.2f}")
        return False, "LIQUIDATION_TRIGGERED"

    return True, "OK"


def update_daily_cb(mt5_exec: MT5Executor, now_utc: pd.Timestamp):
    """Snapshot balance debut de jour + check si CB doit etre active."""
    today = now_utc.normalize()
    balance = mt5_exec.get_balance()
    if today not in _daily_open_balance:
        _daily_open_balance[today] = balance
        log.info(f"  [CB] nouveau jour {today.date()}, balance open={balance:.2f}")
        return
    # Check intra-day loss
    loss_pct = (_daily_open_balance[today] - balance) / _daily_open_balance[today]
    if loss_pct >= DAILY_STOP_PCT and today not in _daily_blocked:
        _daily_blocked.add(today)
        log.warning(f"  [CB] CIRCUIT BREAKER -{DAILY_STOP_PCT*100:.0f}% active "
                     f"({_daily_open_balance[today]:.2f} -> {balance:.2f})")


# =====================================================================
# Pipeline V22 par actif
# =====================================================================

def process_asset(asset: str, mt5_exec: MT5Executor, pusher: DashboardPusher) -> dict:
    """Scan + applique config V22 + decision pour 1 actif."""
    t_asset = time.time()
    res = {"asset": asset, "n_obs": 0, "n_passes_rule": 0, "n_traded": 0,
           "rejets": [], "errors": []}
    cfg = PER_ASSET_CONFIG[asset]

    try:
        # Fetch M5/H1/D1
        df_m5 = fetch_ohlcv(asset, mt5.TIMEFRAME_M5, 500)
        df_h1 = fetch_ohlcv(asset, mt5.TIMEFRAME_H1, 300)
        df_d1 = fetch_ohlcv(asset, mt5.TIMEFRAME_D1, 60)
        if df_m5 is None or len(df_m5) < 100:
            res["errors"].append("M5_insufficient")
            return res
        if df_d1 is None or len(df_d1) < 5:
            res["errors"].append("D1_insufficient")
            return res

        # Detect OBs sur M5 (filtre cons selon config actif)
        obs = find_obs_simple(df_m5, as_of_index=len(df_m5) - 1, min_consecutive=cfg["cons"])

        # Filtre OBs recents (last_bar - RECENT_CUTOFF_MIN)
        last_bar = df_m5.index[-1]
        cutoff = last_bar - pd.Timedelta(minutes=60)
        if _boot_ts is not None and _boot_ts > cutoff:
            cutoff = _boot_ts
        obs_recent = [ob for ob in obs if ob.validation_ts >= cutoff]
        res["n_obs"] = len(obs_recent)
        if not obs_recent:
            return res

        # Pour chaque OB recent
        for ob in obs_recent:
            ob_key = (asset, ob.validation_ts.isoformat(), ob.direction)
            if ob_key in _seen_obs:
                continue
            _seen_obs.add(ob_key)

            # Filtre USER (D1+H1+PD)
            ok, details = passes_user_rule(df_m5, df_h1, df_d1, ob)
            if not ok:
                res["rejets"].append({"asset": asset, "ts": ob.validation_ts.isoformat(),
                                       "direction": ob.direction, "reason": details["reason"]})
                continue

            # Filtres config actif (hour, h1_mom, disp, atr_ratio)
            if not (cfg["h_min"] <= details["hour_fr"] < cfg["h_max"]):
                res["rejets"].append({"asset": asset, "ts": ob.validation_ts.isoformat(),
                                       "direction": ob.direction, "reason": "hour_out_of_range"})
                continue
            if details["h1_mom"] < cfg["h1m"]:
                res["rejets"].append({"asset": asset, "ts": ob.validation_ts.isoformat(),
                                       "direction": ob.direction, "reason": "h1_mom_low"})
                continue
            if details["disp"] < cfg["disp"]:
                res["rejets"].append({"asset": asset, "ts": ob.validation_ts.isoformat(),
                                       "direction": ob.direction, "reason": "disp_low"})
                continue
            if cfg["atr_max"] is not None and details["atr_ratio"] >= cfg["atr_max"]:
                res["rejets"].append({"asset": asset, "ts": ob.validation_ts.isoformat(),
                                       "direction": ob.direction, "reason": "atr_too_high"})
                continue

            res["n_passes_rule"] += 1

            # Compute entry/SL/TP
            atr = details["atr"]
            if ob.direction == "bullish":
                sl = ob.ob_low - cfg["sl_buf"] * atr
                entry = ob.ob_high
                risk = abs(entry - sl)
                tp = entry + TP_RR * risk
            else:
                sl = ob.ob_high + cfg["sl_buf"] * atr
                entry = ob.ob_low
                risk = abs(entry - sl)
                tp = entry - TP_RR * risk

            # Scoring tier
            ob_size_atr = (ob.ob_high - ob.ob_low) / atr if atr > 0 else 0
            tier, score = score_setup(details["hour_fr"], details["h1_mom"],
                                        details["disp"], ob_size_atr)

            # Tier D = skip
            if tier == "D":
                res["rejets"].append({"asset": asset, "ts": ob.validation_ts.isoformat(),
                                       "direction": ob.direction, "reason": "tier_D_skip",
                                       "tier": tier, "score": score})
                continue

            # Garde-fous
            now_utc = pd.Timestamp.now(tz="UTC")
            ok_guard, guard_reason = check_guards(mt5_exec, asset, now_utc)
            if not ok_guard:
                res["rejets"].append({"asset": asset, "ts": ob.validation_ts.isoformat(),
                                       "direction": ob.direction, "reason": f"guard_{guard_reason}",
                                       "tier": tier})
                continue

            # Calcul lots
            balance = mt5_exec.get_balance()
            risk_pct = TIER_RISK[tier]
            risk_amount = balance * risk_pct
            risk_pts = abs(entry - sl)
            lots = calc_lots(asset, risk_amount, risk_pts)
            if lots is None:
                res["rejets"].append({"asset": asset, "ts": ob.validation_ts.isoformat(),
                                       "direction": ob.direction, "reason": "no_lot_size",
                                       "tier": tier})
                continue

            # SPLIT en 2 ordres (Vantage ne supporte pas la fermeture partielle) :
            #  - Order 1 (50% vol) -> TP = 1R  (= "partial")
            #  - Order 2 (50% vol) -> TP = 2R  (= "runner", SL movera a BE quand 1R touche)
            # On respecte le volume_step et le volume_min du broker.
            info = mt5.symbol_info(broker_sym(asset))
            vol_min = info.volume_min if info else 0.01
            vol_step = info.volume_step if info else 0.01
            half_raw = lots / 2
            half = round(half_raw / vol_step) * vol_step
            half = round(half, 2)
            # Si half < vol_min, on ne peut pas splitter : on prend 1 seul ordre @ vol_min, TP 2R
            can_split = (half >= vol_min) and ((lots - half) >= vol_min)

            # TP 1R = entry + 1*risk (partial 50%)
            risk_amt = abs(entry - sl)
            if ob.direction == "bullish":
                tp_1r = entry + 1.0 * risk_amt
            else:
                tp_1r = entry - 1.0 * risk_amt

            tickets = []
            if can_split:
                log.info(f"  [TRADE-SPLIT] {asset} {ob.direction} tier={tier} score={score} "
                          f"total_lots={lots:.2f} = 2x{half:.2f} | entry={entry:.5f} sl={sl:.5f} "
                          f"tp1R={tp_1r:.5f} tp2R={tp:.5f}")
                # Order 1 : 50% TP 1R
                r1 = mt5_exec.place_market_order(
                    symbol=asset, direction=ob.direction, volume=half,
                    sl=sl, tp=tp_1r,
                    comment=f"V22-{tier}-{ob.direction[:1].upper()}-P",  # P = Partial
                    magic=BOT_MAGIC,
                )
                # Order 2 : 50% TP 2R (runner)
                r2 = mt5_exec.place_market_order(
                    symbol=asset, direction=ob.direction, volume=half,
                    sl=sl, tp=tp,
                    comment=f"V22-{tier}-{ob.direction[:1].upper()}-R",  # R = Runner
                    magic=BOT_MAGIC,
                )
                if r1 is None and r2 is None:
                    res["errors"].append(f"order_failed_both_{asset}")
                    continue
                if r1: tickets.append(("P", r1))
                if r2: tickets.append(("R", r2))
                # Enregistre les 2 tickets pour suivi BE
                if r1 and r2:
                    _open_pairs[r1["ticket"]] = {
                        "asset": asset, "direction": ob.direction,
                        "entry": r1["price"], "sl": sl, "tp_1r": tp_1r,
                        "runner_ticket": r2["ticket"], "be_moved": False,
                    }
            else:
                # Volume trop petit pour splitter : 1 seul ordre TP 2R
                log.info(f"  [TRADE-SINGLE] {asset} {ob.direction} tier={tier} score={score} "
                          f"lots={lots:.2f} (split impossible: half={half:.2f} < min={vol_min}) "
                          f"entry={entry:.5f} sl={sl:.5f} tp={tp:.5f}")
                r1 = mt5_exec.place_market_order(
                    symbol=asset, direction=ob.direction, volume=lots,
                    sl=sl, tp=tp,
                    comment=f"V22-{tier}-{ob.direction[:1].upper()}",
                    magic=BOT_MAGIC,
                )
                if r1 is None:
                    res["errors"].append(f"order_failed_{asset}")
                    continue
                tickets.append(("F", r1))   # F = Full

            res["n_traded"] += 1

            # Track trades pour detect close + push TRADE_CLOSED apres
            opened_ts = pd.Timestamp.now(tz="UTC")
            current_balance = mt5_exec.get_balance()
            for tag, r in tickets:
                tp_this = tp_1r if tag == "P" else tp
                _open_trades[r["ticket"]] = {
                    "asset": asset, "direction": ob.direction,
                    "entry": r["price"], "sl": sl, "tp": tp_this,
                    "volume": r["volume"], "opened_ts": opened_ts,
                    "opened_balance": current_balance, "tag": tag,
                    "tier": tier, "score": score,
                }

            # Push dashboard (1 setup, 1 ou 2 trades)
            kz_label = f"FR_h{int(details['hour_fr'])}"
            # Tier S/A/B/C en "ml_proba" (le dashboard l'affiche comme un score qualite 0-1)
            tier_proba = {"S": 0.95, "A": 0.80, "B": 0.65, "C": 0.50, "D": 0.30}.get(tier, 0.0)
            try:
                first = tickets[0][1]
                pusher.push_setup(
                    instrument=asset, ts=ob.validation_ts,
                    direction=ob.direction,
                    entry_price=float(first["price"]),
                    sl=float(sl), tp=float(tp), rr=float(TP_RR),
                    score=int(score),
                    ml_proba=float(tier_proba),
                    killzone=kz_label,
                )
                for tag, r in tickets:
                    tp_this = tp_1r if tag == "P" else tp
                    pusher.push_trade_executed(
                        instrument=asset, ticket=int(r["ticket"]),
                        direction=ob.direction,
                        entry=float(r["price"]),
                        sl=float(sl), tp=float(tp_this),
                        volume=float(r["volume"]),
                        rr=float(TP_RR),
                        ml_proba=float(tier_proba),
                        score=int(score),
                        killzone=kz_label,
                    )
            except Exception as e:
                log.warning(f"  push dashboard fail : {e}")

    except Exception as e:
        res["errors"].append(f"asset_{asset}: {type(e).__name__}: {str(e)[:120]}")

    # Latence (ms) pour le dashboard
    res["latency_ms"] = int((time.time() - t_asset) * 1000)
    return res


# =====================================================================
# MAIN
# =====================================================================

def manage_open_pairs(mt5_exec: MT5Executor):
    """Pour chaque paire (partial, runner) :
    - Si le partial a touche son TP 1R (= position fermee) -> bouge le SL du runner a BE (= entry)
    - Si le partial a touche son SL (= position fermee en perte) -> on laisse le runner sur son SL
    - Si les 2 sont deja fermes -> retire la paire du tracking
    """
    if not _open_pairs:
        return
    to_remove = []
    open_tickets = {p["ticket"]: p for p in mt5_exec.get_open_positions(magic=BOT_MAGIC)}
    for partial_ticket, pair in list(_open_pairs.items()):
        partial_pos = open_tickets.get(partial_ticket)
        runner_pos = open_tickets.get(pair["runner_ticket"])
        # Si les 2 sont fermes -> cleanup
        if partial_pos is None and runner_pos is None:
            log.info(f"  [PAIR-DONE] partial={partial_ticket} runner={pair['runner_ticket']} ({pair['asset']})")
            to_remove.append(partial_ticket)
            continue
        # Si le partial est FERME mais runner ouvert : verifier si BE a deja ete bouge
        if partial_pos is None and runner_pos is not None and not pair["be_moved"]:
            # Bouger le SL du runner a entry (BE)
            entry_be = pair["entry"]
            # Pour bullish on s'assure que BE > SL actuel ; pour bearish que BE < SL actuel
            ok = False
            try:
                req = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": pair["runner_ticket"],
                    "sl": entry_be,
                    "tp": runner_pos["tp"],
                    "symbol": runner_pos["symbol"],
                }
                result = mt5.order_send(req)
                if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                    ok = True
            except Exception as e:
                log.warning(f"  [BE-FAIL] runner={pair['runner_ticket']}: {e}")
            if ok:
                pair["be_moved"] = True
                log.info(f"  [BE-MOVED] {pair['asset']} runner={pair['runner_ticket']} SL -> {entry_be:.5f} (entry)")
            else:
                log.warning(f"  [BE-FAIL] runner={pair['runner_ticket']} ({pair['asset']}) retcode KO")
    for k in to_remove:
        _open_pairs.pop(k, None)


def detect_closed_trades(mt5_exec: MT5Executor, pusher) -> int:
    """Detecte les trades qui se sont fermes et push TRADE_CLOSED.
    Met a jour _live_stats. Retourne nb trades fermes ce cycle."""
    if not _open_trades:
        return 0
    open_tickets = {p["ticket"] for p in mt5_exec.get_open_positions(magic=BOT_MAGIC)}
    closed_now = []
    for ticket, info in list(_open_trades.items()):
        if ticket not in open_tickets:
            closed_now.append((ticket, info))
            _open_trades.pop(ticket, None)
    if not closed_now:
        return 0

    # Pour chaque trade ferme, recup le deal MT5 pour avoir le PnL reel
    now_ts = pd.Timestamp.now(tz="UTC")
    from_ts = now_ts - pd.Timedelta(hours=12)
    try:
        all_closed = mt5_exec.get_closed_deals(from_ts, now_ts, magic=BOT_MAGIC)
    except Exception:
        all_closed = []
    deals_by_ticket: dict[int, dict] = {}
    for d in all_closed:
        # On match par position_id (ticket de l'ordre d'origine)
        pos_id = d.get("position_id", d.get("ticket"))
        if pos_id:
            deals_by_ticket[int(pos_id)] = d

    for ticket, info in closed_now:
        deal = deals_by_ticket.get(ticket, {})
        pnl_real = float(deal.get("profit", 0.0)) + float(deal.get("commission", 0.0)) + float(deal.get("swap", 0.0))
        duration_min = (now_ts - info["opened_ts"]).total_seconds() / 60
        # Outcome : WIN si PnL > 0, LOSS sinon
        outcome = "WIN" if pnl_real > 0 else ("LOSS" if pnl_real < 0 else "BE")
        # Update stats live
        _live_stats["total"] += 1
        if pnl_real > 0:
            _live_stats["wins"] += 1
        elif pnl_real < 0:
            _live_stats["losses"] += 1
        _live_stats["pnl_total"] += pnl_real
        log.info(f"  [CLOSED] {info['asset']} ticket={ticket} {outcome} pnl={pnl_real:+.2f}EUR "
                  f"duration={duration_min:.0f}min")
        # Push TRADE_CLOSED
        try:
            if pusher:
                pusher.push_trade_closed(
                    ticket=ticket, outcome=outcome,
                    pnl_real=pnl_real, duration_min=duration_min,
                    instrument=info["asset"],
                )
        except Exception as e:
            log.warning(f"  push trade_closed fail : {e}")
    return len(closed_now)


def cleanup_seen_obs():
    """Toutes les 30 cycles, vire les OBs de >2h pour eviter memory leak."""
    global _seen_obs
    if len(_seen_obs) < 5000: return
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=2)
    new_set = set()
    for k in _seen_obs:
        try:
            ts = pd.Timestamp(k[1])
            if ts >= cutoff:
                new_set.add(k)
        except Exception:
            new_set.add(k)
    log.info(f"  [cleanup] _seen_obs : {len(_seen_obs)} -> {len(new_set)}")
    _seen_obs = new_set


def main():
    global _boot_ts, _initial_balance, _cycle_count, _liquidated

    log.info("=" * 80)
    log.info("V22 LIVE RUNNER - demarrage")
    log.info(f"Assets: {len(ASSETS)} | TIER_RISK={TIER_RISK} | "
              f"LOT_MAX={LOT_MAX} | MAX_CONCURRENT={MAX_CONCURRENT}")
    log.info(f"Daily CB: -{DAILY_STOP_PCT*100:.0f}% | Liquidation: -{LIQUIDATION_PCT*100:.0f}%")
    log.info("=" * 80)

    # Init MT5
    mt5_exec = MT5Executor()
    login = int(os.environ.get("VANTAGE_LOGIN", "25420721"))
    pwd = os.environ.get("VANTAGE_PASSWORD")
    server = os.environ.get("VANTAGE_SERVER", "VantageInternational-Demo")
    if not mt5_exec.initialize(login=login, password=pwd, server=server):
        log.error("MT5 init FAILED")
        return
    log.info(f"MT5 connecte : login={login} server={server} balance={mt5_exec.get_balance():.2f}")

    # Init balance de reference
    _initial_balance = float(os.environ.get("INITIAL_BALANCE", mt5_exec.get_balance()))
    log.info(f"Balance de reference : {_initial_balance:.2f}")

    # Boot ts (anti-rafale : on ignore les OBs vieux de plus de 5 min au demarrage)
    _boot_ts = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=5)
    log.info(f"_boot_ts : {_boot_ts}")

    # Init dashboard
    pusher = None
    try:
        dash_url = os.environ.get("DASHBOARD_URL")
        session_id = f"v22_{int(time.time())}"
        pusher = DashboardPusher(url=dash_url, session_id=session_id)
        pusher.push_start("V22 live runner demo")
        log.info(f"DashboardPusher initialise : url={dash_url} session={session_id}")
    except Exception as e:
        log.warning(f"Dashboard push fail : {e}")
        pusher = None

    # Activate tous les symboles
    n_active = 0
    for asset in ASSETS:
        try:
            if mt5.symbol_select(broker_sym(asset), True):
                n_active += 1
            else:
                log.warning(f"  symbol_select({broker_sym(asset)}) FAIL")
        except Exception as e:
            log.warning(f"  ensure_symbol({asset}) fail : {e}")
    log.info(f"Symboles actives : {n_active}/{len(ASSETS)}")

    # ============ Boucle principale ============
    try:
        while True:
            if _liquidated:
                log.error("LIQUIDATED - bot arrete")
                break

            _cycle_count += 1
            t_cycle = time.time()
            now_utc = pd.Timestamp.now(tz="UTC")

            # Update CB journalier
            update_daily_cb(mt5_exec, now_utc)

            # Gestion BE des paires partial+runner (Vantage : pas de fermeture partielle)
            try:
                manage_open_pairs(mt5_exec)
            except Exception as e:
                log.warning(f"  manage_open_pairs error : {e}")

            # Detect closed trades + maj stats live + push TRADE_CLOSED
            try:
                n_closed = detect_closed_trades(mt5_exec, pusher)
                if n_closed > 0:
                    log.info(f"  [STATS] {n_closed} trades fermes -> total={_live_stats['total']} "
                              f"W={_live_stats['wins']} L={_live_stats['losses']} "
                              f"PnL={_live_stats['pnl_total']:+.2f}EUR")
            except Exception as e:
                log.warning(f"  detect_closed error : {e}")

            # Process tous les actifs sequentiellement (22 actifs)
            n_obs_total = 0; n_passes_total = 0; n_traded_total = 0
            all_rejets = []; all_errors = []
            for asset in ASSETS:
                try:
                    r = process_asset(asset, mt5_exec, pusher) if pusher else process_asset(asset, mt5_exec, _NoopPusher())
                    n_obs_total += r["n_obs"]
                    n_passes_total += r["n_passes_rule"]
                    n_traded_total += r["n_traded"]
                    all_rejets.extend(r["rejets"])
                    all_errors.extend(r["errors"])
                    _last_latencies[asset] = r.get("latency_ms", 0)
                except Exception as e:
                    log.error(f"  asset {asset} crash : {type(e).__name__}: {e}")

            # Cleanup periodique
            if _cycle_count % 30 == 0:
                cleanup_seen_obs()

            elapsed = time.time() - t_cycle

            # Push stats / cycle / positions_sync
            balance = mt5_exec.get_balance()
            equity = mt5_exec.get_equity()
            open_positions = mt5_exec.get_open_positions(magic=BOT_MAGIC)
            n_open = len(open_positions)
            tot = _live_stats["total"]
            wr_pct = (_live_stats["wins"] / tot * 100) if tot > 0 else 0.0

            try:
                if pusher:
                    pusher.push_stats(
                        total=_live_stats["total"],
                        wins=_live_stats["wins"],
                        losses=_live_stats["losses"],
                        wr_pct=wr_pct,
                        pnl_total=_live_stats["pnl_total"],
                        balance=balance, equity=equity,
                        positions_open=n_open,
                    )
                    if all_rejets:
                        pusher.push_rejected_batch(all_rejets[:50])
                    pusher.push_cycle(
                        actifs_scanned=len(ASSETS),
                        total_s=elapsed, fetch_s=0.0, compute_s=elapsed,
                        latencies=dict(_last_latencies),
                    )
                    # POSITIONS_SYNC pour KPI + statut FILLED des trades
                    sync_positions = [
                        {"ticket": p["ticket"], "pnl": p.get("pnl", p.get("profit", 0.0))}
                        for p in open_positions
                    ]
                    pusher.push_positions_sync(
                        open_positions=sync_positions,
                        pending_tickets=[],
                    )
            except Exception as e:
                log.warning(f"push stats fail : {e}")

            log.info(f"  cycle #{_cycle_count:5d} | OBs={n_obs_total:>4} pass_rule={n_passes_total:>3} "
                      f"trades={n_traded_total:>2} | balance={balance:.2f} eq={equity:.2f} open={n_open} "
                      f"| {elapsed:.1f}s ({len(all_errors)} errors)")
            if all_errors:
                log.debug(f"  errors sample : {all_errors[:3]}")

            # Sleep jusqu'a la prochaine minute boundary
            sleep_s = max(5, 60 - elapsed)
            time.sleep(sleep_s)

    except KeyboardInterrupt:
        log.info("Bot stoppe (Ctrl+C)")
    except Exception as e:
        log.exception(f"CRASH boucle main : {e}")
    finally:
        try:
            if pusher:
                pusher.push_stop("V22 live runner arrete")
        except Exception:
            pass
        mt5_exec.shutdown()
        log.info("Bot arrete proprement")


class _NoopPusher:
    """Fallback si DashboardPusher fail."""
    def push_start(self, *a, **k): pass
    def push_stop(self, *a, **k): pass
    def push_error(self, *a, **k): pass
    def push_cycle(self, *a, **k): pass
    def push_diag(self, *a, **k): pass
    def push_setup(self, *a, **k): pass
    def push_setup_cancelled(self, *a, **k): pass
    def push_rejected_batch(self, *a, **k): pass
    def push_trade_executed(self, *a, **k): pass
    def push_trade_closed(self, *a, **k): pass
    def push_stats(self, *a, **k): pass
    def push_positions_sync(self, *a, **k): pass


if __name__ == "__main__":
    main()
