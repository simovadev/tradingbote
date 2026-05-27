"""LIVE RUNNER V20 - bot dedie au Deep Learning V20 (ICT universel sans embedding).

Design propre, focalise V20 :
- V20Predictor (PyTorch) en moteur principal
- MT5 demo Vantage en execution
- DashboardPusher (Railway) pour le monitoring
- 1 cycle = 1 scan/minute, 1 OB valide = 1 prediction V20 = 1 decision

Configuration via env :
- DASHBOARD_URL : URL Railway (push events)
- BOT_THRESHOLD : seuil ML pour trader (default 0.30)
- RECENT_CUTOFF_MIN : age max d'un OB pour etre evalue (default 60)
- RISK_PCT : risque par trade (default 0.005 = 0.5% du compte)

Usage : python -m bot_v2.live_runner_v20
"""
from __future__ import annotations

# CRITIQUE Windows : torch en premier avant tout bot_v2.*
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

# ROOT du projet
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Defaults env (setdefault = respecte l'env si deja defini)
os.environ.setdefault("DASHBOARD_URL", "https://tradingbote-production.up.railway.app/api/ingest")
os.environ.setdefault("BOT_THRESHOLD", "0.60")
os.environ.setdefault("RECENT_CUTOFF_MIN", "60")

# ============ Logging ============
log = logging.getLogger("live_v20")
log.setLevel(logging.INFO)
_fh = logging.FileHandler(ROOT / "live_v20.log", mode="a", encoding="utf-8")
_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
log.addHandler(_fh)
_sh = logging.StreamHandler()
_sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
log.addHandler(_sh)


# ============ Imports bot_v2 (apres torch+env) ============
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
from bot_v2.v20_inference import V20Predictor, predict_proba_v20, is_v20_available
from bot_v2.push_dashboard import DashboardPusher


# ============ Configuration ============
# Mapping nom actif (asset_id du model) -> nom broker MT5 (peut avoir suffixe + pour RAW ECN)
# Genere via _vps_test_all_28_plus.py sur le compte demo Vantage RAW
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
    "ETHUSD":    "ETHUSD",
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
    # KO sur compte RAW : USDZAR (INVALID_STOPS), Wheat-C (MARKET_CLOSED)
}

# Actifs qui necessitent un SL min plus large que trade_stops_level (broker ment)
# Format : asset -> SL min en % du prix
MIN_SL_PCT = {
    "BTCUSD": 0.001,  # 0.1% du prix BTC (broker dit 0 mais rejette < 0.1%)
}

# ASSETS = noms actifs (utilises pour V20Predictor + features). Lookup MT5 via BROKER_MAP.
ASSETS = list(BROKER_MAP.keys())

THRESHOLD = float(os.environ["BOT_THRESHOLD"])
RECENT_CUTOFF_MIN = int(os.environ["RECENT_CUTOFF_MIN"])

BOT_MAGIC = 20000  # ordres normaux V20

MAX_CONCURRENT_TRADES = 5  # cap global ordres simultanes


# ============ State global ============
_seen_obs: set[tuple] = set()  # (asset, ts_ob) deja traites pour pas re-trader
_boot_ts: pd.Timestamp | None = None  # UTC : OBs anterieurs ignores


# ============ Fetch data ============
def broker_sym(asset: str) -> str:
    """Traduit un nom actif en nom broker MT5 (avec suffixe + si necessaire)."""
    return BROKER_MAP.get(asset, asset)


def fetch_ohlcv(asset: str, tf, n: int = 500) -> pd.DataFrame | None:
    """Recupere les N dernieres bougies d'un actif (utilise nom broker)."""
    rates = mt5.copy_rates_from_pos(broker_sym(asset), tf, 0, n)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.rename(columns={"tick_volume": "volume"})
    return df.set_index("time").sort_index()


# ============ Pipeline V20 par actif ============
def process_asset(asset: str, predictor: V20Predictor | None,
                   pusher: DashboardPusher, mt5_exec: MT5Executor,
                   balance: float) -> dict:
    """Scan + predict V20 + trade decision pour 1 actif.

    Returns dict avec stats : {asset, n_obs, rejets, setups, trades_taken}
    """
    res = {"asset": asset, "n_obs": 0, "rejets": [], "setups": [],
           "trades_taken": 0, "errors": []}

    try:
        # 1. Fetch data multi-TF
        df_m1 = fetch_ohlcv(asset, mt5.TIMEFRAME_M1, 500)
        df_m15 = fetch_ohlcv(asset, mt5.TIMEFRAME_M15, 200)
        df_h1 = fetch_ohlcv(asset, mt5.TIMEFRAME_H1, 200)
        df_d1 = fetch_ohlcv(asset, mt5.TIMEFRAME_D1, 60)

        if df_m1 is None or len(df_m1) < 100:
            res["errors"].append("M1 insuffisant")
            return res

        # 2. Detect OBs
        swing_strength = get_param(asset, "swing_strength_m1", 2)
        obs = detect_order_blocks(df_m1, swing_strength=swing_strength)

        # Filtre OBs recents (validation_ts >= last_bar - RECENT_CUTOFF_MIN)
        # ET >= _boot_ts (anti-rafale au demarrage : on ne trade pas les OBs deja vieux)
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
        # Debug : 1ere fois qu'on voit un OB filtre, on log la comparaison
        n_total = len([ob for ob in obs if ob.validation_ts is not None])
        if n_total > len(obs_recent) and asset == ASSETS[0]:
            sample = next((ob for ob in obs if ob.validation_ts is not None), None)
            if sample:
                log.info(f"  [filter] {asset} : {len(obs_recent)}/{n_total} OBs. cutoff={cutoff} (boot={_boot_ts}, age={cutoff_age}). sample_ts={sample.validation_ts} type={type(sample.validation_ts).__name__}")
        res["n_obs"] = len(obs_recent)

        if not obs_recent:
            return res

        # 3. Pour chaque OB recent, evalue
        for ob in obs_recent:
            ob_key = (asset, ob.validation_ts.isoformat(), ob.direction)
            if ob_key in _seen_obs:
                continue  # deja traite

            try:
                r = evaluate_ob(
                    ob, df_m1, df_m15, df_d1,
                    instrument=asset,
                    df_htf2=df_h1, htf2_name="H1",
                )
                if r is None or r.verdict != "TRADE" or r.trade_setup is None:
                    reason = r.rejection_reason if r else "no_result"
                    _seen_obs.add(ob_key)
                    res["rejets"].append({"asset": asset, "ts": ob.validation_ts,
                                           "direction": ob.direction, "reason": reason or "no_trade"})
                    continue

                # 4. PREDICT V20 (ou fallback)
                proba = None
                if predictor is not None:
                    try:
                        proba = predict_proba_v20(
                            r, ob, asset,
                            df_m1=df_m1, df_m15=df_m15, df_h1=df_h1,
                            df_d1=df_d1, mss_setups=None,
                        )
                    except Exception as e:
                        res["errors"].append(f"V20 predict {asset}: {e}")

                if proba is None:
                    # V20 fail / indispo : on skip ce trade (mode V20 only)
                    _seen_obs.add(ob_key)
                    res["rejets"].append({"asset": asset, "ts": ob.validation_ts,
                                           "direction": ob.direction, "reason": "V19_unavailable"})
                    continue

                # 5. Decision
                _seen_obs.add(ob_key)
                kz = killzone_at(ob.validation_ts) or "None"
                setup = r.trade_setup

                # Push REJECTED ou SETUP au dashboard
                if proba >= THRESHOLD:
                    # Trade NORMAL (sens de l'OB)
                    pusher.push_setup(
                        instrument=asset, ts=ob.validation_ts,
                        direction=ob.direction,
                        entry_price=setup.entry_price,
                        sl=setup.stop_loss, tp=setup.take_profit,
                        rr=setup.rr, score=r.score or 0,
                        ml_proba=proba, killzone=kz,
                    )
                    if execute_trade(mt5_exec, asset, setup, proba, balance,
                                      ob, kz, pusher):
                        res["trades_taken"] += 1
                    res["setups"].append({"asset": asset, "proba": proba})

                else:
                    # Proba < THRESHOLD : skip + log REJECTED
                    res["rejets"].append({
                        "asset": asset, "ts": ob.validation_ts,
                        "direction": ob.direction,
                        "reason": f"ml_below_thr_{proba:.3f}",
                        "proba": proba,
                        "threshold": THRESHOLD,
                        "entry": setup.entry_price,
                        "sl": setup.stop_loss, "tp": setup.take_profit, "rr": setup.rr,
                    })
            except Exception as e:
                res["errors"].append(f"OB process fail {asset}: {type(e).__name__}: {str(e)[:100]}")

    except Exception as e:
        res["errors"].append(f"asset {asset} fail: {type(e).__name__}: {str(e)[:100]}")

    return res


# ============ Execution MT5 ============
def execute_trade(mt5_exec: MT5Executor, asset: str, setup, proba: float,
                   balance: float, ob, killzone: str,
                   pusher: DashboardPusher) -> bool:
    """Place un ordre MT5 (market) dans le sens de l'OB."""
    try:
        direction = setup.direction
        sl = float(setup.stop_loss)
        tp = float(setup.take_profit)
        entry = float(setup.entry_price)
        rr = float(setup.rr)

        # Sizing proportionnel au compte : lots = (risk_pct * balance) / risk_per_lot
        risk_pct = float(os.getenv("RISK_PCT", "0.005"))
        risk_eur = balance * risk_pct
        bsym = broker_sym(asset)
        info = mt5.symbol_info(bsym)
        if info is None:
            log.error(f"{asset} ({bsym}) : symbol_info None")
            return False

        # Distance SL en points
        tick_value = getattr(info, "trade_tick_value", 1.0)
        tick_size = getattr(info, "trade_tick_size", 0.01)
        # Prix marche actuel
        tick = mt5.symbol_info_tick(bsym)
        if tick is None:
            log.error(f"{asset} ({bsym}) : tick None")
            return False
        market_price = tick.ask if direction == "bullish" else tick.bid
        sl_distance = abs(market_price - sl)
        if sl_distance <= 0:
            log.warning(f"{asset} sl_distance=0, skip")
            return False

        # Respect stop-level broker (distance min SL/TP en points)
        # Plus : MIN_SL_PCT pour les actifs ou le broker ment (BTC stops_level=0)
        stops_level = getattr(info, "trade_stops_level", 0) * info.point
        min_pct = MIN_SL_PCT.get(asset)
        if min_pct:
            min_dist_pct = market_price * min_pct
            min_required = max(stops_level, min_dist_pct)
        else:
            min_required = stops_level
        if min_required > 0 and sl_distance < min_required:
            # Elargit SL au minimum requis (au lieu de skip) - garde le ratio TP
            sl_extra = min_required - sl_distance
            if direction == "bullish":
                sl = sl - sl_extra
                tp = tp + sl_extra * rr
            else:
                sl = sl + sl_extra
                tp = tp - sl_extra * rr
            sl_distance = abs(market_price - sl)
            log.info(f"{asset} SL elargi a {sl_distance:.5f} (min broker {min_required:.5f})")

        n_ticks = sl_distance / tick_size
        risk_per_lot = n_ticks * tick_value
        if risk_per_lot <= 0:
            return False

        lots = risk_eur / risk_per_lot
        lots = max(info.volume_min, round(lots / info.volume_step) * info.volume_step)
        lots = min(lots, info.volume_max)

        # Place ordre
        comment = f"V20-{direction[0].upper()} ml={proba:.2f}"

        result = mt5_exec.place_market_order(
            symbol=bsym, direction=direction, volume=lots,
            sl=sl, tp=tp, comment=comment, magic=BOT_MAGIC,
        )
        if result is None:
            log.warning(f"{asset} : ordre rejete par MT5")
            return False

        log.info(
            f"TRADE OK {asset} {direction} "
            f"entry={result['price']:.5f} SL={sl:.5f} TP={tp:.5f} "
            f"vol={result['volume']:.2f} rr={rr:.2f} ml={proba:.3f}"
        )

        # Push trade au dashboard
        pusher.push_trade_executed(
            instrument=asset, ticket=result["ticket"],
            direction=direction, entry=result["price"],
            sl=sl, tp=tp, volume=result["volume"],
            rr=rr, ml_proba=proba, score=0,
            killzone=killzone,
        )
        return True
    except Exception as e:
        log.exception(f"execute_trade fail {asset}: {e}")
        return False


# ============ Main loop ============
def main():
    log.info("=" * 60)
    log.info("BOT V20 LIVE - DEEP LEARNING ICT")
    log.info("=" * 60)
    log.info(f"THRESHOLD = {THRESHOLD}")
    log.info(f"Mode V20 (ICT universel) : trade quand proba >= {THRESHOLD}")
    log.info(f"RECENT_CUTOFF_MIN = {RECENT_CUTOFF_MIN}")
    log.info(f"ASSETS = {ASSETS}")

    # 1. Charge V20
    log.info("Chargement V20 model...")
    predictor = V20Predictor.get_instance()
    if predictor is None:
        log.error("V20 model INDISPONIBLE - bot ne demarre pas")
        return
    log.info(f"V20 OK : device={predictor.device}, n_ict={predictor.n_ict} (sans asset_emb)")

    # 2. Init MT5
    mt5_exec = MT5Executor()
    if not mt5_exec.initialize():
        log.error(f"MT5 init failed : {mt5.last_error()}")
        return
    log.info(f"MT5 connecte : login={mt5_exec.account_info.login} server={mt5_exec.account_info.server}")
    log.info(f"  Balance : {mt5_exec.account_info.balance} {mt5_exec.account_info.currency}")
    log.info(f"  Trade mode : {mt5_exec.account_info.trade_mode} (0=demo, 2=real)")

    # Force symbol_select sur tous les actifs (nom broker, sinon symbol_info_tick = bid=0)
    activated, skipped = [], []
    for a in ASSETS:
        bsym = broker_sym(a)
        if mt5.symbol_select(bsym, True):
            activated.append(bsym)
        else:
            skipped.append((a, bsym))
    log.info(f"  Symbols actives : {len(activated)}/{len(ASSETS)}")
    if skipped:
        log.warning(f"  Symbols KO (introuvables broker) : {skipped}")

    # 3. Init dashboard pusher
    session_id = uuid.uuid4().hex[:8]
    pusher = DashboardPusher(
        url=os.getenv("DASHBOARD_URL"),
        session_id=session_id,
    )
    log.info(f"DashboardPusher : url={pusher.url}, session={session_id}")
    pusher.push_start(f"Bot V20 demarre (threshold={THRESHOLD})")

    # Anti-rafale au boot : on n'evalue que les OBs valides APRES le demarrage
    # IMPORTANT : aligne sur l'heure broker (last M1 candle) car MT5 timestamps
    # sont en heure broker, pas UTC reelle (Vantage = UTC+2)
    global _boot_ts
    boot_probe = mt5.copy_rates_from_pos(broker_sym("EURUSD"), mt5.TIMEFRAME_M1, 0, 1)
    if boot_probe is not None and len(boot_probe) > 0:
        _boot_ts = pd.to_datetime(boot_probe[0]["time"], unit="s", utc=True)
        log.info(f"BOOT_TS = {_boot_ts} (heure broker - OBs anterieurs ignores)")
    else:
        log.warning("BOOT_TS : impossible de lire last candle, fallback UTC")
        _boot_ts = pd.Timestamp.now(tz="UTC")

    log.info("=" * 60)
    log.info("DEMARRAGE BOUCLE PRINCIPALE - 1 cycle/min")
    log.info("=" * 60)

    cycle_n = 0
    try:
        while True:
            cycle_n += 1
            t0 = time.time()
            now = datetime.now(timezone.utc)

            # Compte positions ouvertes
            positions = mt5.positions_get() or []
            n_total_open = len(positions)

            # Refresh balance
            account = mt5.account_info()
            balance = account.balance if account else 0
            equity = account.equity if account else 0

            log.info(f"--- CYCLE {cycle_n} | T={now.strftime('%H:%M:%S')} UTC | balance={balance:.2f} | open={n_total_open} ---")

            # Process tous les actifs sequentiellement
            all_rejets = []
            all_setups = []
            total_obs = 0
            total_trades = 0

            for asset in ASSETS:
                res = process_asset(asset, predictor, pusher, mt5_exec, balance)
                total_obs += res["n_obs"]
                total_trades += res["trades_taken"]
                all_rejets.extend(res["rejets"])
                all_setups.extend(res["setups"])
                if res["errors"]:
                    for e in res["errors"]:
                        log.warning(f"  {asset} : {e}")

            elapsed = time.time() - t0
            log.info(f"  Total : {total_obs} OBs | {len(all_setups)} setups | {len(all_rejets)} rejets | {total_trades} trades (cycle {elapsed:.1f}s)")

            # Push CYCLE + STATS
            pusher.push_cycle(
                actifs_scanned=len(ASSETS),
                total_s=elapsed, fetch_s=0, compute_s=elapsed,
                latencies={a: 0 for a in ASSETS},
                last_bar_ts=now,
            )

            stats = mt5_exec.get_stats() if hasattr(mt5_exec, "get_stats") else {}
            pusher.push_stats(
                total=stats.get("total", 0),
                wins=stats.get("wins", 0),
                losses=stats.get("losses", 0),
                wr_pct=stats.get("wr_pct", 0),
                pnl_total=stats.get("pnl_total", 0),
                balance=balance, equity=equity,
                positions_open=n_total_open, pending_orders=0,
            )

            # Push REJECTED batch (avec filtre 30min)
            if all_rejets:
                _real_cutoff = now - pd.Timedelta(minutes=30)
                _filtered = []
                for r in all_rejets:
                    ts = r.get("ts")
                    if hasattr(ts, "tz_localize"):
                        if ts.tz is None:
                            ts = ts.tz_localize("UTC")
                    try:
                        if ts >= _real_cutoff:
                            _filtered.append({
                                "instrument": r["asset"],
                                "ts": str(r["ts"]),
                                "direction": r.get("direction", "?"),
                                "reason": r.get("reason", "?"),
                                "ml_proba": r.get("proba"),
                                "threshold": r.get("threshold"),
                                "entry": r.get("entry"),
                                "sl": r.get("sl"),
                                "tp": r.get("tp"),
                                "rr": r.get("rr"),
                            })
                    except Exception:
                        pass
                if _filtered:
                    pusher.push_rejected_batch(_filtered)
                    log.info(f"  Pushed {len(_filtered)} rejets au dashboard")

            # Cleanup _seen_obs vieux (> 2h)
            if cycle_n % 30 == 0:
                cutoff_2h = now - pd.Timedelta(hours=2)
                _seen_obs_copy = set()
                for k in _seen_obs:
                    try:
                        if pd.Timestamp(k[1]) >= cutoff_2h:
                            _seen_obs_copy.add(k)
                    except Exception:
                        _seen_obs_copy.add(k)
                _seen_obs.clear()
                _seen_obs.update(_seen_obs_copy)
                log.info(f"  Cleanup _seen_obs : {len(_seen_obs)} entries gardees")

            # Wait next minute boundary
            next_minute = (now + pd.Timedelta(minutes=1)).replace(second=0, microsecond=0)
            sleep_s = max(1, (next_minute - datetime.now(timezone.utc)).total_seconds())
            time.sleep(sleep_s)

    except KeyboardInterrupt:
        log.info("CTRL+C recu, shutdown propre")
    except Exception as e:
        log.exception(f"FATAL : {e}")
        pusher.push_error(f"Bot crash: {type(e).__name__}: {e}")
    finally:
        pusher.push_stop("Bot arrete")
        pusher.shutdown()
        mt5.shutdown()
        log.info("Bot V20 arrete proprement")


if __name__ == "__main__":
    main()
