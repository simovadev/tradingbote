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
from bot_v2.push_dashboard import DashboardPusher

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
    # Crypto (broker exige > 0.1% min)
    "BTCUSD":  0.0015, "ETHUSD":  0.0015, "LTCUSD":  0.002,
    "XRPUSD":  0.002,  "ADAUSD":  0.005,  "BCHUSD":  0.002,
    "DOTUSD":  0.008,  "LNKUSD":  0.015,  "SOLUSD":  0.015,
    # Metaux
    "XPDUSD":  0.005,  "XPTUSD":  0.005,
    "XAUJPY":  0.001,
    # Forex exotiques (besoin plus large)
    "USDZAR":  0.002,  "USDTRY":  0.005,  "USDMXN":  0.002,
    "USDNOK":  0.002,  "USDSEK":  0.002,  "USDDKK":  0.001,
    "USDPLN":  0.002,
    "EURHUF":  0.002,  "EURPLN":  0.002,
    "EURNOK":  0.002,  "EURSEK":  0.002,  "EURCZK":  0.002,
    # Indices europe/asia (souvent stops_level eleve)
    "GER40":   0.001,  "FRA40":   0.001,  "UK100":   0.001,
    "Nikkei225": 0.001,
    # Softs
    "Cotton-C": 0.005,
}

ASSETS = list(BROKER_MAP.keys())
THRESHOLD = float(os.environ["BOT_THRESHOLD"])
RECENT_CUTOFF_MIN = int(os.environ["RECENT_CUTOFF_MIN"])
BOT_MAGIC = 21000  # V21
MAX_CONCURRENT_TRADES = 5

_seen_obs: set[tuple] = set()
_boot_ts: pd.Timestamp | None = None
_ref_cache: dict[str, pd.DataFrame] = {}  # cache des df_m15 ref par cycle


def broker_sym(asset: str) -> str:
    return BROKER_MAP.get(asset, asset)


def fetch_ohlcv(asset: str, tf, n: int = 500) -> pd.DataFrame | None:
    rates = mt5.copy_rates_from_pos(broker_sym(asset), tf, 0, n)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
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
           "trades_taken": 0, "errors": []}
    try:
        df_m1 = fetch_ohlcv(asset, mt5.TIMEFRAME_M1, 500)
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

        for ob in obs_recent:
            ob_key = (asset, ob.validation_ts.isoformat(), ob.direction)
            if ob_key in _seen_obs:
                continue
            try:
                r = evaluate_ob(
                    ob, df_m1, df_m15, df_d1,
                    instrument=asset, df_htf2=df_h1, htf2_name="H1",
                )
                if r is None or r.verdict != "TRADE" or r.trade_setup is None:
                    reason = r.rejection_reason if r else "no_result"
                    _seen_obs.add(ob_key)
                    res["rejets"].append({"asset": asset, "ts": ob.validation_ts,
                                           "direction": ob.direction, "reason": reason or "no_trade"})
                    continue
                # V21 predict (avec cross-asset)
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
                    res["rejets"].append({"asset": asset, "ts": ob.validation_ts,
                                           "direction": ob.direction, "reason": "V21_unavailable"})
                    continue

                _seen_obs.add(ob_key)
                kz = killzone_at(ob.validation_ts) or "None"
                setup = r.trade_setup

                if proba >= THRESHOLD:
                    pusher.push_setup(
                        instrument=asset, ts=ob.validation_ts,
                        direction=ob.direction,
                        entry_price=setup.entry_price,
                        sl=setup.stop_loss, tp=setup.take_profit,
                        rr=setup.rr, score=r.score or 0,
                        ml_proba=proba, killzone=kz,
                    )
                    if execute_trade(mt5_exec, asset, setup, proba, balance, ob, kz, pusher):
                        res["trades_taken"] += 1
                    res["setups"].append({"asset": asset, "proba": proba})
                else:
                    res["rejets"].append({
                        "asset": asset, "ts": ob.validation_ts,
                        "direction": ob.direction,
                        "reason": f"ml_below_thr_{proba:.3f}",
                        "proba": proba, "threshold": THRESHOLD,
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
                   pusher: DashboardPusher) -> bool:
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
            return False
        tick_value = getattr(info, "trade_tick_value", 1.0)
        tick_size = getattr(info, "trade_tick_size", 0.01)
        tick = mt5.symbol_info_tick(bsym)
        if tick is None:
            return False
        market_price = tick.ask if direction == "bullish" else tick.bid
        sl_distance = abs(market_price - sl)
        if sl_distance <= 0:
            return False
        stops_level = getattr(info, "trade_stops_level", 0) * info.point
        min_pct = MIN_SL_PCT.get(asset)
        if min_pct:
            min_dist_pct = market_price * min_pct
            min_required = max(stops_level, min_dist_pct)
        else:
            min_required = stops_level
        if min_required > 0 and sl_distance < min_required:
            sl_extra = min_required - sl_distance
            if direction == "bullish":
                sl = sl - sl_extra
                tp = tp + sl_extra * rr
            else:
                sl = sl + sl_extra
                tp = tp - sl_extra * rr
            sl_distance = abs(market_price - sl)
            log.info(f"{asset} SL elargi a {sl_distance:.5f}")
        n_ticks = sl_distance / tick_size
        risk_per_lot = n_ticks * tick_value
        if risk_per_lot <= 0:
            return False
        lots = risk_eur / risk_per_lot
        lots = max(info.volume_min, round(lots / info.volume_step) * info.volume_step)
        lots = min(lots, info.volume_max)
        comment = f"V21-{direction[0].upper()} ml={proba:.2f}"
        result = mt5_exec.place_market_order(
            symbol=bsym, direction=direction, volume=lots,
            sl=sl, tp=tp, comment=comment, magic=BOT_MAGIC,
        )
        if result is None:
            log.warning(f"{asset} : ordre rejete par MT5")
            return False
        log.info(
            f"TRADE OK {asset} {direction} entry={result['price']:.5f} "
            f"SL={sl:.5f} TP={tp:.5f} vol={result['volume']:.2f} "
            f"rr={rr:.2f} ml={proba:.3f}"
        )
        pusher.push_trade_executed(
            instrument=asset, ticket=result["ticket"],
            direction=direction, entry=result["price"],
            sl=sl, tp=tp, volume=result["volume"],
            rr=rr, ml_proba=proba, score=0, killzone=killzone,
        )
        return True
    except Exception as e:
        log.exception(f"execute_trade fail {asset}: {e}")
        return False


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

    global _boot_ts
    boot_probe = mt5.copy_rates_from_pos(broker_sym("EURUSD"), mt5.TIMEFRAME_M1, 0, 1)
    if boot_probe is not None and len(boot_probe) > 0:
        _boot_ts = pd.to_datetime(boot_probe[0]["time"], unit="s", utc=True)
        log.info(f"BOOT_TS = {_boot_ts} (heure broker)")
    else:
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
            positions = mt5.positions_get() or []
            n_total_open = len(positions)
            account = mt5.account_info()
            balance = account.balance if account else 0

            log.info(f"--- CYCLE {cycle_n} | T={now.strftime('%H:%M:%S')} UTC | balance={balance:.2f} | open={n_total_open} ---")

            # Refresh cache des 5 refs
            refresh_ref_cache()

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

            pusher.push_cycle(
                actifs_scanned=len(ASSETS),
                total_s=elapsed,
                fetch_s=0, compute_s=elapsed,
                latencies={},
            )
            pusher.push_stats(
                total=cycle_n, wins=0, losses=0, wr_pct=0,
                pnl_total=0, balance=balance, equity=account.equity if account else 0,
                positions_open=n_total_open, pending_orders=0,
            )
            if all_rejets:
                pusher.push_rejected_batch(all_rejets[:50])
                log.info(f"  Pushed {min(len(all_rejets), 50)} rejets au dashboard")

            # Cleanup _seen_obs tous les 30 cycles (retention 2h)
            if cycle_n % 30 == 0:
                global _seen_obs
                _seen_obs.clear()
                log.info("  _seen_obs cleanup (cycle 30)")

            # Sleep jusqu'a la prochaine minute boundary
            next_minute = now.replace(second=0, microsecond=0) + pd.Timedelta(minutes=1)
            sleep_s = max(0.1, (next_minute - datetime.now(timezone.utc)).total_seconds())
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
