"""Backtest V12 ultra-logge pour DIAGNOSTIC.

Pour chaque setup detecte :
- Contexte HTF : trend H1/H4, killzone (asia/london/NY), heure UTC, jour semaine
- Features ML : les 59 features V12 telles que evaluees au moment du placement
- Prix au placement : prix marche au tick exact T (vs entry calcule par OB)
- Prix au fill : prix d'execution reel + delai entre placement et fill
- MFE/MAE : max profit/perte intra-trade en R (jusqu'a SL ou TP)
- Sequence tick : nb ticks bid touchant SL avant TP (et inverse)

Sortie : un seul CSV ultra-large bt_diagnostic_<date>_trades.csv pour analyse.

Usage :
    python backtest_v12_diagnostic.py --start 2026-05-12 --end 2026-05-16 \\
                                       --assets XAUUSD EURUSD
    python backtest_v12_diagnostic.py --start 2026-05-12 --end 2026-05-16  # 14 actifs

Tourne local (utilise data_vantage + data_ticks). Pour Vast, scp puis lancer la-bas.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault("SWS_OVERRIDE", "1")
os.environ.setdefault("RR_OVERRIDE", "1.5")
os.environ.setdefault("BUILD_DATA_DIR", "data_vantage")

ROOT = "c:/Users/Shadow/TradingBot" if sys.platform == "win32" else "/workspace/TradingBot"
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

import multiprocessing as _mp
try:
    _mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

LIVE_ASSETS = [
    "XAUUSD", "NAS100", "GER40", "BTCUSD",
    "EURUSD", "GBPUSD", "AUDUSD", "USDJPY",
    "SP500", "DJ30", "UK100", "FRA40",
    "USDCAD", "USDCHF",
]

CAP_AGE_OB_MIN = 30
EXPIRE_PENDING_MIN = 60
INITIAL_BALANCE = 150.0


def get_killzone(ts: pd.Timestamp) -> str:
    """Identifie la session de trading au moment ts (UTC)."""
    h = ts.hour
    if 0 <= h < 6:
        return "asia"
    elif 6 <= h < 12:
        return "london"
    elif 12 <= h < 16:
        return "ny_morning"
    elif 16 <= h < 21:
        return "ny_afternoon"
    else:
        return "off_hours"


def compute_htf_context(df_h1, df_h4, cur_ts) -> dict:
    """Calcule le contexte HTF au moment cur_ts (lookback dynamique).

    Returns dict avec :
      - h1_trend : "bullish" / "bearish" / "range"
      - h4_trend : idem
      - h1_close_above_ma20 : bool (proxy de tendance)
      - h4_close_above_ma20 : bool
      - h1_atr_5 : ATR des 5 dernieres H1 (volatilite)
      - dist_to_h1_swing_high : distance en % au dernier swing H1 high
      - dist_to_h1_swing_low : idem low
    """
    out = {}
    # H1
    h1_cut = df_h1[df_h1.index <= cur_ts]
    if len(h1_cut) >= 20:
        last_h1 = h1_cut.iloc[-1]
        ma20 = h1_cut['close'].iloc[-20:].mean()
        out['h1_close_above_ma20'] = bool(last_h1['close'] > ma20)
        # ATR 5 H1
        recent = h1_cut.iloc[-5:]
        out['h1_atr_5'] = float((recent['high'] - recent['low']).mean())
        # Simple trend : higher highs + higher lows sur les 5 dernieres H1
        highs = recent['high'].values
        lows = recent['low'].values
        if all(highs[i+1] >= highs[i] for i in range(len(highs)-1)) and all(lows[i+1] >= lows[i] for i in range(len(lows)-1)):
            out['h1_trend'] = "bullish"
        elif all(highs[i+1] <= highs[i] for i in range(len(highs)-1)) and all(lows[i+1] <= lows[i] for i in range(len(lows)-1)):
            out['h1_trend'] = "bearish"
        else:
            out['h1_trend'] = "range"
        # Distance au swing H1 (high/low des 20 dernieres H1)
        recent_20 = h1_cut.iloc[-20:]
        swing_high = recent_20['high'].max()
        swing_low = recent_20['low'].min()
        last_close = last_h1['close']
        out['dist_to_h1_swing_high_pct'] = float((swing_high - last_close) / last_close * 100)
        out['dist_to_h1_swing_low_pct'] = float((last_close - swing_low) / last_close * 100)
    else:
        out.update({'h1_close_above_ma20': None, 'h1_atr_5': None,
                    'h1_trend': None, 'dist_to_h1_swing_high_pct': None,
                    'dist_to_h1_swing_low_pct': None})

    # H4
    if df_h4 is not None and len(df_h4) > 0:
        h4_cut = df_h4[df_h4.index <= cur_ts]
        if len(h4_cut) >= 20:
            last_h4 = h4_cut.iloc[-1]
            ma20 = h4_cut['close'].iloc[-20:].mean()
            out['h4_close_above_ma20'] = bool(last_h4['close'] > ma20)
            recent = h4_cut.iloc[-5:]
            highs = recent['high'].values
            lows = recent['low'].values
            if all(highs[i+1] >= highs[i] for i in range(len(highs)-1)) and all(lows[i+1] >= lows[i] for i in range(len(lows)-1)):
                out['h4_trend'] = "bullish"
            elif all(highs[i+1] <= highs[i] for i in range(len(highs)-1)) and all(lows[i+1] <= lows[i] for i in range(len(lows)-1)):
                out['h4_trend'] = "bearish"
            else:
                out['h4_trend'] = "range"
        else:
            out['h4_close_above_ma20'] = None
            out['h4_trend'] = None
    else:
        out['h4_close_above_ma20'] = None
        out['h4_trend'] = None

    return out


def simulate_limit_with_tracking(rec, tick_times, tick_bid, tick_ask,
                                  commission_r=0.0,
                                  entry_mode="limit",
                                  apply_breakeven_at_R=None,
                                  latency_s=0.0):
    """Simulate fill + calcule MFE/MAE + premier tick a toucher SL/TP.

    FIX #4 : entry_mode="market" -> on entre au 1er tick apres placement
             au lieu d'attendre un retracement (LIMIT)
    FIX #2 : apply_breakeven_at_R=0.5 -> si MFE >= 0.5R, deplace SL a entry
             (= breakeven, plus de LOSS sur ce trade)
    """
    placed = pd.Timestamp(rec['placed_ts']).value + int(latency_s * 1_000_000_000)
    expire = placed + EXPIRE_PENDING_MIN * 60 * 1_000_000_000
    n = len(tick_times)
    direction = rec['direction']
    entry = rec['entry']
    sl_orig = rec['sl']
    tp = rec['tp']

    # 1. Trouver le 1er tick apres placement (prix de reference)
    ref_idx = None
    for i in range(n):
        if tick_times[i] > placed:
            ref_idx = i; break
    if ref_idx is None:
        rec['outcome'] = "NO_FILL"
        return

    # Prix au placement (= ref price)
    if direction == "bullish":
        ref_price = tick_ask[ref_idx]
        rec['price_at_placement'] = ref_price
        rec['entry_vs_ref_price_pct'] = (ref_price - entry) / entry * 100
    else:
        ref_price = tick_bid[ref_idx]
        rec['price_at_placement'] = ref_price
        rec['entry_vs_ref_price_pct'] = (entry - ref_price) / entry * 100

    # === FIX #4 : Mode MARKET — on entre tout de suite au prix marche ===
    if entry_mode == "market":
        fill_idx = ref_idx
        # Le prix d'entree devient le prix marche (pas l'entry calcule par OB)
        if direction == "bullish":
            actual_entry = tick_ask[ref_idx]
        else:
            actual_entry = tick_bid[ref_idx]
        # FIX #4b : recalcul TP pour preserver le RR du setup OB.
        # Sinon, en MARKET, le prix est deja engage vers le TP -> RR effectif tres faible
        # (median observe = 0.67 au lieu de 1.5) -> WIN ne paye pas les LOSS.
        # On garde le RR theorique du setup en deplacant le TP depuis actual_entry.
        risk_from_actual = abs(actual_entry - sl_orig)
        rr_target = rec.get("rr", 1.5)  # RR theorique du setup (= du OB)
        if direction == "bullish":
            tp = actual_entry + risk_from_actual * rr_target
        else:
            tp = actual_entry - risk_from_actual * rr_target
        rr_effective = rr_target  # par construction
        rec['actual_entry'] = actual_entry
        rec['rr_effective'] = rr_effective
        rec['tp'] = tp  # le TP utilise pour la simulation est le nouveau
    else:
        # === Mode LIMIT (original) ===
        # Check prix valide
        if direction == "bullish" and ref_price <= entry:
            rec['outcome'] = "INVALID_PRICE"
            return
        if direction == "bearish" and ref_price >= entry:
            rec['outcome'] = "INVALID_PRICE"
            return
        # Chercher le fill (retracement vers entry)
        fill_idx = None
        for i in range(ref_idx, n):
            if tick_times[i] > expire:
                rec['outcome'] = "NO_FILL"; return
            if direction == "bullish":
                if tick_ask[i] <= entry:
                    fill_idx = i; break
            else:
                if tick_bid[i] >= entry:
                    fill_idx = i; break
        if fill_idx is None:
            rec['outcome'] = "NO_FILL"
            return
        actual_entry = entry  # LIMIT fill exactement a entry

    rec['fill_ts'] = str(pd.Timestamp(tick_times[fill_idx], tz="UTC"))
    rec['time_to_fill_s'] = (tick_times[fill_idx] - placed) / 1e9
    rec['price_at_fill'] = actual_entry

    # === 3. Tracking MFE/MAE + outcome (avec SL dynamique via BE) ===
    risk = abs(actual_entry - sl_orig)
    if risk == 0:
        rec['outcome'] = "INVALID_PRICE"
        return
    # tp peut avoir ete recalcule en mode MARKET (FIX #4b)
    reward = abs(tp - actual_entry)
    rec['risk_abs'] = risk
    rec['reward_abs'] = reward

    sl_active = sl_orig  # SL dynamique (peut etre deplace par BE)
    be_threshold = (apply_breakeven_at_R * risk) if apply_breakeven_at_R else None
    be_triggered = False

    max_favor = 0.0
    max_adverse = 0.0
    ticks_before_outcome = 0
    final_outcome_idx = None
    final_outcome = None

    for i in range(fill_idx + 1, n):
        ticks_before_outcome += 1
        if direction == "bullish":
            cur_price = tick_bid[i]
            move_favor = cur_price - actual_entry
            move_adverse = actual_entry - cur_price
        else:
            cur_price = tick_ask[i]
            move_favor = actual_entry - cur_price
            move_adverse = cur_price - actual_entry
        if move_favor > max_favor:
            max_favor = move_favor
        if move_adverse > max_adverse:
            max_adverse = move_adverse

        # === FIX #2 : Breakeven a +X*R ===
        if be_threshold and not be_triggered and move_favor >= be_threshold:
            sl_active = actual_entry  # deplace SL a entry = breakeven
            be_triggered = True

        # check SL/TP avec sl_active (peut etre BE)
        if direction == "bullish":
            if tick_bid[i] <= sl_active:
                # Si BE active et touche entry = trade NEUTRE
                if be_triggered and sl_active == actual_entry:
                    final_outcome = "BE"; final_outcome_idx = i; break
                final_outcome = "LOSS"; final_outcome_idx = i; break
            if tick_bid[i] >= tp:
                final_outcome = "WIN"; final_outcome_idx = i; break
        else:
            if tick_ask[i] >= sl_active:
                if be_triggered and sl_active == actual_entry:
                    final_outcome = "BE"; final_outcome_idx = i; break
                final_outcome = "LOSS"; final_outcome_idx = i; break
            if tick_ask[i] <= tp:
                final_outcome = "WIN"; final_outcome_idx = i; break

    rec['mfe_R'] = max_favor / risk if risk > 0 else 0
    rec['mae_R'] = max_adverse / risk if risk > 0 else 0
    rec['ticks_count_in_trade'] = ticks_before_outcome
    rec['be_triggered'] = be_triggered

    if final_outcome is None:
        rec['outcome'] = "OPEN"
        return

    rec['outcome'] = final_outcome
    rec['exit_ts'] = str(pd.Timestamp(tick_times[final_outcome_idx], tz="UTC"))
    rec['hold_time_s'] = (tick_times[final_outcome_idx] - tick_times[fill_idx]) / 1e9
    if final_outcome == "WIN":
        # En mode MARKET, on utilise le RR effectif (depuis actual_entry), pas rr theorique
        rec['pnl_r'] = rec.get('rr_effective', rec['rr']) - commission_r
    elif final_outcome == "LOSS":
        rec['pnl_r'] = -1.0 - commission_r
    elif final_outcome == "BE":
        rec['pnl_r'] = 0.0 - commission_r  # BE = 0R (juste la commission)


def precompute_shared_cache(assets, start_date, end_date, output_dir):
    """Pre-calcule le cache complet pour chaque actif UNE SEULE FOIS.

    Les workers liront ensuite ce pickle au lieu de recalculer.
    Gain : N_actifs caches calcules au lieu de N_actifs * N_dates.

    Args:
        assets : liste des actifs
        start_date, end_date : dates limites (UTC)
        output_dir : dossier ou ecrire les pickles cache_<asset>.pkl
    """
    import pickle as _pickle
    from pathlib import Path
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, ROOT)
    from bot_v2.data_loader import load
    from bot_v2.concepts.order_block import detect_order_blocks as _dob
    from bot_v2.concepts.liquidity import find_swings as _fs
    from bot_v2.concepts.fvg import detect_fvg as _dfvg
    from bot_v2.concepts.breaker import detect_breakers as _dbrk
    from bot_v2.concepts.structure import detect_structure_breaks as _dsb
    from bot_v2.config import get_param as _gp

    start = pd.Timestamp(start_date, tz="UTC")
    end = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(hours=24)
    ws = start - pd.Timedelta(days=60)

    print(f"=== PRE-CALCUL CACHES PARTAGES ({len(assets)} actifs) ===", flush=True)
    t0 = time.time()
    for i, asset in enumerate(assets, 1):
        ta = time.time()
        try:
            df_m1 = load(asset, "M1", start=ws, end=end + pd.Timedelta(days=1))
            df_m15 = load(asset, "M15", start=ws, end=end + pd.Timedelta(days=1))
            df_h1 = load(asset, "H1", start=ws - pd.Timedelta(days=30), end=end + pd.Timedelta(days=1))
            sws = _gp(asset, "swing_strength_m1", 2)
            cache = {
                "obs": _dob(df_m1, swing_strength=sws),
                "swings_ltf": _fs(df_m1, strength=sws),
                "fvgs_ltf": _dfvg(df_m1),
                "breakers_ltf": _dbrk(df_m1),
                "obs_htf": _dob(df_m15),
                "obs_htf2": _dob(df_h1),
                "htf_trend": None,
            }
            cache["structure_breaks"] = _dsb(
                df_m1, swings=cache["swings_ltf"], fvgs=cache["fvgs_ltf"]
            )
            out = f"{output_dir}/cache_{asset}.pkl"
            with open(out, "wb") as f:
                _pickle.dump(cache, f, protocol=_pickle.HIGHEST_PROTOCOL)
            dt = time.time() - ta
            print(f"  [{i:2d}/{len(assets)}] {asset:8s} OK {dt:.1f}s  obs={len(cache['obs'])} fvg={len(cache['fvgs_ltf'])} brk={len(cache['breakers_ltf'])}", flush=True)
        except Exception as e:
            print(f"  [{i:2d}/{len(assets)}] {asset:8s} FAIL : {e}", flush=True)
    print(f"=== Pre-calcul fini en {time.time()-t0:.1f}s ===\n", flush=True)


def compute_atr_m5(df_m1, cut_ts, n=20):
    """Calcule l'ATR M5 sur les N dernieres bougies M5 avant cut_ts.
    On reagrege M1 -> M5 pour avoir un proxy ATR.
    """
    sub = df_m1[df_m1.index <= cut_ts].tail(n * 5)
    if len(sub) < 10:
        return None
    # Resample to M5
    m5 = sub.resample("5min").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    if len(m5) < 5:
        return None
    return float((m5["high"].tail(n) - m5["low"].tail(n)).mean())


def backtest_one(args_tuple):
    # Backward compat : (asset, date) -> defaults; (asset, date, options) -> dict d'options
    if len(args_tuple) == 2:
        asset, date_str = args_tuple
        opts = {}
    else:
        asset, date_str, opts = args_tuple
    apply_sl_min_atr = opts.get("apply_sl_min_atr", False)
    sl_min_atr_factor = opts.get("sl_min_atr_factor", 0.5)
    block_hours = set(opts.get("block_hours", []))
    entry_mode = opts.get("entry_mode", "limit")
    apply_breakeven_at_R = opts.get("apply_breakeven_at_R", None)
    latency_s = opts.get("latency_s", 0.0)
    commission_r = opts.get("commission_r", 0.0)
    step_min = opts.get("step_min", 5)
    # Cache partage : si fourni, on lit le pickle au lieu de recalculer
    shared_cache_path = opts.get("shared_cache_path", None)
    cache_end_date = opts.get("cache_end_date", None)  # date max couverte par le cache
    try:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["SWS_OVERRIDE"] = "1"
        os.environ["RR_OVERRIDE"] = "1.5"
        os.environ["BUILD_DATA_DIR"] = "data_vantage"
        sys.path.insert(0, ROOT)

        from bot_v2.data_loader import load
        from bot_v2.live_runner_v2 import compute_asset, load_model
        from bot_v2.concepts.daily_bias import build_d1_from_h1
        from bot_v2.concepts.order_block import detect_order_blocks as _dob
        from bot_v2.concepts.liquidity import find_swings as _fs
        from bot_v2.concepts.fvg import detect_fvg as _dfvg
        from bot_v2.concepts.breaker import detect_breakers as _dbrk
        from bot_v2.concepts.structure import detect_structure_breaks as _dsb
        from bot_v2.config import get_param as _gp
        from bot_v2 import ml_filter as _mlf
        from bot_v2.concepts.mss_setup import detect_mss_setups as _dmss
        from bot_v2.cache_filter import (
            filter_cache_for_cut as _filter_cache,
            shift_obs as _shift_obs,
            shift_swings as _shift_swings,
            shift_fvgs as _shift_fvgs,
            shift_breakers as _shift_breakers,
            shift_structure_breaks as _shift_sb,
        )

        day = pd.Timestamp(date_str, tz="UTC")
        start = day
        end = day + pd.Timedelta(hours=24)

        # Si cache partage : chargement bougies couvre TOUTES les dates (pas juste cette task)
        if shared_cache_path is not None and cache_end_date is not None:
            end_for_load = pd.Timestamp(cache_end_date, tz="UTC") + pd.Timedelta(hours=24)
        else:
            end_for_load = end
        ws = start - pd.Timedelta(days=60)
        df_m1 = load(asset, "M1", start=ws, end=end_for_load + pd.Timedelta(days=1))
        df_m15 = load(asset, "M15", start=ws, end=end_for_load + pd.Timedelta(days=1))
        df_h1 = load(asset, "H1", start=ws - pd.Timedelta(days=30), end=end_for_load + pd.Timedelta(days=1))
        try:
            df_h4 = load(asset, "H4", start=ws - pd.Timedelta(days=120), end=end_for_load + pd.Timedelta(days=1))
        except Exception:
            df_h4 = None
        try:
            df_d1 = load(asset, "D1", start=ws - pd.Timedelta(days=400), end=end_for_load + pd.Timedelta(days=1))
        except Exception:
            df_d1 = build_d1_from_h1(df_h1)

        tick_path = f"{ROOT}/data_ticks/{asset}_ticks_{date_str.replace('-','')}.parquet"
        if not os.path.exists(tick_path):
            return {"ok": False, "asset": asset, "date": date_str,
                    "error": f"ticks absents", "rows": []}
        tdf = pd.read_parquet(tick_path)
        tick_times = tdf["time_ns"].values.astype(np.int64)
        tick_bid = tdf["bid"].values.astype(float)
        tick_ask = tdf["ask"].values.astype(float)

        loaded = load_model(asset)
        if loaded is None:
            return {"ok": False, "asset": asset, "date": date_str,
                    "error": "no model", "rows": []}

        # === Cache complet : lecture partagee OU recalcul ===
        sws = _gp(asset, "swing_strength_m1", 2)
        if shared_cache_path is not None:
            # Lecture depuis pickle partage (1 calcul pour 14 actifs * N dates)
            import pickle as _pickle
            cache_pkl = f"{shared_cache_path}/cache_{asset}.pkl"
            if not os.path.exists(cache_pkl):
                return {"ok": False, "asset": asset, "date": date_str,
                        "error": f"cache pickle manquant: {cache_pkl}", "rows": []}
            with open(cache_pkl, "rb") as f:
                cache_full = _pickle.load(f)
        else:
            # Fallback : calcul direct dans le worker (comportement actuel)
            cache_full = {
                "obs": _dob(df_m1, swing_strength=sws),
                "swings_ltf": _fs(df_m1, strength=sws),
                "fvgs_ltf": _dfvg(df_m1),
                "breakers_ltf": _dbrk(df_m1),
                "obs_htf": _dob(df_m15),
                "obs_htf2": _dob(df_h1),
                "htf_trend": None,
            }
            cache_full["structure_breaks"] = _dsb(
                df_m1, swings=cache_full["swings_ltf"], fvgs=cache_full["fvgs_ltf"]
            )

        # === Boucle de scan (step minutes) ===
        all_rows = []
        evaluated = set()
        cur = start
        step = pd.Timedelta(minutes=step_min)
        total_cycles = int((end - start).total_seconds() / 60 / step_min)
        log_every = max(50, total_cycles // 5)  # 5 logs intermediaires par task
        cycle_idx = 0
        n_setups_so_far = 0
        _t_task_start = time.time()
        while cur <= end:
            cycle_idx += 1
            cut = cur - pd.Timedelta(minutes=1)
            ie = df_m1.index.searchsorted(cut, side="right")
            sub_start = max(0, ie - 88000)
            sub_m1 = df_m1.iloc[sub_start:ie]
            if len(sub_m1) < 200:
                cur += step; continue
            i15 = df_m15.index.searchsorted(cut, side="right")
            i15_start = max(0, i15 - 11000)
            i1 = df_h1.index.searchsorted(cut, side="right")
            i1_start = max(0, i1 - 2800)
            id1 = df_d1.index.searchsorted(cut, side="right")

            # Cache filtre + shift
            _cache_at_cut = _filter_cache(
                cache_full, cut_iloc_m1=ie, cut_iloc_m15=i15, cut_iloc_h1=i1,
                df_m1_cut=df_m1.iloc[:ie],
            )
            _precomputed = {
                "obs": _shift_obs(_cache_at_cut["obs"], sub_start),
                "swings_ltf": _shift_swings(_cache_at_cut["swings_ltf"], sub_start),
                "fvgs_ltf": _shift_fvgs(_cache_at_cut["fvgs_ltf"], sub_start),
                "breakers_ltf": _shift_breakers(_cache_at_cut["breakers_ltf"], sub_start),
                "structure_breaks": _shift_sb(_cache_at_cut["structure_breaks"], sub_start),
                "obs_htf": _shift_obs(_cache_at_cut["obs_htf"], i15_start),
                "obs_htf2": _shift_obs(_cache_at_cut["obs_htf2"], i1_start),
                "htf_trend": _cache_at_cut["htf_trend"],
            }

            payload = {
                "instrument": asset, "df_m1": sub_m1,
                "df_m15": df_m15.iloc[i15_start:i15],
                "df_h1": df_h1.iloc[i1_start:i1],
                "df_h4": (df_h4.iloc[:df_h4.index.searchsorted(cut, side="right")][-500:]
                          if df_h4 is not None else None),
                "df_d1": df_d1.iloc[max(0, id1 - 120):id1],
                "correlated_dfs": {}, "balance": INITIAL_BALANCE, "debug_diag": False,
                "precomputed_cache": _precomputed,
            }
            try:
                res = compute_asset(payload)
            except Exception as e:
                cur += step; continue

            for s in res.get("setups", []):
                ob = s["ob"]; ob_ts = s["ts"]; proba = s.get("proba", 0)
                key = (str(ob_ts), ob.direction)
                if key in evaluated:
                    continue
                age_min = (cur - ob_ts).total_seconds() / 60
                if age_min > CAP_AGE_OB_MIN:
                    continue

                # FIX #3 : Filtre heures pourries (00h, 19-21h UTC)
                if cur.hour in block_hours:
                    evaluated.add(key)
                    continue

                t2 = s["r"].trade_setup

                # FIX #1 : SL minimum base sur ATR M5
                #   Si le SL naturel de l'OB est < sl_min_atr_factor * ATR_M5,
                #   on l'elargit (et on recalcule TP pour garder le meme RR).
                sl_adjusted = t2.stop_loss
                tp_adjusted = t2.take_profit
                rr_adjusted = t2.rr
                sl_was_widened = False
                if apply_sl_min_atr:
                    atr_m5 = compute_atr_m5(df_m1, cur, n=20)
                    if atr_m5 is not None:
                        sl_dist_natural = abs(t2.entry_price - t2.stop_loss)
                        sl_dist_min = sl_min_atr_factor * atr_m5
                        if sl_dist_natural < sl_dist_min:
                            # Elargir SL et TP pour garder le meme RR
                            if ob.direction == "bullish":
                                sl_adjusted = t2.entry_price - sl_dist_min
                                tp_adjusted = t2.entry_price + sl_dist_min * t2.rr
                            else:
                                sl_adjusted = t2.entry_price + sl_dist_min
                                tp_adjusted = t2.entry_price - sl_dist_min * t2.rr
                            sl_was_widened = True

                # Capture le contexte HTF
                htf_ctx = compute_htf_context(df_h1, df_h4, cur)
                # Recalcul des features ML (compute_asset ne les retourne pas)
                features = {}
                try:
                    mss_setups = _dmss(
                        sub_m1,
                        structure_breaks=_precomputed["structure_breaks"],
                        swings=_precomputed["swings_ltf"],
                        fvgs=_precomputed["fvgs_ltf"],
                    )
                    features = _mlf._features_from_result(
                        s["r"], ob, asset,
                        df_ltf=sub_m1, df_d1=df_d1.iloc[max(0, id1 - 120):id1],
                        mss_setups=mss_setups,
                    )
                except Exception:
                    features = {}
                row = {
                    "instrument": asset, "date": date_str,
                    "direction": ob.direction,
                    "ob_ts": str(ob_ts),
                    "placed_ts": str(cur),
                    "age_at_placement_min": age_min,
                    "hour_utc": cur.hour,
                    "weekday": cur.day_name(),
                    "killzone": get_killzone(cur),
                    "ml_proba": proba,
                    "entry": t2.entry_price,
                    "sl": sl_adjusted,    # FIX #1 applique
                    "tp": tp_adjusted,
                    "rr": rr_adjusted,
                    "sl_original": t2.stop_loss,
                    "tp_original": t2.take_profit,
                    "sl_was_widened": sl_was_widened,
                    "ob_high": ob.ob_high, "ob_low": ob.ob_low,
                    "ob_size_pct": abs(ob.ob_high - ob.ob_low) / t2.entry_price * 100,
                    # HTF context
                    **{f"ctx_{k}": v for k, v in htf_ctx.items()},
                    # ML features (les top key features)
                    "features_json": json.dumps({k: float(v) if isinstance(v, (int, float, np.number)) else str(v)
                                                 for k, v in features.items()})[:5000],  # limit pour CSV
                }
                # Simulate fill + tracking (avec FIX #2 et FIX #4)
                simulate_limit_with_tracking(
                    row, tick_times, tick_bid, tick_ask,
                    commission_r=commission_r,
                    entry_mode=entry_mode,
                    apply_breakeven_at_R=apply_breakeven_at_R,
                    latency_s=latency_s,
                )
                all_rows.append(row)
                evaluated.add(key)
                n_setups_so_far += 1
            cur += step

            # Log progression intra-task toutes les N cycles
            if cycle_idx % log_every == 0:
                pct = cycle_idx / total_cycles * 100
                dt = time.time() - _t_task_start
                eta = dt / cycle_idx * (total_cycles - cycle_idx) if cycle_idx > 0 else 0
                print(f"  [{asset} {date_str}] {pct:.0f}%  cycle={cycle_idx}/{total_cycles}  "
                      f"setups={n_setups_so_far}  elapsed={dt:.0f}s  ETA={eta:.0f}s", flush=True)

        return {"ok": True, "asset": asset, "date": date_str, "rows": all_rows}
    except Exception as e:
        import traceback
        return {"ok": False, "asset": asset, "date": date_str,
                "error": f"{e}\n{traceback.format_exc()[:500]}", "rows": []}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD (inclusif)")
    p.add_argument("--assets", nargs="+", default=LIVE_ASSETS)
    p.add_argument("--workers", type=int, default=14)
    p.add_argument("--output", default=f"{ROOT}/bt_diagnostic_trades.csv")
    # === FIXES ===
    p.add_argument("--entry_mode", default="limit", choices=["limit", "market"],
                   help="FIX #4 : 'market' entre immediatement au prix marche")
    p.add_argument("--sl_min_atr", action="store_true",
                   help="FIX #1 : force SL minimum = sl_min_atr_factor * ATR_M5")
    p.add_argument("--sl_min_atr_factor", type=float, default=0.5,
                   help="Facteur multiplicateur ATR (default 0.5)")
    p.add_argument("--breakeven_at_R", type=float, default=None,
                   help="FIX #2 : deplace SL a entry quand MFE atteint X*R (ex: 0.5)")
    p.add_argument("--block_hours", nargs="+", type=int, default=[],
                   help="FIX #3 : heures UTC a bloquer (ex: 0 1 19 20 21)")
    p.add_argument("--latency_s", type=float, default=0.0,
                   help="Latence simulee en s entre placement et execution")
    p.add_argument("--commission_r", type=float, default=0.0,
                   help="Commission par trade en R (ex: 0.07 = 7% du risque)")
    p.add_argument("--step_min", type=int, default=5,
                   help="Pas de scan en minutes (default 5, mais 1 reproduit le live sync close M1)")
    p.add_argument("--shared_cache", action="store_true",
                   help="Pre-calcule le cache UNE FOIS par actif (gain ~3-5x). Workers lisent pickle.")
    p.add_argument("--shared_cache_dir", default="/tmp/bt_shared_cache",
                   help="Dossier ou ecrire les caches partages (default /tmp/bt_shared_cache)")
    args = p.parse_args()

    dates = []
    d = pd.Timestamp(args.start)
    end_d = pd.Timestamp(args.end)
    while d <= end_d:
        if d.weekday() < 5:  # skip weekends
            dates.append(d.strftime("%Y-%m-%d"))
        d += pd.Timedelta(days=1)

    opts = {
        "apply_sl_min_atr": args.sl_min_atr,
        "sl_min_atr_factor": args.sl_min_atr_factor,
        "block_hours": args.block_hours,
        "entry_mode": args.entry_mode,
        "apply_breakeven_at_R": args.breakeven_at_R,
        "latency_s": args.latency_s,
        "commission_r": args.commission_r,
        "step_min": args.step_min,
    }

    # === Cache partage : pre-calcul UNE FOIS par actif ===
    if args.shared_cache:
        precompute_shared_cache(args.assets, args.start, args.end, args.shared_cache_dir)
        opts["shared_cache_path"] = args.shared_cache_dir
        opts["cache_end_date"] = args.end

    print(f"=== BACKTEST DIAGNOSTIC ULTRA-LOGGE ===")
    print(f"Periode    : {args.start} -> {args.end} ({len(dates)} jours ouvres)")
    print(f"Actifs     : {args.assets} ({len(args.assets)})")
    print(f"Tasks      : {len(dates) * len(args.assets)}")
    print(f"Workers    : {args.workers}")
    print(f"--- FIXES ---")
    print(f"  entry_mode      : {args.entry_mode}  (FIX #4)")
    print(f"  sl_min_atr      : {args.sl_min_atr} (factor={args.sl_min_atr_factor})  (FIX #1)")
    print(f"  breakeven_at_R  : {args.breakeven_at_R}  (FIX #2)")
    print(f"  block_hours     : {args.block_hours}  (FIX #3)")
    print(f"  step_min        : {args.step_min}min  (FIX #5)")
    print(f"  shared_cache    : {args.shared_cache}  (perf)")
    print(f"  latency_s       : {args.latency_s}")
    print(f"  commission_r    : {args.commission_r}")
    print()

    tasks = [(a, d, opts) for a in args.assets for d in dates]
    all_rows = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(backtest_one, t): t for t in tasks}
        done = 0
        for fut in as_completed(futs):
            done += 1
            r = fut.result()
            if r.get("ok"):
                all_rows.extend(r["rows"])
                n_set = len(r["rows"])
                n_close = sum(1 for x in r["rows"] if x.get("outcome") in ("WIN", "LOSS"))
                print(f"[{done:3d}/{len(tasks)}] {r['asset']:8s} {r['date']} setups={n_set:3d} fermes={n_close:3d}", flush=True)
            else:
                print(f"[{done:3d}/{len(tasks)}] FAIL {r['asset']} {r['date']} : {r.get('error','')[:100]}", flush=True)

    elapsed = time.time() - t0
    print(f"\n=== TERMINE en {elapsed/60:.1f}min ===\n")

    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(args.output, index=False)
        print(f"Sauve : {args.output} ({len(df)} lignes)")
        print()
        # Stats rapides
        print("=== STATS RAPIDES ===")
        for o in ["WIN", "LOSS", "BE", "NO_FILL", "INVALID_PRICE", "OPEN"]:
            n = (df["outcome"] == o).sum()
            print(f"  {o:14}: {n:4d}")
        fermes = df[df["outcome"].isin(["WIN", "LOSS", "BE"])]
        if len(fermes) > 0:
            wr_strict = (fermes["outcome"] == "WIN").sum() / max((fermes["outcome"].isin(["WIN", "LOSS"])).sum(), 1) * 100
            pnl = fermes["pnl_r"].sum()
            print(f"\n  Fermes (incl BE) : {len(fermes)}")
            print(f"  WR strict (W/W+L): {wr_strict:.1f}%")
            print(f"  PnL (R)          : {pnl:+.2f}")
            if "sl_was_widened" in df.columns:
                n_widened = df["sl_was_widened"].sum()
                print(f"  SL elargis (FIX#1): {n_widened}")
            if "be_triggered" in df.columns:
                n_be_trig = df["be_triggered"].sum() if df["be_triggered"].dtype != "object" else 0
                print(f"  BE triggers (FIX#2): {n_be_trig}")


if __name__ == "__main__":
    main()
