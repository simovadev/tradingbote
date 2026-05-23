"""Backtest V12 TICK PAR TICK - version VAST (sans MT5).

Lit :
- les bougies depuis data_vantage/<ASSET>_<TF>.parquet (detection)
- les ticks depuis data_ticks/<ASSET>_ticks_<YYYYMMDD>.parquet (fill/SL/TP exact)

Detection sur M1 (= live), fill+exit sur vrais ticks bid/ask.
Parallelise par actif pour saturer Vast (regle saturation).

Usage Vast :
    cd /workspace/TradingBot
    nohup python3 -u backtest_v12_tick_vast.py --date 2026-05-19 \
        --workers 14 > /workspace/bt_tick.log 2>&1 & disown
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault("SWS_OVERRIDE", "1")
os.environ.setdefault("RR_OVERRIDE", "1.5")
os.environ.setdefault("BUILD_DATA_DIR", "data_vantage")

ROOT = "c:/Users/Shadow/TradingBot" if sys.platform == "win32" else "/workspace/TradingBot"
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

import multiprocessing as _mp
_mp.set_start_method("spawn", force=True)

LIVE_ASSETS = [
    "XAUUSD", "NAS100", "GER40", "BTCUSD",
    "EURUSD", "GBPUSD", "AUDUSD", "USDJPY",
    "SP500", "DJ30", "UK100", "FRA40",
    "USDCAD", "USDCHF",
]

CAP_AGE_OB_MIN = 30
EXPIRE_PENDING_MIN = 60
INITIAL_BALANCE = 150.0


@dataclass
class TradeRec:
    instrument: str
    direction: str
    ob_ts: str
    placed_ts: str
    ml: float
    entry: float
    sl: float
    tp: float
    rr: float
    fill_ts: str = ""
    exit_ts: str = ""
    outcome: str = "PENDING"
    pnl_r: float = 0.0


def simulate_on_ticks(rec, tick_times, tick_bid, tick_ask, commission_r=0.0):
    placed = pd.Timestamp(rec.placed_ts).value
    expire = placed + EXPIRE_PENDING_MIN * 60 * 1_000_000_000
    n = len(tick_times)

    # --- Reproduit la VRAIE logique d'un ordre LIMIT MT5 ---
    # SELL LIMIT : doit etre place AU-DESSUS du prix courant, se remplit quand
    #   le prix REMONTE toucher entry. Si au placement le prix est deja >= entry
    #   -> MT5 rejette "Invalid price" (le pending n'aurait pas existe).
    # BUY LIMIT : doit etre place EN DESSOUS, se remplit quand le prix DESCEND.
    # On trouve le 1er tick apres placement = prix de reference.
    ref_idx = None
    for i in range(n):
        if tick_times[i] > placed:
            ref_idx = i
            break
    if ref_idx is None:
        rec.outcome = "NO_FILL"; return

    # Prix de reference au placement
    if rec.direction == "bullish":
        ref_price = tick_ask[ref_idx]   # on achetera au ask
        # BUY LIMIT valide seulement si prix de ref AU-DESSUS de l'entry
        # (sinon le prix est deja en dessous -> fill instantane = ordre invalide)
        if ref_price <= rec.entry:
            rec.outcome = "INVALID_PRICE"; return
    else:
        ref_price = tick_bid[ref_idx]   # on vendra au bid
        # SELL LIMIT valide seulement si prix de ref EN DESSOUS de l'entry
        if ref_price >= rec.entry:
            rec.outcome = "INVALID_PRICE"; return

    # Cherche le fill : le prix doit REVENIR toucher entry (vrai retracement)
    fill_idx = None
    for i in range(ref_idx, n):
        t = tick_times[i]
        if t > expire:
            rec.outcome = "NO_FILL"; return
        if rec.direction == "bullish":
            if tick_ask[i] <= rec.entry:
                fill_idx = i; break
        else:
            if tick_bid[i] >= rec.entry:
                fill_idx = i; break
    if fill_idx is None:
        rec.outcome = "NO_FILL"; return
    rec.fill_ts = str(pd.Timestamp(tick_times[fill_idx], tz="UTC"))
    for i in range(fill_idx + 1, n):
        if rec.direction == "bullish":
            if tick_bid[i] <= rec.sl:
                rec.outcome = "LOSS"; rec.pnl_r = -1.0 - commission_r
                rec.exit_ts = str(pd.Timestamp(tick_times[i], tz="UTC")); return
            if tick_bid[i] >= rec.tp:
                rec.outcome = "WIN"; rec.pnl_r = rec.rr - commission_r
                rec.exit_ts = str(pd.Timestamp(tick_times[i], tz="UTC")); return
        else:
            if tick_ask[i] >= rec.sl:
                rec.outcome = "LOSS"; rec.pnl_r = -1.0 - commission_r
                rec.exit_ts = str(pd.Timestamp(tick_times[i], tz="UTC")); return
            if tick_ask[i] <= rec.tp:
                rec.outcome = "WIN"; rec.pnl_r = rec.rr - commission_r
                rec.exit_ts = str(pd.Timestamp(tick_times[i], tz="UTC")); return
    rec.outcome = "OPEN"


def simulate_market_on_ticks(rec, tick_times, tick_bid, tick_ask,
                             latency_s=0, commission_r=0.0):
    """Entree MARKET : on entre AU PRIX MARCHE des le placement (pas de retracement).
    SL/TP gardent leurs niveaux absolus (= ceux de l'OB). Le PnL en R est
    recalcule depuis le VRAI prix d'entree marche.

    latency_s : delai (s) entre decision et execution -> on entre au prix
                qu'il y avait latency_s APRES le placement (realisme VPS/MT5).
    commission_r : cout en R par trade (ex: 0.07 = 7% du risque). Deduit du PnL.
    """
    placed = pd.Timestamp(rec.placed_ts).value + int(latency_s * 1_000_000_000)
    n = len(tick_times)
    ref_idx = None
    for i in range(n):
        if tick_times[i] > placed:
            ref_idx = i; break
    if ref_idx is None:
        rec.outcome = "NO_FILL"; return

    # Entree au marche : bullish -> ask, bearish -> bid
    if rec.direction == "bullish":
        entry_mkt = tick_ask[ref_idx]
    else:
        entry_mkt = tick_bid[ref_idx]
    rec.fill_ts = str(pd.Timestamp(tick_times[ref_idx], tz="UTC"))
    rec.entry = entry_mkt  # on note le vrai prix d'entree

    # Risk reel = distance entree marche -> SL. Reward = distance -> TP.
    risk = abs(entry_mkt - rec.sl)
    if risk <= 0:
        rec.outcome = "INVALID_PRICE"; return
    reward = abs(rec.tp - entry_mkt)
    rec.rr = reward / risk  # RR effectif

    # Si l'entree marche est DEJA au-dela du SL ou du TP -> trade degenere
    if rec.direction == "bullish":
        if entry_mkt >= rec.tp:
            rec.outcome = "INVALID_PRICE"; return  # deja au TP, rien a gagner
        if entry_mkt <= rec.sl:
            rec.outcome = "INVALID_PRICE"; return  # deja au SL
    else:
        if entry_mkt <= rec.tp:
            rec.outcome = "INVALID_PRICE"; return
        if entry_mkt >= rec.sl:
            rec.outcome = "INVALID_PRICE"; return

    # Suit les ticks pour SL/TP (commission deduite du PnL R)
    for i in range(ref_idx + 1, n):
        if rec.direction == "bullish":
            if tick_bid[i] <= rec.sl:
                rec.outcome = "LOSS"; rec.pnl_r = -1.0 - commission_r
                rec.exit_ts = str(pd.Timestamp(tick_times[i], tz="UTC")); return
            if tick_bid[i] >= rec.tp:
                rec.outcome = "WIN"; rec.pnl_r = rec.rr - commission_r
                rec.exit_ts = str(pd.Timestamp(tick_times[i], tz="UTC")); return
        else:
            if tick_ask[i] >= rec.sl:
                rec.outcome = "LOSS"; rec.pnl_r = -1.0 - commission_r
                rec.exit_ts = str(pd.Timestamp(tick_times[i], tz="UTC")); return
            if tick_ask[i] <= rec.tp:
                rec.outcome = "WIN"; rec.pnl_r = rec.rr - commission_r
                rec.exit_ts = str(pd.Timestamp(tick_times[i], tz="UTC")); return
    rec.outcome = "OPEN"


def backtest_asset(args_tuple):
    # args : (asset, date_str, step, scan_start_str, scan_end_str, entry_mode)
    # scan_start/end = fenetre de DETECTION (pour paralleliser). Les ticks
    # couvrent toute la journee donc les trades ne sont jamais coupes.
    # entry_mode : "limit" (defaut, retracement) ou "market" (entree immediate)
    # Tuple : (asset, date, step, scan_start, scan_end, entry_mode, latency, commission, no_cache_optim?)
    if len(args_tuple) == 9:
        asset, date_str, step, scan_start_str, scan_end_str, entry_mode, latency_s, commission_r, no_cache_optim = args_tuple
    else:
        asset, date_str, step, scan_start_str, scan_end_str, entry_mode, latency_s, commission_r = args_tuple
        no_cache_optim = False
    try:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["SWS_OVERRIDE"] = "1"
        os.environ["RR_OVERRIDE"] = "1.5"
        os.environ["BUILD_DATA_DIR"] = "data_vantage"
        import sys as _sys
        _sys.path.insert(0, ROOT)
        import pandas as _pd
        import numpy as _np
        from bot_v2.data_loader import load
        from bot_v2.live_runner_v2 import compute_asset, load_model
        from bot_v2.concepts.daily_bias import build_d1_from_h1
        from bot_v2.concepts.order_block import detect_order_blocks as _dob
        from bot_v2.concepts.liquidity import find_swings as _fs
        from bot_v2.concepts.fvg import detect_fvg as _dfvg
        from bot_v2.concepts.breaker import detect_breakers as _dbrk
        from bot_v2.concepts.structure import detect_structure_breaks as _dsb, detect_trend as _dtr
        from bot_v2.config import get_param as _gp
        from bot_v2.cache_filter import (
            filter_cache_for_cut as _filter_cache,
            shift_obs as _shift_obs,
            shift_swings as _shift_swings,
            shift_fvgs as _shift_fvgs,
            shift_breakers as _shift_breakers,
            shift_structure_breaks as _shift_sb,
        )

        day = _pd.Timestamp(date_str, tz="UTC")
        # Fenetre de detection (sous-periode pour parallelisme)
        start = _pd.Timestamp(scan_start_str, tz="UTC")
        end = _pd.Timestamp(scan_end_str, tz="UTC")

        # Bougies depuis data_vantage (avec buffer 60j amont)
        ws = start - _pd.Timedelta(days=60)
        df_m1 = load(asset, "M1", start=ws, end=end + _pd.Timedelta(days=1))
        df_m15 = load(asset, "M15", start=ws, end=end + _pd.Timedelta(days=1))
        df_h1 = load(asset, "H1", start=ws - _pd.Timedelta(days=30), end=end + _pd.Timedelta(days=1))
        try:
            df_h4 = load(asset, "H4", start=ws - _pd.Timedelta(days=120), end=end + _pd.Timedelta(days=1))
        except Exception:
            df_h4 = None
        try:
            df_d1 = load(asset, "D1", start=ws - _pd.Timedelta(days=400), end=end + _pd.Timedelta(days=1))
        except Exception:
            df_d1 = build_d1_from_h1(df_h1)

        # Ticks depuis data_ticks
        tick_path = f"{ROOT}/data_ticks/{asset}_ticks_{date_str.replace('-','')}.parquet"
        if not os.path.exists(tick_path):
            return {"ok": False, "asset": asset, "error": f"ticks absents {tick_path}"}
        tdf = _pd.read_parquet(tick_path)
        tick_times = tdf["time_ns"].values.astype(_np.int64)
        tick_bid = tdf["bid"].values.astype(float)
        tick_ask = tdf["ask"].values.astype(float)

        loaded = load_model(asset)
        if loaded is None:
            return {"ok": False, "asset": asset, "error": "pas de modele"}

        # === OPTIM 2026-05-23 : pre-calcul du cache complet UNE SEULE FOIS ===
        # Au lieu de recalculer obs/swings/fvgs/breakers/sb 288 fois (1/cycle),
        # on les calcule UNE fois sur df_m1 complet, puis on filtre par cycle.
        # Equivalence stricte prouvee par test_cache_filter_equivalence.py.
        # Si no_cache_optim=True : on n'utilise pas l'optim (comportement original).
        cache_full = None
        if not no_cache_optim:
            _sws = _gp(asset, "swing_strength_m1", 2)
            _all_obs = _dob(df_m1, swing_strength=_sws)
            _all_swings = _fs(df_m1, strength=_sws)
            _all_fvgs = _dfvg(df_m1)
            _all_breakers = _dbrk(df_m1)
            _all_sb = _dsb(df_m1, swings=_all_swings, fvgs=_all_fvgs)
            _all_obs_m15 = _dob(df_m15)
            _all_obs_h1 = _dob(df_h1)
            cache_full = {
                "obs": _all_obs, "swings_ltf": _all_swings, "fvgs_ltf": _all_fvgs,
                "breakers_ltf": _all_breakers, "structure_breaks": _all_sb,
                "obs_htf": _all_obs_m15, "obs_htf2": _all_obs_h1, "htf_trend": None,
            }

        # Scan boucle (detection M1)
        active = []
        evaluated = set()
        cur = start
        n_setups = 0
        while cur <= end:
            cut = cur - _pd.Timedelta(minutes=1)
            ie = df_m1.index.searchsorted(cut, side="right")
            sub_start = max(0, ie - 88000)
            sub_m1 = df_m1.iloc[sub_start:ie]
            if len(sub_m1) < 200:
                cur += _pd.Timedelta(minutes=step); continue
            i15 = df_m15.index.searchsorted(cut, side="right")
            i15_start = max(0, i15 - 11000)
            i1 = df_h1.index.searchsorted(cut, side="right")
            i1_start = max(0, i1 - 2800)
            id1 = df_d1.index.searchsorted(cut, side="right")

            _precomputed = None
            if cache_full is not None:
                # Filtre + shift du cache pour ce cycle.
                # 1. Filtre dans le referentiel "df_m1 complet" (cut_iloc=ie).
                _cache_at_cut = _filter_cache(
                    cache_full,
                    cut_iloc_m1=ie,
                    cut_iloc_m15=i15,
                    cut_iloc_h1=i1,
                    df_m1_cut=df_m1.iloc[:ie],
                )
                # 2. Shift dans le referentiel "sub_m1" (offsets sub_start, i15_start, i1_start).
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
                "df_h4": (df_h4.iloc[:df_h4.index.searchsorted(cut, side="right")][-500:] if df_h4 is not None else None),
                "df_d1": df_d1.iloc[max(0, id1 - 120):id1],
                "correlated_dfs": {}, "balance": INITIAL_BALANCE, "debug_diag": False,
                "precomputed_cache": _precomputed,
            }
            try:
                res = compute_asset(payload)
            except Exception:
                cur += _pd.Timedelta(minutes=step); continue
            for s in res.get("setups", []):
                ob = s["ob"]; ob_ts = s["ts"]; proba = s.get("proba", 0)
                key = (str(ob_ts), ob.direction)
                if key in evaluated:
                    continue
                if (cur - ob_ts).total_seconds() / 60 > CAP_AGE_OB_MIN:
                    continue
                t2 = s["r"].trade_setup
                active.append(TradeRec(
                    instrument=asset, direction=ob.direction, ob_ts=str(ob_ts),
                    placed_ts=str(cur), ml=proba,
                    entry=t2.entry_price, sl=t2.stop_loss, tp=t2.take_profit, rr=t2.rr))
                evaluated.add(key)
                n_setups += 1
            cur += _pd.Timedelta(minutes=step)

        for rec in active:
            if entry_mode == "market":
                simulate_market_on_ticks(rec, tick_times, tick_bid, tick_ask,
                                         latency_s=latency_s, commission_r=commission_r)
            else:
                simulate_on_ticks(rec, tick_times, tick_bid, tick_ask,
                                  commission_r=commission_r)

        closed = [t for t in active if t.outcome in ("WIN", "LOSS")]
        wr = (sum(1 for t in closed if t.outcome == "WIN") / len(closed) * 100) if closed else 0
        return {
            "ok": True, "asset": asset, "n_setups": n_setups,
            "n_closed": len(closed),
            "n_win": sum(1 for t in closed if t.outcome == "WIN"),
            "wr": wr, "pnl_r": sum(t.pnl_r for t in closed),
            "n_nofill": sum(1 for t in active if t.outcome == "NO_FILL"),
            "n_open": sum(1 for t in active if t.outcome == "OPEN"),
            "trades": [vars(t) for t in active],
        }
    except Exception as e:
        import traceback
        return {"ok": False, "asset": asset, "error": str(e)[:200],
                "trace": traceback.format_exc()[-500:]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--date", required=True, help="YYYY-MM-DD")
    p.add_argument("--assets", nargs="+", default=LIVE_ASSETS)
    p.add_argument("--step", type=int, default=5)
    p.add_argument("--workers", type=int, default=128)
    p.add_argument("--scan_hours", type=int, default=2,
                   help="Decoupe la detection en fenetres de X heures (parallelisme)")
    p.add_argument("--entry_mode", default="limit", choices=["limit", "market"],
                   help="limit (retracement) ou market (entree immediate a la validation)")
    p.add_argument("--latency_s", type=float, default=0.0,
                   help="Latence en s entre decision et execution (realisme VPS/MT5). Default 0.")
    p.add_argument("--commission_r", type=float, default=0.0,
                   help="Commission par trade en R (ex: 0.07 = 7%% du risque). Default 0.")
    p.add_argument("--no_cache_optim", action="store_true",
                   help="Desactive le cache_filter (recalcul a chaque cycle, comme avant). Utile pour valider l'equivalence.")
    args = p.parse_args()

    day = pd.Timestamp(args.date, tz="UTC")
    # Decoupe la journee en fenetres de detection de scan_hours
    windows = []
    cur = day
    day_end = day + pd.Timedelta(hours=24)
    while cur < day_end:
        nxt = min(cur + pd.Timedelta(hours=args.scan_hours), day_end)
        windows.append((str(cur), str(nxt)))
        cur = nxt

    print(f"=== BACKTEST TICK PAR TICK (VAST) ===", flush=True)
    print(f"Date          : {args.date}", flush=True)
    print(f"Actifs        : {len(args.assets)}", flush=True)
    print(f"Fenetres scan : {len(windows)} x {args.scan_hours}h", flush=True)
    print(f"Tasks         : {len(args.assets) * len(windows)}", flush=True)
    print(f"Workers       : {args.workers}", flush=True)
    print(f"Entry mode    : {args.entry_mode.upper()}", flush=True)
    print(f"Latence       : {args.latency_s}s", flush=True)
    print(f"Commission/R  : {args.commission_r:.3f} (= {args.commission_r*100:.1f}% du risque par trade)", flush=True)
    print()

    tasks = []
    for a in args.assets:
        for (ws, we) in windows:
            tasks.append((a, args.date, args.step, ws, we, args.entry_mode,
                          args.latency_s, args.commission_r, args.no_cache_optim))
    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(backtest_asset, t): t[0] for t in tasks}
        done = 0
        for fut in as_completed(futs):
            done += 1
            r = fut.result()
            results.append(r)
            if r.get("ok"):
                print(f"[{done:2d}/{len(tasks)}] {r['asset']:8s} setups={r['n_setups']:3d} "
                      f"fermes={r['n_closed']:3d} WR={r['wr']:5.1f}% PnL={r['pnl_r']:+6.1f}R "
                      f"NF={r['n_nofill']} OPEN={r['n_open']}", flush=True)
            else:
                print(f"[{done:2d}/{len(tasks)}] FAIL {r['asset']} : {r.get('error','')[:90]}", flush=True)

    elapsed = time.time() - t0
    print(f"\n=== TERMINE en {elapsed/60:.1f}min ===")

    ok = [r for r in results if r.get("ok")]

    # Collecte TOUS les trades et DEDUPLIQUE (un OB peut etre vu dans 2 fenetres
    # contigues si a cheval). Cle = (instrument, ob_ts, direction).
    all_tr = []
    for r in ok:
        all_tr.extend(r.get("trades", []))
    seen = set()
    dedup = []
    for t in all_tr:
        key = (t["instrument"], t["ob_ts"], t["direction"])
        if key in seen:
            continue
        seen.add(key)
        dedup.append(t)
    all_tr = dedup

    closed = [t for t in all_tr if t["outcome"] in ("WIN", "LOSS")]
    tot_closed = len(closed)
    tot_win = sum(1 for t in closed if t["outcome"] == "WIN")
    tot_pnl = sum(t["pnl_r"] for t in closed)
    wr = (tot_win / tot_closed * 100) if tot_closed else 0
    print(f"\n=== RECAP GLOBAL TICK (dedup) ===")
    print(f"  Trades fermes : {tot_closed}")
    print(f"  WR            : {wr:.1f}%")
    print(f"  PnL (R)       : {tot_pnl:+.1f}")
    print(f"  NO_FILL       : {sum(1 for t in all_tr if t['outcome']=='NO_FILL')}")
    print(f"  INVALID_PRICE : {sum(1 for t in all_tr if t['outcome']=='INVALID_PRICE')} (prix deja du mauvais cote = ordre rejete MT5)")
    print(f"  OPEN          : {sum(1 for t in all_tr if t['outcome']=='OPEN')}")
    print(f"\n  Par actif :")
    by_a = {}
    for t in closed:
        by_a.setdefault(t["instrument"], []).append(t)
    for a in sorted(by_a):
        ts = by_a[a]
        w = sum(1 for x in ts if x["outcome"] == "WIN")
        print(f"    {a:8s} : {len(ts):3d} trades, WR={w/len(ts)*100:5.1f}%, PnL={sum(x['pnl_r'] for x in ts):+.1f}R")

    out = f"{ROOT}/bt_tick_{args.date.replace('-','')}_trades.csv"
    if all_tr:
        pd.DataFrame(all_tr).to_csv(out, index=False)
        print(f"\nJournal : {out}")


if __name__ == "__main__":
    main()
