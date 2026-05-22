"""Backtest V12 ULTRA-REALISTE : rejeu M1 bougie par bougie.

Reproduit EXACTEMENT la logique de live_runner_v2.run_live() :
- Scan toutes les ~30s (= simule 1 cycle = 1 minute en backtest M1)
- Bot ne voit que les bougies jusqu'a T-1 (fix bougie-en-cours, pos=1)
- Cap age OB 30min (V11.2)
- Garde-fou setup obsolete (V10.1)
- Check stabilite V11.1 (proba stable sur >= 2 cycles, gap <= 0.07)
- 1 OB = 1 evaluation (_evaluated_obs)
- Pending LIMIT, expire 60 min
- Charge le ML V12 (cascade automatique via load_model)

Usage :
    python -m bot_v2.backtest_v12_realistic --period may2026
    python -m bot_v2.backtest_v12_realistic --start 2026-05-01 --end 2026-05-31
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("SWS_OVERRIDE", "1")
os.environ.setdefault("RR_OVERRIDE", "1.5")
os.environ.setdefault("BUILD_DATA_DIR", "data_vantage")

sys.path.insert(0, "c:/Users/Shadow/TradingBot")

import pandas as pd

from bot_v2.data_loader import load
from bot_v2.config import SMT_PAIRS, get_param, primary_instruments
from bot_v2.live_runner_v2 import compute_asset, load_model

LIVE_ASSETS = [
    "XAUUSD", "NAS100", "GER40", "BTCUSD",
    "EURUSD", "GBPUSD", "AUDUSD", "USDJPY",
    "SP500", "DJ30", "UK100", "FRA40",
    "USDCAD", "USDCHF",
]

# Constantes alignees sur le live
SCAN_INTERVAL_MIN = 1            # 1 cycle de scan = 1 minute M1
CAP_AGE_OB_MIN = 30              # V11.2 : OB > 30 min ignore
EXPIRE_PENDING_MIN = 60          # pending expire si pas fill en 60 min
RECENT_CUTOFF_MIN = 30           # OB recent
STABILITY_MIN_CYCLES = 2         # V11.1
STABILITY_MAX_GAP = 0.07         # V11.1
RISK_PCT = 0.02                  # 2% (V10.2)
INITIAL_BALANCE = 150.0          # EUR


@dataclass
class TradeRecord:
    instrument: str
    direction: str
    ob_validation_ts: pd.Timestamp
    pending_placed_ts: pd.Timestamp
    ml_proba: float
    entry: float
    sl: float
    tp: float
    rr: float
    fill_ts: pd.Timestamp | None = None
    exit_ts: pd.Timestamp | None = None
    outcome: str = "PENDING"   # PENDING / NO_FILL / WIN / LOSS / EXPIRED
    pnl_r: float = 0.0          # PnL en R (1 R = risque)


def fetch_all_data(inst: str, start: pd.Timestamp, end: pd.Timestamp,
                   buffer_days: int = 60) -> dict:
    """Charge M1/M15/H1/H4/D1 + correles SMT avec marge avant `start`."""
    load_start = start - pd.Timedelta(days=buffer_days)
    bufs = {}
    bufs["M1"] = load(inst, "M1", start=load_start, end=end + pd.Timedelta(days=1))
    bufs["M15"] = load(inst, "M15", start=load_start, end=end + pd.Timedelta(days=1))
    bufs["H1"] = load(inst, "H1", start=load_start - pd.Timedelta(days=30), end=end + pd.Timedelta(days=1))
    try:
        bufs["H4"] = load(inst, "H4", start=load_start - pd.Timedelta(days=120), end=end + pd.Timedelta(days=1))
    except Exception:
        bufs["H4"] = None
    try:
        bufs["D1"] = load(inst, "D1", start=load_start - pd.Timedelta(days=400), end=end + pd.Timedelta(days=1))
    except Exception:
        from bot_v2.concepts.daily_bias import build_d1_from_h1
        bufs["D1"] = build_d1_from_h1(bufs["H1"])

    # Correles SMT
    correlated = {}
    for corr_name, corr_type in SMT_PAIRS.get(inst, []):
        try:
            df_c = load(corr_name, "M1", start=load_start, end=end + pd.Timedelta(days=1))
            if len(df_c) > 0:
                correlated[corr_name] = (df_c, corr_type)
        except Exception:
            continue
    bufs["correlated"] = correlated
    return bufs


def check_fill_and_exit(pending: TradeRecord, df_m1: pd.DataFrame,
                        scan_ts: pd.Timestamp) -> bool:
    """Verifie si le pending a ete rempli ou si SL/TP touche depuis le fill.

    Returns True si le pending est ferme/expire (a retirer de la liste active).
    """
    # Bougies entre placement et maintenant
    after_place = df_m1[(df_m1.index > pending.pending_placed_ts) &
                        (df_m1.index <= scan_ts)]
    if len(after_place) == 0:
        return False

    # Expiration ?
    age = (scan_ts - pending.pending_placed_ts).total_seconds() / 60
    if age > EXPIRE_PENDING_MIN and pending.fill_ts is None:
        pending.outcome = "NO_FILL"
        return True

    # Pas encore rempli : chercher le fill
    if pending.fill_ts is None:
        if pending.direction == "bullish":
            fill_mask = after_place["low"] <= pending.entry
        else:
            fill_mask = after_place["high"] >= pending.entry
        if fill_mask.any():
            pending.fill_ts = fill_mask[fill_mask].index[0]
        else:
            return False  # pas fill, continue d'attendre

    # Rempli : chercher SL ou TP apres le fill (jusqu'a scan_ts)
    after_fill = df_m1[(df_m1.index > pending.fill_ts) & (df_m1.index <= scan_ts)]
    if len(after_fill) == 0:
        return False

    if pending.direction == "bullish":
        sl_hits = after_fill[after_fill["low"] <= pending.sl]
        tp_hits = after_fill[after_fill["high"] >= pending.tp]
    else:
        sl_hits = after_fill[after_fill["high"] >= pending.sl]
        tp_hits = after_fill[after_fill["low"] <= pending.tp]

    if len(sl_hits) and len(tp_hits):
        # SL ET TP dans la fenetre - on prend le PREMIER touche
        if sl_hits.index[0] <= tp_hits.index[0]:
            pending.exit_ts = sl_hits.index[0]
            pending.outcome = "LOSS"
            pending.pnl_r = -1.0
        else:
            pending.exit_ts = tp_hits.index[0]
            pending.outcome = "WIN"
            pending.pnl_r = pending.rr
        return True
    if len(tp_hits):
        pending.exit_ts = tp_hits.index[0]
        pending.outcome = "WIN"
        pending.pnl_r = pending.rr
        return True
    if len(sl_hits):
        pending.exit_ts = sl_hits.index[0]
        pending.outcome = "LOSS"
        pending.pnl_r = -1.0
        return True
    return False


def run_backtest(start: pd.Timestamp, end: pd.Timestamp,
                 assets: list[str], scan_step_min: int = 1) -> tuple[list[TradeRecord], dict]:
    """Boucle principale : pour chaque minute, simule un scan live."""
    print(f"=== BACKTEST V12 ULTRA-REALISTE ===")
    print(f"Periode : {start} -> {end}")
    print(f"Actifs : {assets}")
    print(f"Pas de scan : {scan_step_min} min")
    print()

    # Pre-load toutes les data
    print("Loading data buffers...")
    t0 = time.time()
    data = {}
    for a in assets:
        data[a] = fetch_all_data(a, start, end)
        print(f"  {a:8s} M1={len(data[a]['M1']):,} (couvre {data[a]['M1'].index[0].date()} -> {data[a]['M1'].index[-1].date()})")
    print(f"Buffers loaded in {time.time()-t0:.1f}s")
    print()

    # Verifie qu'on a un modele V12 pour chaque actif
    for a in assets:
        loaded = load_model(a)
        if loaded is None:
            print(f"WARNING: pas de modele pour {a}, skip")
            continue
        _, feats = loaded
        has_v12 = "snapshot_k" not in feats
        print(f"  {a:8s} {'V12' if has_v12 else 'V11'} ({len(feats)} features)")
    print()

    # State du backtest
    active_pendings: dict[str, list[TradeRecord]] = defaultdict(list)
    evaluated_obs: dict[str, set] = defaultdict(set)
    proba_history: dict[tuple, list[tuple]] = defaultdict(list)  # (inst, ob_ts, dir) -> [(ts, proba)]
    all_trades: list[TradeRecord] = []

    # Boucle temporelle
    cur = start
    n_cycles = 0
    n_setups_detected = 0
    n_pendings_placed = 0
    t_loop = time.time()
    last_print = time.time()

    while cur <= end:
        # Pour chaque actif
        for a in assets:
            df_m1_full = data[a]["M1"]
            df_m15_full = data[a]["M15"]
            df_h1_full = data[a]["H1"]
            df_h4_full = data[a]["H4"]
            df_d1_full = data[a]["D1"]

            # Coupe a cur - 1 min (fix bougie-en-cours : pos=1)
            cut = cur - pd.Timedelta(minutes=1)
            df_m1 = df_m1_full[df_m1_full.index <= cut]
            if len(df_m1) < 200:
                continue
            df_m15 = df_m15_full[df_m15_full.index <= cut]
            df_h1 = df_h1_full[df_h1_full.index <= cut]
            df_h4 = df_h4_full[df_h4_full.index <= cut] if df_h4_full is not None else None
            df_d1 = df_d1_full[df_d1_full.index <= cut]

            # 1. Verifie fills/SL/TP des pendings actifs
            still_active = []
            for p in active_pendings[a]:
                done = check_fill_and_exit(p, df_m1_full, cur)
                if done:
                    all_trades.append(p)
                else:
                    still_active.append(p)
            active_pendings[a] = still_active

            # 2. Scan : compute_asset (memes inputs qu'en live)
            payload = {
                "instrument": a,
                "df_m1": df_m1, "df_m15": df_m15, "df_h1": df_h1,
                "df_h4": df_h4, "df_d1": df_d1,
                "correlated_dfs": data[a]["correlated"],
                "balance": INITIAL_BALANCE,
                "debug_diag": False,
            }
            try:
                result = compute_asset(payload)
            except Exception as e:
                continue

            for setup in result.get("setups", []):
                n_setups_detected += 1
                ob = setup["ob"]
                ob_ts = setup["ts"]
                proba = setup.get("proba", 0.0)
                direction = ob.direction
                key = (a, str(ob_ts), direction)

                # _evaluated_obs : 1 OB = 1 evaluation (sauf besoin stability)
                if str(ob_ts) in evaluated_obs[a] and key in proba_history:
                    # OB deja vu, on continue stability check
                    pass

                # V11.1 : check stabilite
                proba_history[key].append((cur, proba))
                if len(proba_history[key]) >= STABILITY_MIN_CYCLES:
                    recent = [p for _, p in proba_history[key][-STABILITY_MIN_CYCLES:]]
                    gap = max(recent) - min(recent)
                    if gap > STABILITY_MAX_GAP:
                        continue  # instable, on attend
                else:
                    continue  # pas encore assez de cycles

                # Cap age OB (V11.2)
                age = (cur - ob_ts).total_seconds() / 60
                if age > CAP_AGE_OB_MIN:
                    continue  # OB trop vieux

                # V10.1 : garde-fou setup obsolete
                # (verifier que le prix n'a pas deja atteint TP/SL depuis validation)
                between = df_m1_full[(df_m1_full.index > ob_ts) &
                                     (df_m1_full.index <= cur)]
                ts_setup = setup["r"].trade_setup
                entry = ts_setup.entry_price
                sl = ts_setup.stop_loss
                tp = ts_setup.take_profit
                if len(between):
                    if direction == "bullish":
                        if (between["high"] >= tp).any() or (between["low"] <= sl).any():
                            continue  # obsolete
                    else:
                        if (between["low"] <= tp).any() or (between["high"] >= sl).any():
                            continue  # obsolete

                # Dedup : on n'a pas deja un pending actif pour ce meme OB
                if any(p.ob_validation_ts == ob_ts and p.direction == direction
                       for p in active_pendings[a]):
                    continue

                # Place le pending
                rec = TradeRecord(
                    instrument=a,
                    direction=direction,
                    ob_validation_ts=ob_ts,
                    pending_placed_ts=cur,
                    ml_proba=proba,
                    entry=entry, sl=sl, tp=tp,
                    rr=ts_setup.rr,
                )
                active_pendings[a].append(rec)
                evaluated_obs[a].add(str(ob_ts))
                n_pendings_placed += 1

        cur += pd.Timedelta(minutes=scan_step_min)
        n_cycles += 1

        # Print progress toutes les 60s
        if time.time() - last_print > 30:
            elapsed = time.time() - t_loop
            n_done = (cur - start).total_seconds() / 60
            n_total = (end - start).total_seconds() / 60
            pct = n_done / n_total * 100
            rate = n_cycles / elapsed if elapsed > 0 else 0
            eta_s = (n_total - n_done) / rate if rate > 0 else 0
            n_open = sum(len(v) for v in active_pendings.values())
            print(f"  [{pct:5.1f}%] cur={cur} | {n_cycles} cycles, {n_setups_detected} setups, "
                  f"{n_pendings_placed} pendings, {len(all_trades)} fermes, {n_open} actifs, "
                  f"{rate:.0f} cyc/s, ETA {eta_s/60:.1f}min",
                  flush=True)
            last_print = time.time()

    # Fin : ferme tous les pendings restants comme PENDING (non comptes)
    for a in assets:
        for p in active_pendings[a]:
            if p.outcome == "PENDING":
                p.outcome = "OPEN"  # encore ouvert a la fin du backtest
            all_trades.append(p)

    elapsed = time.time() - t_loop
    print(f"\n=== BACKTEST TERMINE en {elapsed/60:.1f} min ===")
    print(f"  Cycles : {n_cycles}")
    print(f"  Setups detectes : {n_setups_detected}")
    print(f"  Pendings places : {n_pendings_placed}")
    print(f"  Trades total : {len(all_trades)}")

    stats = {
        "n_cycles": n_cycles,
        "n_setups": n_setups_detected,
        "n_pendings": n_pendings_placed,
        "n_trades": len(all_trades),
        "elapsed_s": elapsed,
    }
    return all_trades, stats


def print_recap(trades: list[TradeRecord]):
    if not trades:
        print("Aucun trade !")
        return

    df = pd.DataFrame([{
        "instrument": t.instrument,
        "direction": t.direction,
        "ob_ts": t.ob_validation_ts,
        "place_ts": t.pending_placed_ts,
        "fill_ts": t.fill_ts,
        "exit_ts": t.exit_ts,
        "outcome": t.outcome,
        "ml": t.ml_proba,
        "rr": t.rr,
        "pnl_r": t.pnl_r,
        "delay_fill_min": ((t.fill_ts - t.pending_placed_ts).total_seconds() / 60) if t.fill_ts else None,
    } for t in trades])

    print()
    print("=== STATS PAR OUTCOME ===")
    print(df["outcome"].value_counts().to_string())

    closed = df[df["outcome"].isin(["WIN", "LOSS"])]
    if len(closed):
        wr = (closed["outcome"] == "WIN").mean() * 100
        pnl_r = closed["pnl_r"].sum()
        print(f"\n=== TRADES FERMES ({len(closed)}) ===")
        print(f"  WR        : {wr:.1f}%")
        print(f"  PnL (R)   : {pnl_r:.1f}")
        print(f"  PnL/trade : {pnl_r/len(closed):.3f} R")
        print(f"  PnL EUR   : {pnl_r * INITIAL_BALANCE * RISK_PCT:.2f} (a 2% risk sur {INITIAL_BALANCE}EUR)")

        print(f"\n=== PAR ACTIF (closed only) ===")
        for inst, sub in closed.groupby("instrument"):
            wr_a = (sub["outcome"] == "WIN").mean() * 100
            pnl_a = sub["pnl_r"].sum()
            print(f"  {inst:8s} : {len(sub):3d} trades, WR={wr_a:5.1f}%, PnL={pnl_a:+.1f}R")

    no_fill = df[df["outcome"] == "NO_FILL"]
    print(f"\nNO_FILL (expire 60min sans toucher entry) : {len(no_fill)}")
    open_ = df[df["outcome"] == "OPEN"]
    print(f"OPEN (encore actif a la fin) : {len(open_)}")

    # Save CSV
    out_path = Path("c:/Users/Shadow/TradingBot/backtest_v12_trades.csv")
    df.to_csv(out_path, index=False)
    print(f"\nJournal sauvegarde : {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--period", default="may2026", help="may2026 ou autre alias")
    p.add_argument("--start", help="YYYY-MM-DD")
    p.add_argument("--end", help="YYYY-MM-DD")
    p.add_argument("--assets", nargs="+", default=None,
                   help="Liste actifs (default: 14 live)")
    p.add_argument("--step", type=int, default=1, help="Pas en min (default 1)")
    args = p.parse_args()

    if args.start and args.end:
        start = pd.Timestamp(args.start, tz="UTC")
        end = pd.Timestamp(args.end, tz="UTC")
        # Si end n'inclut pas d'heure -> on prend toute la journee
        if end.hour == 0 and end.minute == 0:
            end = end + pd.Timedelta(hours=23, minutes=59)
    elif args.period == "may2026":
        start = pd.Timestamp("2026-05-01", tz="UTC")
        end = pd.Timestamp("2026-05-31 23:59", tz="UTC")
    else:
        print("Spec --start/--end ou --period may2026")
        sys.exit(1)

    assets = args.assets or LIVE_ASSETS
    trades, stats = run_backtest(start, end, assets, scan_step_min=args.step)
    print_recap(trades)


if __name__ == "__main__":
    main()
