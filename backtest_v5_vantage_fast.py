"""Backtest V5 OOS FAST sur Vantage : reproduit fidelement le live.

CLÉ : le bot live fetch 2000 M1 (~33h) à chaque scan. Donc pour CHAQUE OB,
on ne passe que les ~2000 bougies précédentes au pipeline (au lieu de 200k).

Gain : ~100x plus rapide que backtest_v5_vantage.py. Résultat identique au live.

Usage:
    python backtest_v5_vantage_fast.py XAUUSD
    python backtest_v5_vantage_fast.py --all
"""
import os
import sys
import time
import json
import pickle
import argparse
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("LIGHTGBM_NUM_THREADS", "4")

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from bot_v2.config import SMT_PAIRS, get_param
from bot_v2.concepts.order_block import detect_order_blocks
from bot_v2.concepts.liquidity import find_swings
from bot_v2.concepts.structure import detect_structure_breaks, detect_trend
from bot_v2.concepts.fvg import detect_fvg
from bot_v2.concepts.breaker import detect_breakers
from bot_v2.concepts.mss_setup import detect_mss_setups
from bot_v2.concepts.htf_swings import collect_htf_swings
from bot_v2.concepts.daily_bias import build_d1_from_h1
from bot_v2.pipeline import evaluate_ob
from bot_v2.backtest import simulate_trade
from bot_v2.trade_setup import compute_position_size, TradeSetup
from bot_v2 import ml_filter


VANTAGE_DIR = Path(ROOT) / "data_vantage"
ALL_ASSETS = [
    "XAUUSD", "NAS100", "GER40", "BTCUSD",
    "EURUSD", "GBPUSD", "AUDUSD", "USDJPY",
    "SP500", "DJ30", "UK100", "FRA40",
    "USDCAD", "USDCHF",
]

ML_THRESHOLD = 0.75
RISK_PCT = 0.01
BALANCE = 154.96

# Fenetre live (memes valeurs que live_runner_v2)
# TEST DIAG 2026-05-20 : augmente a ~3 mois pour matcher le training V5
N_BARS_M1 = int(os.environ.get("N_BARS_M1", "2000"))   # 2000=33h | 120000=~3 mois
N_BARS_M15 = int(os.environ.get("N_BARS_M15", "500"))
N_BARS_H1 = int(os.environ.get("N_BARS_H1", "500"))
N_BARS_D1 = int(os.environ.get("N_BARS_D1", "100"))

# Limite periode test (derniers N jours seulement) - 0 = tout
DEFAULT_LAST_DAYS = 14


def load_vantage(asset, tf):
    path = VANTAGE_DIR / f"{asset}_{tf}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def load_model_v5(asset):
    pkl = os.path.join(ROOT, "bot_v2", f"ml_model_{asset}_admiral_v5.pkl")
    feat = os.path.join(ROOT, "bot_v2", f"ml_features_{asset}_admiral_v5.json")
    if not os.path.exists(pkl):
        return None, None
    with open(pkl, "rb") as f:
        model = pickle.load(f)
    features = json.loads(open(feat).read())["features"]
    return model, features


def backtest_one(asset, last_days=DEFAULT_LAST_DAYS):
    print(f"\n{'='*70}", flush=True)
    print(f"BACKTEST V5 FAST (live-like) : {asset}  [{last_days} derniers jours]", flush=True)
    print(f"{'='*70}", flush=True)

    df_m1_full = load_vantage(asset, "M1")
    if df_m1_full is None or len(df_m1_full) < 2000:
        print(f"  KO: M1 manquant", flush=True)
        return None
    df_m15_full = load_vantage(asset, "M15")
    df_h1_full = load_vantage(asset, "H1")
    df_h4_full = load_vantage(asset, "H4")
    df_d1_full = load_vantage(asset, "D1")
    if df_m15_full is None or df_h1_full is None:
        return None
    if df_d1_full is None:
        df_d1_full = build_d1_from_h1(df_h1_full)

    # Cutoff : derniers N jours
    cutoff_ts = df_m1_full.index[-1] - pd.Timedelta(days=last_days) if last_days > 0 else df_m1_full.index[0]
    period_days = (df_m1_full.index[-1] - max(cutoff_ts, df_m1_full.index[0])).days
    print(f"  M1   : {len(df_m1_full):,} bougies total | test: {cutoff_ts.date()} -> {df_m1_full.index[-1].date()} ({period_days}j)", flush=True)

    model, features = load_model_v5(asset)
    if model is None:
        print(f"  KO: modele V5 manquant", flush=True)
        return None
    print(f"  Modele V5 charge ({len(features)} features)", flush=True)

    # SMT (charge complet)
    correlated_full = {}
    for corr_name, corr_type in SMT_PAIRS.get(asset, []):
        df_c = load_vantage(corr_name, "M1")
        if df_c is not None:
            correlated_full[corr_name] = (df_c, corr_type)

    # Detection OBs sur tout l'historique (rapide, ~3s)
    sws = get_param(asset, "swing_strength_m1", 2)
    t0 = time.time()
    obs_all = detect_order_blocks(df_m1_full, swing_strength=sws, max_group_size=2)
    # Filtre OBs sur la fenetre test seulement
    obs_all = [ob for ob in obs_all if df_m1_full.index[ob.validation_index] >= cutoff_ts]
    print(f"  OBs detectes ({last_days}j) : {len(obs_all):,} (en {time.time()-t0:.1f}s)", flush=True)

    # Pour chaque OB : on construit une "vue live" = les 2000 M1 + 500 M15 + 500 H1 jusqu'au OB
    trades = []
    rejets = {}
    all_probas = []  # toutes les probas calculees
    skip_start = 2000  # on ignore les premiers OBs (pas assez d'historique)
    t0 = time.time()
    last_log = t0

    for i, ob in enumerate(obs_all):
        if ob.validation_index < skip_start:
            continue

        ob_ts = df_m1_full.index[ob.validation_index]

        # Vue live : 2000 M1 jusqu'a l'OB (inclus)
        end_idx = ob.validation_index + 1
        start_idx = max(0, end_idx - N_BARS_M1)
        df_m1 = df_m1_full.iloc[start_idx:end_idx].copy()

        # M15/H1/D1/H4 : on slice par timestamp (les dernieres N bougies <= ob_ts)
        df_m15 = df_m15_full[df_m15_full.index <= ob_ts].tail(N_BARS_M15).copy()
        df_h1 = df_h1_full[df_h1_full.index <= ob_ts].tail(N_BARS_H1).copy()
        df_d1 = df_d1_full[df_d1_full.index <= ob_ts].tail(N_BARS_D1).copy()
        df_h4 = df_h4_full[df_h4_full.index <= ob_ts].tail(500).copy() if df_h4_full is not None else None

        if len(df_m1) < 200 or len(df_m15) < 100 or len(df_h1) < 100:
            continue

        # HTF swings (recalcule sur la vue locale)
        htf_dfs = {"H1": df_h1, "D1": df_d1}
        if df_h4 is not None and len(df_h4) > 50:
            htf_dfs["H4"] = df_h4
        htf_swings = collect_htf_swings(htf_dfs, swing_strength=3)

        # Correles slice
        correlated_dfs = {}
        for corr_name, (df_c_full, corr_type) in correlated_full.items():
            df_c = df_c_full[df_c_full.index <= ob_ts].tail(N_BARS_M1).copy()
            if len(df_c) > 0:
                correlated_dfs[corr_name] = (df_c, corr_type)

        # Cache local (sur les 2000 M1, donc rapide)
        cache = {
            "swings_ltf": find_swings(df_m1, strength=sws),
            "fvgs_ltf": detect_fvg(df_m1),
            "breakers_ltf": detect_breakers(df_m1),
            "obs_htf": detect_order_blocks(df_m15),
        }
        cache["structure_breaks"] = detect_structure_breaks(df_m1, swings=cache["swings_ltf"])
        cache["htf_trend"] = detect_trend(cache["swings_ltf"], lookback=6)
        cache["obs_htf2"] = detect_order_blocks(df_h1)

        # MSS local
        mss_setups = detect_mss_setups(df_m1, structure_breaks=cache["structure_breaks"], swings=cache["swings_ltf"])

        # On doit retrouver l'OB dans la vue locale (son index a change)
        # Cherche par validation_ts
        local_obs = detect_order_blocks(df_m1, swing_strength=sws, max_group_size=2)
        ob_local = None
        for o in local_obs:
            if df_m1.index[o.validation_index] == ob_ts and o.direction == ob.direction:
                ob_local = o
                break
        if ob_local is None:
            rejets["ob_not_in_view"] = rejets.get("ob_not_in_view", 0) + 1
            continue

        try:
            r = evaluate_ob(
                ob_local, df_m1, df_m15, df_d1, asset,
                ltf_name="M1", htf_name="M15",
                df_htf2=df_h1, htf2_name="H1",
                correlated_dfs=correlated_dfs,
                htf_swings=htf_swings,
                df_h1=df_h1, min_score=0, min_quality=0,
                cache=cache,
            )
        except Exception as e:
            rejets["exception"] = rejets.get("exception", 0) + 1
            continue

        if r.verdict != "TRADE" or r.trade_setup is None:
            reason = (r.rejection_reason or "no_trade")[:50]
            rejets[reason] = rejets.get(reason, 0) + 1
            continue

        feats = ml_filter._features_from_result(r, ob_local, asset, df_ltf=df_m1, df_d1=df_d1, mss_setups=mss_setups)
        X = pd.DataFrame([[feats.get(f, 0) for f in features]], columns=features)
        proba = float(model.predict_proba(X)[0, 1])

        # Track distribution probas
        all_probas.append(proba)

        if proba < ML_THRESHOLD:
            rejets[f"ml<{ML_THRESHOLD}"] = rejets.get(f"ml<{ML_THRESHOLD}", 0) + 1
            continue

        # Simule le trade sur le M1 full (apres l'OB)
        setup = r.trade_setup
        lots, risk_usd = compute_position_size(setup.entry_price, setup.stop_loss, asset, balance=BALANCE, risk_pct=RISK_PCT)
        if lots <= 0:
            continue
        sim_setup = TradeSetup(
            instrument=asset, direction=setup.direction,
            entry_price=setup.entry_price, stop_loss=setup.stop_loss,
            take_profit=setup.take_profit, rr=setup.rr,
            risk_points=setup.risk_points, reward_points=setup.reward_points,
            risk_usd=risk_usd, reward_usd=risk_usd * setup.rr,
            position_size_lots=lots, ob_validation_ts=setup.ob_validation_ts,
            tp_source=setup.tp_source,
        )
        # Pour simuler : on a besoin du df_m1_full apres ob.validation_index
        tr = simulate_trade(sim_setup, df_m1_full, ob.validation_index + 1)
        trades.append({
            "ts": ob_ts, "direction": ob.direction, "score": r.score,
            "proba": proba, "outcome": tr.outcome, "pnl_usd": tr.pnl_usd,
            "entry": setup.entry_price, "sl": setup.stop_loss, "tp": setup.take_profit, "rr": setup.rr,
        })

        if time.time() - last_log > 30:
            done_pct = i / len(obs_all) * 100
            print(f"  Progression : {i}/{len(obs_all)} ({done_pct:.0f}%) | {len(trades)} trades", flush=True)
            last_log = time.time()

    duration = time.time() - t0
    print(f"  Pipeline+ML termine en {duration:.0f}s", flush=True)

    # Stats
    df_trades = pd.DataFrame(trades)
    if len(df_trades) == 0:
        print(f"\n  AUCUN trade pris au seuil {ML_THRESHOLD}", flush=True)
        # Diagnostic : top rejets + distribution probas
        sorted_rej = sorted(rejets.items(), key=lambda x: -x[1])
        print(f"\n  Top 10 rejets (OBs analyses = {len(obs_all)}):", flush=True)
        for reason, count in sorted_rej[:10]:
            print(f"    {count:>5}x : {reason}", flush=True)

        if all_probas:
            import numpy as _np
            probas = _np.array(all_probas)
            print(f"\n  Distribution probas ({len(probas)} OBs ont atteint le ML) :", flush=True)
            print(f"    min={probas.min():.3f} max={probas.max():.3f} mean={probas.mean():.3f} median={_np.median(probas):.3f}", flush=True)
            for thr in [0.3, 0.5, 0.6, 0.65, 0.70, 0.75, 0.80]:
                n_above = (probas >= thr).sum()
                print(f"    proba >= {thr:.2f} : {n_above} OBs ({n_above/len(probas)*100:.1f}%)", flush=True)
        return {"asset": asset, "trades": 0, "wr": 0, "pnl": 0}

    closed = df_trades[df_trades["outcome"].isin(["WIN", "LOSS"])]
    n_win = (closed["outcome"] == "WIN").sum()
    n_loss = (closed["outcome"] == "LOSS").sum()
    n_pending = (df_trades["outcome"] == "PENDING").sum()
    n_no_fill = (df_trades["outcome"] == "NO_FILL").sum()
    wr = n_win / len(closed) * 100 if len(closed) > 0 else 0
    pnl_total = df_trades["pnl_usd"].sum()

    print(f"\n  RECAP {asset} V5 VANTAGE FAST :", flush=True)
    print(f"    Trades pris (ML>={ML_THRESHOLD}) : {len(df_trades)}", flush=True)
    print(f"    WIN  : {n_win}", flush=True)
    print(f"    LOSS : {n_loss}", flush=True)
    print(f"    NO_FILL : {n_no_fill}", flush=True)
    print(f"    PENDING : {n_pending}", flush=True)
    print(f"    WR brut : {wr:.1f}% (sur {len(closed)} fermes)", flush=True)
    print(f"    PnL total : {pnl_total:+.2f} USD (balance {BALANCE})", flush=True)
    print(f"    PnL % : {pnl_total/BALANCE*100:+.1f}%", flush=True)
    print(f"    Periode : {period_days}j | Trades/jour : {len(df_trades)/period_days:.2f}", flush=True)

    out_csv = Path(ROOT) / f"backtest_v5_vantage_{asset}.csv"
    df_trades.to_csv(out_csv, index=False)

    return {
        "asset": asset, "trades": len(df_trades), "win": n_win, "loss": n_loss,
        "wr": wr, "pnl": pnl_total, "pnl_pct": pnl_total / BALANCE * 100,
        "trades_per_day": len(df_trades) / period_days,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("asset", nargs="?")
    p.add_argument("--all", action="store_true")
    p.add_argument("--days", type=int, default=DEFAULT_LAST_DAYS, help="Test last N days (default: 14)")
    args = p.parse_args()

    assets = ALL_ASSETS if (args.all or args.asset == "--all") else [args.asset]
    if not assets[0]:
        p.print_help()
        return

    results = []
    t0 = time.time()
    for a in assets:
        if a not in ALL_ASSETS:
            print(f"!! Actif inconnu : {a}", flush=True)
            continue
        try:
            r = backtest_one(a, last_days=args.days)
            if r:
                results.append(r)
        except Exception as e:
            print(f"!! FAIL {a}: {e}", flush=True)
            import traceback; traceback.print_exc()

    if len(results) > 1:
        print(f"\n\n{'='*90}", flush=True)
        print(f"RECAP GLOBAL ({time.time()-t0:.0f}s)", flush=True)
        print(f"{'='*90}", flush=True)
        print(f"{'ASSET':<10} {'TRADES':>8} {'WIN':>5} {'LOSS':>5} {'WR':>7} {'PnL$':>10} {'PnL%':>7} {'TR/J':>6}", flush=True)
        print("-"*90, flush=True)
        tot = {"trades": 0, "win": 0, "loss": 0, "pnl": 0}
        for r in results:
            print(f"{r['asset']:<10} {r['trades']:>8} {r['win']:>5} {r['loss']:>5} {r['wr']:>6.1f}% {r['pnl']:>+10.2f} {r['pnl_pct']:>+6.1f}% {r['trades_per_day']:>6.2f}", flush=True)
            tot["trades"] += r['trades']; tot["win"] += r['win']; tot["loss"] += r['loss']; tot["pnl"] += r['pnl']
        wr = tot["win"]/(tot["win"]+tot["loss"])*100 if (tot["win"]+tot["loss"]) > 0 else 0
        print("-"*90, flush=True)
        print(f"{'TOTAL':<10} {tot['trades']:>8} {tot['win']:>5} {tot['loss']:>5} {wr:>6.1f}% {tot['pnl']:>+10.2f} {tot['pnl']/BALANCE*100:>+6.1f}%", flush=True)


if __name__ == "__main__":
    main()
