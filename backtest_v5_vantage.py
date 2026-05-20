"""Backtest V5 OOS sur les donnees Vantage (data_vantage/) exportees depuis MT5.

Rejoue le pipeline Vizion + ML V5 (seuil 0.75) sur les ~7 mois de M1 Vantage
pour les 14 actifs. Compte trades, WR, P&L.

L'avantage vs backtest Admiral 8 ans :
- Vrai broker Vantage RAW ECN (memes spreads, memes timestamps que live)
- Plus court = plus pertinent pour valider la perf actuelle

Usage:
    python backtest_v5_vantage.py XAUUSD       # un seul actif
    python backtest_v5_vantage.py --all        # les 14 actifs

Output:
    backtest_v5_vantage_{ASSET}.txt  : recap par actif
    backtest_v5_vantage_all.txt      : recap global
"""
import os
import sys
import time
import json
import pickle
import argparse
from pathlib import Path

# Limite threads pour eviter crash en parallele 14 actifs
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
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
from bot_v2.concepts.killzones import killzone_at
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

ML_THRESHOLD = 0.75  # V5
RISK_PCT = 0.01      # 1% par trade
BALANCE = 150.0      # equivalent live


def load_vantage(asset, tf):
    """Charge un parquet Vantage."""
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


def backtest_one(asset, verbose=True):
    """Backtest V5 sur Vantage data pour un actif."""
    print(f"\n{'='*70}", flush=True)
    print(f"BACKTEST V5 OOS VANTAGE : {asset}", flush=True)
    print(f"{'='*70}", flush=True)

    df_m1 = load_vantage(asset, "M1")
    if df_m1 is None or len(df_m1) < 1000:
        print(f"  KO: M1 manquant ou trop petit", flush=True)
        return None
    df_m15 = load_vantage(asset, "M15")
    df_h1 = load_vantage(asset, "H1")
    df_h4 = load_vantage(asset, "H4")
    df_d1 = load_vantage(asset, "D1")

    if df_m15 is None or df_h1 is None:
        print(f"  KO: M15 ou H1 manquant", flush=True)
        return None

    if df_d1 is None:
        df_d1 = build_d1_from_h1(df_h1)

    period_days = (df_m1.index[-1] - df_m1.index[0]).days
    print(f"  M1   : {len(df_m1):,} bougies | {df_m1.index[0].date()} -> {df_m1.index[-1].date()} ({period_days}j)", flush=True)
    print(f"  M15  : {len(df_m15):,} bougies", flush=True)
    print(f"  H1   : {len(df_h1):,} bougies", flush=True)

    # Modele V5
    model, features = load_model_v5(asset)
    if model is None:
        print(f"  KO: modele V5 manquant", flush=True)
        return None
    print(f"  Modele V5 charge ({len(features)} features)", flush=True)

    # HTF swings
    htf_dfs = {"H1": df_h1, "D1": df_d1}
    if df_h4 is not None:
        htf_dfs["H4"] = df_h4
    htf_swings = collect_htf_swings(htf_dfs, swing_strength=3)

    # SMT correles
    correlated_dfs = {}
    for corr_name, corr_type in SMT_PAIRS.get(asset, []):
        df_c = load_vantage(corr_name, "M1")
        if df_c is not None:
            correlated_dfs[corr_name] = (df_c, corr_type)

    # Detection OBs (meme params que scan_asset)
    sws = get_param(asset, "swing_strength_m1", 2)
    t0 = time.time()
    obs = detect_order_blocks(df_m1, swing_strength=sws, max_group_size=2)
    print(f"  OBs detectes : {len(obs):,} (en {time.time()-t0:.1f}s)", flush=True)

    cache = {
        "swings_ltf": find_swings(df_m1, strength=sws),
        "fvgs_ltf": detect_fvg(df_m1),
        "breakers_ltf": detect_breakers(df_m1),
        "obs_htf": detect_order_blocks(df_m15),
    }
    cache["structure_breaks"] = detect_structure_breaks(df_m1, swings=cache["swings_ltf"])
    cache["htf_trend"] = detect_trend(cache["swings_ltf"], lookback=6)
    cache["obs_htf2"] = detect_order_blocks(df_h1)

    mss_setups = detect_mss_setups(df_m1, structure_breaks=cache["structure_breaks"], swings=cache["swings_ltf"])
    print(f"  MSS detectes : {len(mss_setups):,}", flush=True)

    # Boucle pipeline + ML
    trades = []
    rejets = {}
    t0 = time.time()
    last_log = t0
    for i, ob in enumerate(obs):
        if time.time() - last_log > 30:
            done_pct = i / len(obs) * 100
            print(f"  Progression : {i}/{len(obs)} ({done_pct:.0f}%) | {len(trades)} trades trouves", flush=True)
            last_log = time.time()

        try:
            r = evaluate_ob(
                ob, df_m1, df_m15, df_d1, asset,
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

        # ML predict
        feats = ml_filter._features_from_result(r, ob, asset, df_ltf=df_m1, df_d1=df_d1, mss_setups=mss_setups)
        X = pd.DataFrame([[feats.get(f, 0) for f in features]], columns=features)
        proba = float(model.predict_proba(X)[0, 1])

        if proba < ML_THRESHOLD:
            rejets[f"ml<{ML_THRESHOLD}"] = rejets.get(f"ml<{ML_THRESHOLD}", 0) + 1
            continue

        # Simule le trade
        setup = r.trade_setup
        lots, risk_usd = compute_position_size(
            setup.entry_price, setup.stop_loss, asset,
            balance=BALANCE, risk_pct=RISK_PCT,
        )
        if lots <= 0:
            rejets["lots_zero"] = rejets.get("lots_zero", 0) + 1
            continue
        sim_setup = TradeSetup(
            instrument=asset, direction=setup.direction,
            entry_price=setup.entry_price, stop_loss=setup.stop_loss,
            take_profit=setup.take_profit, rr=setup.rr,
            risk_points=setup.risk_points, reward_points=setup.reward_points,
            risk_usd=risk_usd, reward_usd=risk_usd * setup.rr,
            position_size_lots=lots,
            ob_validation_ts=setup.ob_validation_ts,
            tp_source=setup.tp_source,
        )
        tr = simulate_trade(sim_setup, df_m1, ob.validation_index + 1)
        trades.append({
            "ts": df_m1.index[ob.validation_index],
            "direction": ob.direction,
            "score": r.score,
            "proba": proba,
            "outcome": tr.outcome,
            "pnl_usd": tr.pnl_usd,
            "entry": setup.entry_price,
            "sl": setup.stop_loss,
            "tp": setup.take_profit,
            "rr": setup.rr,
        })

    duration = time.time() - t0
    print(f"  Pipeline+ML termine en {duration:.0f}s", flush=True)

    # Stats
    df_trades = pd.DataFrame(trades)
    if len(df_trades) == 0:
        print(f"\n  AUCUN trade pris au seuil {ML_THRESHOLD}", flush=True)
        return {"asset": asset, "trades": 0, "wr": 0, "pnl": 0, "trades_df": df_trades}

    closed = df_trades[df_trades["outcome"].isin(["WIN", "LOSS"])]
    n_win = (closed["outcome"] == "WIN").sum()
    n_loss = (closed["outcome"] == "LOSS").sum()
    n_pending = (df_trades["outcome"] == "PENDING").sum()
    n_no_fill = (df_trades["outcome"] == "NO_FILL").sum()
    wr = n_win / len(closed) * 100 if len(closed) > 0 else 0
    pnl_total = df_trades["pnl_usd"].sum()
    avg_proba = df_trades["proba"].mean()

    print(f"\n  RECAP {asset} V5 VANTAGE :", flush=True)
    print(f"    Trades pris (ML>={ML_THRESHOLD}) : {len(df_trades)}", flush=True)
    print(f"    WIN  : {n_win}", flush=True)
    print(f"    LOSS : {n_loss}", flush=True)
    print(f"    NO_FILL : {n_no_fill}", flush=True)
    print(f"    PENDING : {n_pending}", flush=True)
    print(f"    WR brut : {wr:.1f}% (sur {len(closed)} fermes)", flush=True)
    print(f"    PnL total : {pnl_total:+.2f} USD (sur balance {BALANCE})", flush=True)
    print(f"    PnL %     : {pnl_total/BALANCE*100:+.1f}%", flush=True)
    print(f"    Proba moy : {avg_proba:.3f}", flush=True)
    print(f"    Periode : {period_days}j", flush=True)
    print(f"    Trades/jour : {len(df_trades)/period_days:.2f}", flush=True)

    # Save trades csv
    out_csv = Path(ROOT) / f"backtest_v5_vantage_{asset}.csv"
    df_trades.to_csv(out_csv, index=False)
    print(f"    Trades sauves : {out_csv.name}", flush=True)

    return {
        "asset": asset,
        "trades": len(df_trades),
        "win": n_win,
        "loss": n_loss,
        "no_fill": n_no_fill,
        "pending": n_pending,
        "wr": wr,
        "pnl": pnl_total,
        "pnl_pct": pnl_total / BALANCE * 100,
        "period_days": period_days,
        "trades_per_day": len(df_trades) / period_days,
        "avg_proba": avg_proba,
        "trades_df": df_trades,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("asset", nargs="?", help="Asset name ou --all")
    p.add_argument("--all", action="store_true")
    args = p.parse_args()

    assets = ALL_ASSETS if (args.all or args.asset == "--all") else [args.asset]
    if not assets[0]:
        p.print_help()
        return

    all_results = []
    t_global = time.time()
    for a in assets:
        if a not in ALL_ASSETS:
            print(f"!! Actif inconnu : {a}", flush=True)
            continue
        try:
            res = backtest_one(a)
            if res is not None:
                all_results.append(res)
        except Exception as e:
            print(f"!! FAIL {a}: {e}", flush=True)
            import traceback; traceback.print_exc()

    # Recap global
    if len(all_results) > 1:
        print(f"\n\n{'='*90}", flush=True)
        print(f"RECAP GLOBAL V5 VANTAGE ({time.time()-t_global:.0f}s)", flush=True)
        print(f"{'='*90}", flush=True)
        print(f"{'ASSET':<10} {'TRADES':>8} {'WIN':>6} {'LOSS':>6} {'WR':>7} {'PnL$':>10} {'PnL%':>7} {'TR/J':>6} {'PROBA':>7}", flush=True)
        print("-"*90, flush=True)
        total_trades = total_win = total_loss = 0
        total_pnl = 0
        for r in all_results:
            print(f"{r['asset']:<10} {r['trades']:>8} {r['win']:>6} {r['loss']:>6} {r['wr']:>6.1f}% {r['pnl']:>+10.2f} {r['pnl_pct']:>+6.1f}% {r['trades_per_day']:>6.2f} {r['avg_proba']:>7.3f}", flush=True)
            total_trades += r['trades']
            total_win += r['win']
            total_loss += r['loss']
            total_pnl += r['pnl']
        print("-"*90, flush=True)
        total_wr = total_win / (total_win + total_loss) * 100 if (total_win + total_loss) > 0 else 0
        print(f"{'TOTAL':<10} {total_trades:>8} {total_win:>6} {total_loss:>6} {total_wr:>6.1f}% {total_pnl:>+10.2f} {total_pnl/BALANCE*100:>+6.1f}%", flush=True)


if __name__ == "__main__":
    main()
