"""V18.8 ICT PUR : Force le ML a apprendre les patterns ICT sans memorisation d'epoque.

Probleme V18.6/V18.7 (adversarial AUC 0.85-0.99) : le ML utilise des features
ABSOLUES (atr_at_setup en USD, dist_pct, risk_points) qui DRIFTENT entre les
epoques. En live, ces valeurs sont differentes -> drift -> potentiel echec.

Solution V18.8 : DROP toutes les features non-stationnaires.
On garde UNIQUEMENT :
- Patterns ICT (OB, sweep, FVG, SMT, killzones, confluences)
- Features relatives ATR-normalisees deja existantes (V18.7)
- Cyclic time (hour_sin/cos, day_of_week)

Le ML doit apprendre :
    "OB bullish + daily_bias + FVG_sync + KZ NY_AM + retest 1 -> WIN 70%"
PAS :
    "ATR=12 + prix=3500 -> WIN 70% (epoque 2025)"

Usage :
    python -m bot_v2.train_v18_8_ict_pur --all
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

ROOT = "c:/Users/Shadow/TradingBot" if sys.platform == "win32" else "/workspace/TradingBot"
sys.path.insert(0, ROOT)

ALL_ASSETS = [
    "BTCUSD", "ETHUSD", "USDMXN", "XAUUSD", "USDJPY", "USDZAR", "USDCAD", "NZDUSD",
    "GBPUSD", "CL-OIL", "EURUSD", "AUDUSD", "USDCHF", "NAS100", "XAGUSD",
    "DJ30", "GAS-C", "HK50", "GER40", "FRA40", "UK100", "Nikkei225",
    "BVSPX", "SP500", "Coffee-C", "Cocoa-C", "Wheat-C", "Sugar-C",
]

# ====================================================================
# FEATURES INTERDITES (drift d'epoque = memorisation, pas ICT pur)
# ====================================================================
NON_STATIONARY_DROP = {
    # Prix/ATR ABSOLUS
    "atr_at_setup",         # USD/EUR absolu (XAU 2018=0.5, 2026=12)
    "risk_points",          # points absolus (drift avec prix)
    # Distance en % calculee sur prix absolu (drift)
    "dist_to_pdh_pct",
    "dist_to_pdl_pct",
    "dist_to_d1_open_pct",
    # Distance en ATR mais ATR brut (drift)
    "dist_pdh_atr",
    "dist_pdl_atr",
    # ATR regime utilise ATR brut
    "atr_regime",
    # Logs des % qui drift (juste log des memes valeurs)
    "dist_pdh_log",
    "dist_pdl_log",
    # 9 features V18.4 jugees inutiles (V18.6 ablation)
    "dist_round_atr", "liq_asymmetry", "adr_consumed_pct",
    "bias_x_fvg", "bias_x_parent_x_fvg", "kz_ny_am_x_bias",
    "ob_x_sweep_strength", "unicorn_x_ny",
    # Hour linaire (cyclique mal traite) -- on garde hour_sin/cos
    "hour_of_day",
    # bars_to_exit n'est pas une feature mais une cible, double check
    "bars_to_exit",
}

# Features de meta-info
NON_FEATURES = {
    "instrument", "ts", "direction",
    "outcome", "pnl_usd", "bars_to_exit",
    "pnl_R", "pnl_R_clipped",
}

EMBARGO = pd.Timedelta(days=2)
TRAIN_END = pd.Timestamp("2025-05-22", tz="UTC")
VAL_START = TRAIN_END + EMBARGO
VAL_END = pd.Timestamp("2025-11-22", tz="UTC")
OOS_START = VAL_END + EMBARGO

# Hyperparams REGULARISES (memes que V18.6 qui marchaient bien)
HYPERPARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.01,
    "max_depth": 4,
    "num_leaves": 15,
    "min_child_samples": 500,
    "subsample": 0.7,
    "colsample_bytree": 0.5,
    "reg_alpha": 3.0,
    "reg_lambda": 20.0,
    "min_split_gain": 0.05,
    "random_state": 42,
    "verbosity": -1,
    "n_jobs": 4,
}

N_PARALLEL = int(os.environ.get("V18_8_PARALLEL", "6"))


def load_dataset(asset: str) -> pd.DataFrame | None:
    """Charge V18.7 (priorite : contient features normalisees), fallback V18.4."""
    for suffix in ("v18_7", "v18_4", "v18_3"):
        p = Path(f"{ROOT}/data/ml_dataset_{asset}_vantage_{suffix}.parquet")
        if p.exists():
            df = pd.read_parquet(p)
            df["ts"] = pd.to_datetime(df["ts"], utc=True)
            df = df.sort_values("ts").reset_index(drop=True)
            return df
    return None


def prepare_xy(df: pd.DataFrame):
    """V18.8 : drop toutes les features non-stationnaires."""
    df = df[df["outcome"].isin(["WIN", "LOSS"])].copy()
    df["target"] = (df["outcome"] == "WIN").astype(int)
    df = df.sort_values("ts").reset_index(drop=True)
    all_feats = [c for c in df.columns if c not in NON_FEATURES and c != "target"]
    # DROP features non-stationnaires
    feature_cols = [c for c in all_feats if c not in NON_STATIONARY_DROP]
    X = df[feature_cols].copy()
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
    y = df["target"]
    return X, y, feature_cols


def _eval_thresholds(name, y_true, y_pred, metrics):
    try:
        auc = roc_auc_score(y_true, y_pred)
    except Exception:
        auc = float("nan")
    metrics[f"auc_{name}"] = float(auc)
    for thr in (0.55, 0.60, 0.65, 0.70):
        mask = y_pred >= thr
        n = int(mask.sum())
        if n < 5:
            continue
        wr = float(y_true[mask].mean() * 100)
        metrics[f"{name}_n@{thr:.2f}"] = n
        metrics[f"{name}_wr@{thr:.2f}"] = wr


def train_one(asset: str) -> dict:
    df = load_dataset(asset)
    if df is None:
        return {"asset": asset, "status": "no_data"}

    df_closed = df[df["outcome"].isin(["WIN", "LOSS"])].copy()
    if len(df_closed) < 200:
        return {"asset": asset, "status": "not_enough"}

    train_df = df_closed[df_closed["ts"] < TRAIN_END].copy()
    val_df = df_closed[(df_closed["ts"] >= VAL_START) & (df_closed["ts"] < VAL_END)].copy()
    oos_df = df_closed[df_closed["ts"] >= OOS_START].copy()

    if len(train_df) < 200 or len(val_df) < 30:
        return {"asset": asset, "status": "bad_split"}

    X_train, y_train, feat_cols = prepare_xy(train_df)
    X_val, y_val, _ = prepare_xy(val_df)
    has_oos = len(oos_df) >= 10
    if has_oos:
        X_oos, y_oos, _ = prepare_xy(oos_df)
    else:
        X_oos, y_oos = X_val.head(0).copy(), y_val.head(0).copy()

    # Train
    model = lgb.LGBMClassifier(**HYPERPARAMS)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])

    # Calibration
    try:
        from sklearn.frozen import FrozenEstimator
        calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
        calibrated.fit(X_val, y_val)
    except ImportError:
        calibrated = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
        calibrated.fit(X_val, y_val)

    pred_tr = calibrated.predict_proba(X_train)[:, 1]
    pred_vl = calibrated.predict_proba(X_val)[:, 1]
    pred_oo = calibrated.predict_proba(X_oos)[:, 1] if has_oos else None

    metrics = {"asset": asset, "status": "trained",
               "n_features": len(feat_cols),
               "n_train": len(X_train), "n_val": len(X_val), "n_oos": len(X_oos)}
    _eval_thresholds("train", y_train, pred_tr, metrics)
    _eval_thresholds("val", y_val, pred_vl, metrics)
    if has_oos:
        _eval_thresholds("oos", y_oos, pred_oo, metrics)
    metrics["gap_train_oos"] = metrics.get("auc_train", 0) - metrics.get("auc_oos", 0)

    # Save
    model_path = Path(f"{ROOT}/bot_v2/ml_model_{asset}_vantage_v18_8.pkl")
    feat_path = Path(f"{ROOT}/bot_v2/ml_features_{asset}_vantage_v18_8.json")
    with open(model_path, "wb") as f:
        pickle.dump(calibrated, f)
    feat_path.write_text(json.dumps({"features": feat_cols}, indent=2))

    # Verdict
    verdict = "PASS"
    reasons = []
    auc_oos = metrics.get("auc_oos", 0)
    wr_oos = metrics.get("oos_wr@0.65", 0)
    n_oos = metrics.get("oos_n@0.65", 0)
    gap = metrics.get("gap_train_oos", 1)
    if auc_oos < 0.62:
        verdict = "FAIL"
        reasons.append(f"AUC OOS {auc_oos:.3f}")
    if gap > 0.18:
        verdict = "FAIL"
        reasons.append(f"GAP {gap:+.3f}")
    if n_oos >= 5 and wr_oos < 60:
        verdict = "FAIL"
        reasons.append(f"WR@65 {wr_oos:.1f}%")

    metrics["verdict"] = verdict
    metrics["reasons"] = reasons
    return metrics


def _worker(asset: str) -> dict:
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    try:
        t0 = time.time()
        r = train_one(asset)
        r["elapsed_sec"] = round(time.time() - t0, 1)
        return r
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"asset": asset, "status": "exception", "error": str(e)[:300]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true")
    p.add_argument("asset", nargs="?")
    args = p.parse_args()

    print(f"=== V18.8 ICT PUR : {len(ALL_ASSETS)} actifs ===")
    print(f"Drop {len(NON_STATIONARY_DROP)} features non-stationnaires :")
    for f in sorted(NON_STATIONARY_DROP):
        print(f"  - {f}")
    print(f"Parallel : {N_PARALLEL}")
    print()

    if args.all or args.asset == "--all":
        assets = ALL_ASSETS
    elif args.asset:
        assets = [args.asset]
    else:
        p.print_help()
        return

    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=N_PARALLEL) as ex:
        futs = {ex.submit(_worker, a): a for a in assets}
        for fut in as_completed(futs):
            a = futs[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {"asset": a, "status": "exception", "error": str(e)[:200]}
            results.append(r)
            v = r.get("verdict", r.get("status", "?"))
            auc = r.get("auc_oos", 0)
            wr = r.get("oos_wr@0.65", 0)
            n = r.get("oos_n@0.65", 0)
            gap = r.get("gap_train_oos", 0)
            elapsed = r.get("elapsed_sec", "?")
            nf = r.get("n_features", "?")
            print(f"  >>> [{len(results)}/{len(assets)}] {a:<10} "
                  f"{v:<6} feats={nf} "
                  f"AUC={auc:.3f} gap={gap:+.3f} "
                  f"WR@65={wr:.1f}% N={n} ({elapsed}s)", flush=True)

    elapsed_total = (time.time() - t0) / 60
    print(f"\n=== TERMINE en {elapsed_total:.1f} min ===")
    print(f"\n{'='*100}")
    print(f"=== RECAP V18.8 ICT PUR ===")
    print(f"{'='*100}")
    print(f"{'Asset':<11}{'NFeats':<8}{'AUC oos':<9}{'GAP':<9}{'WR@65':<8}{'N@65':<7}{'Verdict':<8}")
    for r in sorted(results, key=lambda x: -(x.get("auc_oos", 0) or 0)):
        if r.get("status") != "trained":
            print(f"{r['asset']:<11}{r.get('status', '?'):<10}")
            continue
        print(f"{r['asset']:<11}"
              f"{r.get('n_features', 0):<8}"
              f"{r.get('auc_oos', 0):<9.3f}"
              f"{r.get('gap_train_oos', 0):<+9.3f}"
              f"{r.get('oos_wr@0.65', 0):<8.1f}"
              f"{r.get('oos_n@0.65', '--')!s:<7}"
              f"{r.get('verdict', '--'):<8}")

    n_pass = sum(1 for r in results if r.get("verdict") == "PASS")
    print(f"\nPASS : {n_pass} / {len(results)}")

    recap = Path(f"{ROOT}/ml_metrics_v18_8_recap.json")
    recap.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nRecap : {recap.name}")


if __name__ == "__main__":
    main()
