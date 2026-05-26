"""V18.5 TRAIN FINAL : utilise les hyperparams optimises par genetique.

vs V18.4 :
- Charge ml_hyperparams_<ASSET>_v18_4.json (best_config) au lieu d'hyperparams en dur
- Fallback baseline V18.4 si fichier absent
- Sortie : ml_model_<ASSET>_vantage_v18_5.pkl

Usage :
    python -m bot_v2.train_v18_5_vantage --all
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
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

EMBARGO = pd.Timedelta(days=2)
TRAIN_END = pd.Timestamp("2025-05-22", tz="UTC")
VAL_START = TRAIN_END + EMBARGO
VAL_END = pd.Timestamp("2025-11-22", tz="UTC")
OOS_START = VAL_END + EMBARGO

NON_FEATURES = {
    "instrument", "ts", "direction",
    "outcome", "pnl_usd", "bars_to_exit",
    "pnl_R", "pnl_R_clipped",
}

# Fallback baseline si genetique non dispo
FALLBACK_HYPERPARAMS = {
    "n_estimators": 2000, "learning_rate": 0.01,
    "max_depth": 5, "num_leaves": 31, "min_child_samples": 200,
    "subsample": 0.7, "colsample_bytree": 0.6,
    "reg_alpha": 1.0, "reg_lambda": 5.0,
}


def load_hyperparams(asset: str) -> tuple[dict, str]:
    """Charge hyperparams optimises par genetique, ou fallback baseline."""
    p = Path(f"{ROOT}/ml_hyperparams_{asset}_v18_4.json")
    if p.exists():
        try:
            r = json.loads(p.read_text())
            cfg = r.get("best_config")
            if cfg:
                return cfg, f"genetic (val_gen AUC={r.get('best_auc_val_gen', 0):.4f})"
        except Exception as e:
            print(f"  [warn] {p.name} unreadable : {e}")
    return dict(FALLBACK_HYPERPARAMS), "fallback baseline"


def load_dataset(asset: str) -> pd.DataFrame | None:
    """Charge dataset V18.4 (priorite), fallback V18.3."""
    for suffix in ("v18_4", "v18_3"):
        p = Path(f"{ROOT}/data/ml_dataset_{asset}_vantage_{suffix}.parquet")
        if p.exists():
            print(f"  Chargement {p.name}")
            df = pd.read_parquet(p)
            df["ts"] = pd.to_datetime(df["ts"], utc=True)
            df = df.sort_values("ts").reset_index(drop=True)
            return df
    print(f"  Aucun dataset V18.4/V18.3 pour {asset}")
    return None


def prepare_xy(df: pd.DataFrame):
    df = df[df["outcome"].isin(["WIN", "LOSS"])].copy()
    df["target"] = (df["outcome"] == "WIN").astype(int)
    df = df.sort_values("ts").reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in NON_FEATURES and c != "target"]
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
    print(f"\n  {name.upper()} ({len(y_true)} trades) AUC={auc:.3f}")
    for thr in (0.55, 0.60, 0.65, 0.70, 0.75):
        mask = y_pred >= thr
        n = int(mask.sum())
        if n < 5:
            print(f"    seuil={thr:.2f} : {n:>3} trades, WR=-- (trop peu)")
            continue
        wr = float(y_true[mask].mean() * 100)
        metrics[f"{name}_n@{thr:.2f}"] = n
        metrics[f"{name}_wr@{thr:.2f}"] = wr
        print(f"    seuil={thr:.2f} : {n:>3} trades, WR={wr:.1f}%")


def train_one(asset: str) -> dict:
    print(f"\n{'='*60}")
    print(f"=== TRAIN {asset} V18.5 ===")
    print(f"{'='*60}")

    df = load_dataset(asset)
    if df is None:
        return {"asset": asset, "status": "no_data"}

    hyperparams, source = load_hyperparams(asset)
    print(f"  Hyperparams source : {source}")
    print(f"  Config : depth={hyperparams.get('max_depth')} "
          f"leaves={hyperparams.get('num_leaves')} "
          f"lr={hyperparams.get('learning_rate')} "
          f"min_child={hyperparams.get('min_child_samples')} "
          f"reg_a={hyperparams.get('reg_alpha')} "
          f"reg_l={hyperparams.get('reg_lambda')}")

    df_closed = df[df["outcome"].isin(["WIN", "LOSS"])].copy()
    n_total = len(df_closed)
    if n_total < 200:
        return {"asset": asset, "status": "not_enough", "n": n_total}

    wr_brut = (df_closed["outcome"] == "WIN").mean() * 100
    print(f"  N={n_total} | WR brut {wr_brut:.1f}%")

    train_df = df_closed[df_closed["ts"] < TRAIN_END].copy()
    val_df = df_closed[(df_closed["ts"] >= VAL_START) & (df_closed["ts"] < VAL_END)].copy()
    oos_df = df_closed[df_closed["ts"] >= OOS_START].copy()
    print(f"  Train={len(train_df)} Val={len(val_df)} OOS={len(oos_df)}")

    if len(train_df) < 100 or len(val_df) < 20:
        return {"asset": asset, "status": "bad_split",
                "n_train": len(train_df), "n_val": len(val_df), "n_oos": len(oos_df)}

    X_train, y_train, feat_cols = prepare_xy(train_df)
    X_val, y_val, _ = prepare_xy(val_df)
    has_oos = len(oos_df) >= 10
    if has_oos:
        X_oos, y_oos, _ = prepare_xy(oos_df)
    else:
        X_oos, y_oos = X_val.head(0).copy(), y_val.head(0).copy()

    cat_cols = []
    for c in feat_cols:
        if X_train[c].dtype == "object" or X_train[c].dtype.name == "category":
            X_train[c] = X_train[c].astype("category")
            X_val[c] = X_val[c].astype("category")
            if has_oos:
                X_oos[c] = X_oos[c].astype("category")
            cat_cols.append(c)

    print(f"  Features : {len(feat_cols)} (cat: {cat_cols})")

    metrics = {"asset": asset, "status": "trained",
               "n_train": len(X_train), "n_val": len(X_val), "n_oos": len(X_oos),
               "wr_brut_oos": float((y_oos.mean()) * 100) if has_oos else 0.0,
               "hyperparams_source": source,
               "hyperparams": hyperparams}

    model = lgb.LGBMClassifier(
        **hyperparams,
        min_split_gain=0.01,
        random_state=42, verbosity=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              categorical_feature=cat_cols if cat_cols else "auto",
              callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])

    print(f"  Calibration isotonique sur VAL ({len(X_val)})...")
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

    _eval_thresholds("train", y_train, pred_tr, metrics)
    _eval_thresholds("val", y_val, pred_vl, metrics)
    if has_oos:
        _eval_thresholds("oos", y_oos, pred_oo, metrics)

    model_path = Path(f"{ROOT}/bot_v2/ml_model_{asset}_vantage_v18_5.pkl")
    features_path = Path(f"{ROOT}/bot_v2/ml_features_{asset}_vantage_v18_5.json")
    with open(model_path, "wb") as f:
        pickle.dump(calibrated, f)
    features_path.write_text(json.dumps({"features": feat_cols}, indent=2))
    print(f"\n  Saved : {model_path.name}")

    # Verdict
    verdict = "PASS"
    reasons = []
    if has_oos:
        auc_oos = metrics.get("auc_oos", 0)
        wr_oos = metrics.get("oos_wr@0.65", 0)
        n_oos = metrics.get("oos_n@0.65", 0)
        if auc_oos < 0.65:
            verdict = "FAIL"
            reasons.append(f"AUC OOS {auc_oos:.3f} < 0.65")
        if n_oos >= 5 and wr_oos < 60:
            verdict = "FAIL"
            reasons.append(f"WR OOS @0.65 {wr_oos:.1f}% < 60%")
    metrics["verdict"] = verdict
    metrics["reasons"] = reasons
    print(f"\n  ===> {verdict} {' | '.join(reasons) if reasons else 'OK'}")
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("asset", nargs="?")
    p.add_argument("--all", action="store_true")
    args = p.parse_args()

    if args.all or args.asset == "--all":
        all_metrics = []
        for a in ALL_ASSETS:
            try:
                m = train_one(a)
            except Exception as e:
                print(f"!! FAIL {a} : {e}")
                import traceback; traceback.print_exc()
                m = {"asset": a, "status": "exception", "error": str(e)[:200]}
            all_metrics.append(m)

        print("\n" + "=" * 80)
        print(f"=== RECAP V18.5 ===")
        print("=" * 80)
        print(f"{'Asset':<10} {'Source':<10} {'AUC OOS':<8} {'WR@0.65':<8} {'N':<6} {'Verdict':<6}")
        for m in all_metrics:
            src = "genetic" if "genetic" in m.get("hyperparams_source", "") else "fallback"
            auc = f"{m.get('auc_oos', 0):.3f}" if "auc_oos" in m else "--"
            wr = f"{m.get('oos_wr@0.65', 0):.1f}%" if "oos_wr@0.65" in m else "--"
            n = m.get("oos_n@0.65", "--")
            verdict = m.get("verdict", "--")
            print(f"{m['asset']:<10} {src:<10} {auc:<8} {wr:<8} {n!s:<6} {verdict:<6}")
        n_pass = sum(1 for m in all_metrics if m.get("verdict") == "PASS")
        print(f"\nPASS : {n_pass} / {len(all_metrics)}")

        recap = Path(f"{ROOT}/ml_metrics_v18_5_recap.json")
        recap.write_text(json.dumps(all_metrics, indent=2, default=str))
        print(f"\nRecap : {recap.name}")

    elif args.asset:
        if args.asset not in ALL_ASSETS:
            print(f"Actif inconnu : {args.asset}")
            sys.exit(1)
        train_one(args.asset)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
