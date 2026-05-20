"""Train V5 sur le dataset 3 mois (rebuild_v5_3mois.py).

Modele V5_3mois aligne avec ce que le live verra (3 mois de contexte).

Usage:
    python train_v5_3mois.py XAUUSD
"""
import os
import sys
import json
import pickle
import argparse
from pathlib import Path

import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


NON_FEATURES = {"instrument", "ts", "direction", "outcome", "pnl_usd", "bars_to_exit"}


def prepare(df, feat_cols=None):
    df = df[df["outcome"].isin(["WIN", "LOSS"])].copy()
    df["target"] = (df["outcome"] == "WIN").astype(int)
    df = df.sort_values("ts").reset_index(drop=True)
    if feat_cols is None:
        feat_cols = [c for c in df.columns if c not in NON_FEATURES and c != "target"]
    X = df[feat_cols].copy()
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
    return X, df["target"], feat_cols


def main():
    p = argparse.ArgumentParser()
    p.add_argument("asset")
    args = p.parse_args()

    asset = args.asset
    dataset_path = Path(f"{ROOT}/data/ml_dataset_{asset}_admiral_3mois_V5.parquet")
    model_path = Path(f"{ROOT}/bot_v2/ml_model_{asset}_admiral_v5_3mois.pkl")
    feat_path = Path(f"{ROOT}/bot_v2/ml_features_{asset}_admiral_v5_3mois.json")
    metrics_path = Path(f"{ROOT}/ml_metrics_{asset}_admiral_v5_3mois.txt")

    if not dataset_path.exists():
        print(f"!! Dataset manquant : {dataset_path}")
        return

    print(f"\n=== TRAIN V5 3 MOIS : {asset} ===")
    df = pd.read_parquet(dataset_path)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").reset_index(drop=True)

    closed = df[df["outcome"].isin(["WIN", "LOSS"])].copy()
    print(f"Total closed : {len(closed):,}")
    wr = (closed["outcome"] == "WIN").mean() * 100
    print(f"WR brut      : {wr:.1f}%")

    # Split temporel : 80% train, 10% val, 10% test
    n = len(closed)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    train_df = closed.iloc[:train_end].copy()
    val_df = closed.iloc[train_end:val_end].copy()
    test_df = closed.iloc[val_end:].copy()
    print(f"TRAIN : {len(train_df):>5} | VAL : {len(val_df):>5} | TEST : {len(test_df):>5}")

    X_train, y_train, feat_cols = prepare(train_df)
    X_val, y_val, _ = prepare(val_df, feat_cols)
    X_test, y_test, _ = prepare(test_df, feat_cols)

    cat_cols = []
    for c in feat_cols:
        if X_train[c].dtype == "object":
            X_train[c] = X_train[c].astype("category")
            X_val[c] = X_val[c].astype("category")
            X_test[c] = X_test[c].astype("category")
            cat_cols.append(c)

    print(f"Features : {len(feat_cols)}")

    model = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.04,
        max_depth=5, num_leaves=23, min_child_samples=30,
        subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.2, reg_lambda=0.3,
        random_state=42, verbosity=-1,
    )
    model.fit(
        X_train, y_train, eval_set=[(X_val, y_val)],
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)],
    )

    test_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, test_proba)
    metrics = [f"=== {asset} V5 3 MOIS ===", f"AUC TEST = {auc:.3f}"]
    print(f"\nAUC TEST = {auc:.3f}")

    for thr in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        mask = test_proba >= thr
        if mask.sum() < 3:
            continue
        wr_thr = y_test.values[mask].mean() * 100
        metrics.append(f"  seuil={thr:.2f} : {mask.sum():>4} trades, WR={wr_thr:.1f}%")
        print(f"  seuil={thr:.2f} : {mask.sum():>4} trades, WR={wr_thr:.1f}%")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    feat_path.write_text(json.dumps({"features": feat_cols, "categorical": cat_cols}, indent=2))
    metrics_path.write_text("\n".join(metrics))
    print(f"\nSauve : {model_path.name}")


if __name__ == "__main__":
    main()
