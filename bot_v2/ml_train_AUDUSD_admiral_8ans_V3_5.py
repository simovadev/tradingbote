"""Train LightGBM AUDUSD V3.5 - Admiral 8 ans.

Split temporel custom :
- TRAIN : -> 2025-01 (~6 ans)
- VAL   : 2025-01 -> 2025-05 (4 mois)
- TEST  : 2025-05+ (OOS)

Output :
- bot_v2/ml_model_AUDUSD_admiral_v3_5.pkl
- ml_metrics_AUDUSD_admiral_v3_5.txt
"""
from __future__ import annotations

import sys
ROOT = "c:/Users/Shadow/TradingBot" if sys.platform == "win32" else "/workspace/TradingBot"
sys.path.insert(0, ROOT)

import json
import pickle
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score


DATASET_PATH = Path(f"{ROOT}/data/ml_dataset_AUDUSD_admiral_8ans_V3_5.parquet")
MODEL_PATH = Path(f"{ROOT}/bot_v2/ml_model_AUDUSD_admiral_v3_5.pkl")
FEATURES_PATH = Path(f"{ROOT}/bot_v2/ml_features_AUDUSD_admiral_v3_5.json")
METRICS_PATH = Path(f"{ROOT}/ml_metrics_AUDUSD_admiral_v3_5.txt")

INSTRUMENT = "AUDUSD"
TRAIN_END = pd.Timestamp("2025-01-01", tz="UTC")
VAL_END = pd.Timestamp("2025-05-01", tz="UTC")
NON_FEATURES = {"instrument","ts","direction","outcome","pnl_usd","bars_to_exit"}


def prepare_data(df):
    df = df[df["outcome"].isin(["WIN", "LOSS"])].copy()
    df["target"] = (df["outcome"] == "WIN").astype(int)
    df = df.sort_values("ts").reset_index(drop=True)
    feat = [c for c in df.columns if c not in NON_FEATURES and c != "target"]
    X = df[feat].copy()
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
    return X, df["target"], feat


def main():
    print(f"=== TRAIN {INSTRUMENT} V3.5 ===")
    df = pd.read_parquet(DATASET_PATH)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").reset_index(drop=True)
    closed = df[df["outcome"].isin(["WIN","LOSS"])].copy()
    print(f"  Setups : {len(df)}, fermes : {len(closed)}, WR brut : {(closed['outcome']=='WIN').mean()*100:.1f}%")

    train_df = closed[closed["ts"] < TRAIN_END]
    val_df = closed[(closed["ts"] >= TRAIN_END) & (closed["ts"] < VAL_END)]
    test_df = closed[closed["ts"] >= VAL_END]
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test OOS: {len(test_df)}")

    X_train, y_train, feat = prepare_data(train_df)
    X_val, y_val, _ = prepare_data(val_df)
    X_test, y_test, _ = prepare_data(test_df)

    if len(X_train) < 50 or len(X_test) < 20:
        print(f"  ECHEC : pas assez de data (train={len(X_train)}, test={len(X_test)})")
        return

    model = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.04, max_depth=5,
        num_leaves=23, min_child_samples=30,
        subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.2, reg_lambda=0.3,
        random_state=42, verbosity=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)])

    metrics = []
    for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        proba = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, proba)
        line = f"\n{name.upper()} ({len(y)} trades) AUC={auc:.3f}"
        print(line)
        metrics.append(line)
        for seuil in [0.55, 0.60, 0.65, 0.70, 0.75]:
            mask = proba >= seuil
            if mask.sum() < 5:
                continue
            wr = y[mask].mean() * 100
            n = mask.sum()
            line = f"  seuil={seuil:.2f} : {n} trades, WR={wr:.1f}%"
            print(line)
            metrics.append(line)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    FEATURES_PATH.write_text(json.dumps({"features": feat}, indent=2))
    METRICS_PATH.write_text("\n".join(metrics), encoding="utf-8")
    print(f"\nModele : {MODEL_PATH}")


if __name__ == "__main__":
    main()
