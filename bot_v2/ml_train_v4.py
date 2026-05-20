"""Train LightGBM V4 paramétré par actif.

V4 = entraine sur dataset V4 (displacement dynamique selon volatilite).
Output :
- bot_v2/ml_model_{ASSET}_admiral_v4.pkl
- bot_v2/ml_features_{ASSET}_admiral_v4.json
- ml_metrics_{ASSET}_admiral_v4.txt

Usage :
    python -m bot_v2.ml_train_v4 XAUUSD
    python -m bot_v2.ml_train_v4 --all
"""
from __future__ import annotations

import json
import pickle
import sys
import argparse
from pathlib import Path

ROOT = "c:/Users/Shadow/TradingBot" if sys.platform == "win32" else "/workspace/TradingBot"
sys.path.insert(0, ROOT)

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score


# Split temporel custom (idem V3.5 pour comparaison)
TRAIN_END = pd.Timestamp("2025-01-01", tz="UTC")
VAL_END = pd.Timestamp("2025-05-01", tz="UTC")

NON_FEATURES = {
    "instrument", "ts", "direction",
    "outcome", "pnl_usd", "bars_to_exit",
}

ALL_ASSETS = [
    "XAUUSD", "NAS100", "GER40", "BTCUSD",
    "EURUSD", "GBPUSD", "AUDUSD", "USDJPY",
    "SP500", "DJ30", "UK100", "FRA40",
    "USDCAD", "USDCHF",
]


def prepare_data(df, feat_cols=None):
    df = df[df["outcome"].isin(["WIN", "LOSS"])].copy()
    df["target"] = (df["outcome"] == "WIN").astype(int)
    df = df.sort_values("ts").reset_index(drop=True)
    if feat_cols is None:
        feat_cols = [c for c in df.columns if c not in NON_FEATURES and c != "target"]
    X = df[feat_cols].copy()
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
    y = df["target"]
    return X, y, feat_cols


def train_one(asset: str):
    dataset_path = Path(f"{ROOT}/data/ml_dataset_{asset}_admiral_8ans_V4.parquet")
    model_path = Path(f"{ROOT}/bot_v2/ml_model_{asset}_admiral_v4.pkl")
    features_path = Path(f"{ROOT}/bot_v2/ml_features_{asset}_admiral_v4.json")
    metrics_path = Path(f"{ROOT}/ml_metrics_{asset}_admiral_v4.txt")

    if not dataset_path.exists():
        print(f"!! Dataset manquant : {dataset_path}")
        return

    print(f"\n=== TRAIN {asset} V4 ===")
    df = pd.read_parquet(dataset_path)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").reset_index(drop=True)
    df_closed = df[df["outcome"].isin(["WIN", "LOSS"])].copy()
    print(f"  Total setups closed : {len(df_closed):,}")
    print(f"  Periode : {df_closed['ts'].min()} -> {df_closed['ts'].max()}")
    wr_brut = (df_closed["outcome"] == "WIN").mean() * 100
    print(f"  WR brut : {wr_brut:.1f}%")

    train_df = df_closed[df_closed["ts"] < TRAIN_END].copy()
    val_df = df_closed[(df_closed["ts"] >= TRAIN_END) & (df_closed["ts"] < VAL_END)].copy()
    test_df = df_closed[df_closed["ts"] >= VAL_END].copy()

    if len(train_df) < 100 or len(test_df) < 50:
        print(f"!! Pas assez de data pour {asset} (train={len(train_df)} test={len(test_df)})")
        return

    print(f"  TRAIN : {len(train_df):>6,} | VAL : {len(val_df):>6,} | TEST : {len(test_df):>6,}")

    X_train, y_train, feat_cols = prepare_data(train_df)
    X_val, y_val, _ = prepare_data(val_df, feat_cols)
    X_test, y_test, _ = prepare_data(test_df, feat_cols)

    cat_cols = []
    for c in feat_cols:
        if X_train[c].dtype == "object" or X_train[c].dtype.name == "category":
            X_train[c] = X_train[c].astype("category")
            X_val[c] = X_val[c].astype("category")
            X_test[c] = X_test[c].astype("category")
            cat_cols.append(c)

    print(f"  Features : {len(feat_cols)}")

    model = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.04,
        max_depth=5,
        num_leaves=23,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.2,
        reg_lambda=0.3,
        random_state=42,
        verbosity=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)],
    )

    test_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, test_proba)
    metrics_lines = [f"=== {asset} V4 ==="]
    metrics_lines.append(f"AUC TEST = {auc:.3f}")
    print(f"  AUC TEST = {auc:.3f}")

    for threshold in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        mask = test_proba >= threshold
        if mask.sum() < 5:
            continue
        wr = y_test.values[mask].mean() * 100
        metrics_lines.append(f"  seuil={threshold:.2f} : {mask.sum():>5} trades, WR={wr:.1f}%")
        print(f"  seuil={threshold:.2f} : {mask.sum():>5} trades, WR={wr:.1f}%")

    # Save
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    features_path.write_text(json.dumps({"features": feat_cols, "categorical": cat_cols}, indent=2))
    metrics_path.write_text("\n".join(metrics_lines))
    print(f"  Sauve : {model_path.name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("asset", nargs="?", help="Asset name ou --all")
    p.add_argument("--all", action="store_true")
    args = p.parse_args()

    if args.all or args.asset == "--all":
        for a in ALL_ASSETS:
            try:
                train_one(a)
            except Exception as e:
                print(f"!! FAIL {a} : {e}")
    elif args.asset:
        train_one(args.asset)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
