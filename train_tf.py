"""Train un modele LightGBM sur un dataset specifique (multi-TF).

Usage : python train_tf.py M5
        python train_tf.py M15
        python train_tf.py M1   (= train standard)
"""
import sys
import json
import pickle
sys.path.insert(0, 'c:/Users/Shadow/TradingBot')

from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from bot_v2.ml_train import NON_FEATURES, prepare_data, temporal_split


def train_for_tf(tf: str):
    if tf == "M1":
        dataset_path = Path("c:/Users/Shadow/TradingBot/data/ml_dataset.parquet")
        model_path = Path("c:/Users/Shadow/TradingBot/bot_v2/ml_model.pkl")
        features_path = Path("c:/Users/Shadow/TradingBot/bot_v2/ml_features.json")
    else:
        dataset_path = Path(f"c:/Users/Shadow/TradingBot/data/ml_dataset_{tf}.parquet")
        model_path = Path(f"c:/Users/Shadow/TradingBot/bot_v2/ml_model_{tf}.pkl")
        features_path = Path(f"c:/Users/Shadow/TradingBot/bot_v2/ml_features_{tf}.json")

    print(f"\n{'='*60}")
    print(f"  TRAIN {tf} : {dataset_path.name}")
    print(f"{'='*60}\n", flush=True)

    if not dataset_path.exists():
        print(f"Dataset manquant : {dataset_path}", flush=True)
        return None

    df = pd.read_parquet(dataset_path)
    print(f"  {len(df)} lignes au total")
    df_closed = df[df["outcome"].isin(["WIN", "LOSS"])].sort_values("ts").reset_index(drop=True)
    print(f"  {len(df_closed)} trades fermes (WIN/LOSS)")
    if len(df_closed) < 50:
        print(f"  Pas assez de trades ({len(df_closed)}) pour entrainer.", flush=True)
        return None
    wr_brut = (df_closed["outcome"] == "WIN").mean() * 100
    print(f"  WR brut (sans ML) : {wr_brut:.1f}%")

    train_df, val_df, test_df = temporal_split(df_closed)
    print(f"\nSplit temporel :")
    print(f"  Train : {len(train_df):>5} | Val : {len(val_df):>5} | Test : {len(test_df):>5}")

    X_train, y_train, feat_cols = prepare_data(train_df)
    X_val, y_val, _ = prepare_data(val_df)
    X_test, y_test, _ = prepare_data(test_df)

    for X in (X_train, X_val, X_test):
        for c in X.columns:
            if X[c].dtype == bool:
                X[c] = X[c].astype(int)

    model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6, num_leaves=31,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbosity=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)])

    test_proba = model.predict_proba(X_test)[:, 1]
    auc_test = roc_auc_score(y_test, test_proba)
    print(f"\nAUC test : {auc_test:.3f}")

    for th in [0.45, 0.50, 0.55, 0.60]:
        mask = test_proba >= th
        n_kept = int(mask.sum())
        if n_kept < 3:
            continue
        wr = y_test[mask].mean() * 100
        print(f"  seuil {th:.2f} : {n_kept:>3} trades, WR {wr:.1f}%")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    features_path.write_text(json.dumps({"features": feat_cols, "categorical": []}, indent=2))
    print(f"\nModele sauve : {model_path.name}", flush=True)
    return {"tf": tf, "auc_test": auc_test, "n_test": len(y_test)}


if __name__ == "__main__":
    tf = sys.argv[1] if len(sys.argv) > 1 else "M1"
    train_for_tf(tf)
