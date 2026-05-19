"""Training LightGBM XAUUSD V3 - Admiral 8 ans, features enrichies.

V3 changes vs V2 (2026-05-19) :
- 28 features baseline (vire 7 constantes) + 8 nouvelles (temps/ATR/distance) = 36 features
- Hyperparams ameliores : reg+ accru, min_child_samples augmente pour moins d'overfit
- Output dedie : ml_model_XAUUSD_admiral_v3.pkl

Comparaison cible (V2 8 ans baseline) :
  TRAIN 1622 trades : seuil 0.55 -> WR 89.6% (AUC 0.910)
  VAL    541 trades : seuil 0.55 -> WR 52.2% (AUC 0.713)
  TEST   541 trades : seuil 0.55 -> WR 60.4% (AUC 0.748)
                    : seuil 0.65 -> WR 69.8% (43 trades)
"""
from __future__ import annotations

import sys
sys.path.insert(0, 'c:/Users/Shadow/TradingBot')

import json
import pickle
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score


DATASET_PATH = Path("c:/Users/Shadow/TradingBot/data/ml_dataset_XAUUSD_admiral_8ans_V3.parquet")
MODEL_PATH = Path("c:/Users/Shadow/TradingBot/bot_v2/ml_model_XAUUSD_admiral_v3.pkl")
FEATURES_PATH = Path("c:/Users/Shadow/TradingBot/bot_v2/ml_features_XAUUSD_admiral_v3.json")
METRICS_PATH = Path("c:/Users/Shadow/TradingBot/ml_metrics_XAUUSD_admiral_v3.txt")

INSTRUMENT = "XAUUSD"

NON_FEATURES = {
    "instrument", "ts", "direction",
    "outcome", "pnl_usd", "bars_to_exit",
}


def prepare_data(df):
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


def temporal_split(df, train_pct=0.6, val_pct=0.2):
    n = len(df)
    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def main():
    print(f"Chargement {DATASET_PATH}")
    df = pd.read_parquet(DATASET_PATH)
    df = df[df["instrument"] == INSTRUMENT].copy()
    print(f"  {len(df)} lignes {INSTRUMENT}")

    df_closed = df[df["outcome"].isin(["WIN", "LOSS"])].sort_values("ts").reset_index(drop=True)
    print(f"  {len(df_closed)} trades fermes (WIN/LOSS)")
    wr_brut = (df_closed["outcome"] == "WIN").mean() * 100
    print(f"  WR brut (sans ML) : {wr_brut:.1f}%")

    train_df, val_df, test_df = temporal_split(df_closed)
    print(f"\nSplit temporel :")
    print(f"  Train : {len(train_df):>5} trades ({train_df['ts'].min().date()} -> {train_df['ts'].max().date()})")
    print(f"  Val   : {len(val_df):>5} trades ({val_df['ts'].min().date()} -> {val_df['ts'].max().date()})")
    print(f"  Test  : {len(test_df):>5} trades ({test_df['ts'].min().date()} -> {test_df['ts'].max().date()})")

    X_train, y_train, feat_cols = prepare_data(train_df)
    X_val, y_val, _ = prepare_data(val_df)
    X_test, y_test, _ = prepare_data(test_df)

    cat_cols = []
    for c in feat_cols:
        if X_train[c].dtype == "object" or X_train[c].dtype.name == "category":
            X_train[c] = X_train[c].astype("category")
            X_val[c] = X_val[c].astype("category")
            X_test[c] = X_test[c].astype("category")
            cat_cols.append(c)

    print(f"\nFeatures : {len(feat_cols)}")
    print(f"  Categorielles : {cat_cols}")
    print(f"  Toutes : {feat_cols}")

    print("\nTraining LightGBM V3 (hyperparams ameliores)...")
    # V3 hyperparams : plus de regularisation, min_child_samples up
    # Pour reduire overfit (V2 train AUC 0.91 vs test 0.75 = gap 16 pts)
    model = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.04,           # was 0.05 -> un peu plus lent = moins greedy
        max_depth=5,                  # was 6 -> moins de profondeur = moins d'overfit
        num_leaves=23,                # was 31 -> 2^5-9 (plus serre)
        min_child_samples=30,         # was 20 -> + de samples par feuille = + robuste
        subsample=0.8,
        colsample_bytree=0.7,         # was 0.8 -> + d'aleatoire entre arbres
        reg_alpha=0.2,                # was 0.1 -> L1 doublee
        reg_lambda=0.3,               # was 0.1 -> L2 triplee
        random_state=42,
        verbosity=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)],
    )

    train_proba = model.predict_proba(X_train)[:, 1]
    val_proba = model.predict_proba(X_val)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]

    metrics_lines = []

    def report(name, y_true, y_proba):
        auc = roc_auc_score(y_true, y_proba)
        wr_baseline = y_true.mean() * 100
        for threshold in [0.50, 0.55, 0.60, 0.65, 0.70]:
            mask = y_proba >= threshold
            if mask.sum() < 5:
                continue
            kept_wr = y_true[mask].mean() * 100
            kept_n = mask.sum()
            line = f"  {name:6s} seuil={threshold:.2f} : {kept_n:>4} trades gardes, WR={kept_wr:.1f}% (vs baseline {wr_baseline:.1f}%)"
            print(line)
            metrics_lines.append(line)
        line = f"  {name:6s} AUC = {auc:.3f}"
        print(line)
        metrics_lines.append(line)
        print()

    print("\n=== METRIQUES PAR SET ===\n")
    metrics_lines.append("=== METRIQUES PAR SET ===\n")
    metrics_lines.append(f"\nTRAIN ({len(y_train)} trades)")
    print(f"\nTRAIN ({len(y_train)} trades)")
    report("train", y_train.values, train_proba)
    metrics_lines.append(f"\nVAL ({len(y_val)} trades)")
    print(f"\nVAL ({len(y_val)} trades)")
    report("val", y_val.values, val_proba)
    metrics_lines.append(f"\nTEST ({len(y_test)} trades) - jamais vu en train")
    print(f"\nTEST ({len(y_test)} trades) - jamais vu en train")
    report("test", y_test.values, test_proba)

    print("\n=== TOP 20 FEATURES IMPORTANCE ===")
    metrics_lines.append("\n=== TOP 20 FEATURES IMPORTANCE ===")
    importance = pd.Series(model.feature_importances_, index=feat_cols).sort_values(ascending=False)
    for fname, imp in importance.head(20).items():
        line = f"  {fname:30s} {int(imp)}"
        print(line)
        metrics_lines.append(line)

    print("\n=== FEATURES PEU UTILISEES (importance 0-2) ===")
    metrics_lines.append("\n=== FEATURES PEU UTILISEES ===")
    low_imp = importance[importance <= 2]
    for fname, imp in low_imp.items():
        line = f"  {fname:30s} {int(imp)}"
        print(line)
        metrics_lines.append(line)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModele sauve : {MODEL_PATH}")

    FEATURES_PATH.write_text(json.dumps({
        "features": feat_cols,
        "categorical": cat_cols,
    }, indent=2))
    print(f"Features sauve : {FEATURES_PATH}")

    METRICS_PATH.write_text("\n".join(metrics_lines), encoding="utf-8")
    print(f"Metriques : {METRICS_PATH}")


if __name__ == "__main__":
    main()
