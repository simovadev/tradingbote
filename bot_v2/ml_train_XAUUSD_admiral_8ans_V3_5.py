"""Training LightGBM XAUUSD V3.5 - 30 chunks Admiral, filtres relaches.

Split temporel CUSTOM (user 2026-05-19) :
- TRAIN : 2018-12 -> 2025-01 (~6 ans, incluant 2024)
- VAL   : 2025-01 -> 2025-05 (4 mois)
- TEST  : 2025-05 -> 2025-10 (5 mois OOS recents)

Filtres relaches V3.5 :
- session_sans_direction : malus -8
- pas de parent OB HTF   : malus -15
- pas de grand-parent OB : malus -5
- mauvaise zone P/D      : malus -10
- pas de FVG sync        : malus -5
+ 5 features ML pour apprendre l'impact reel

Hyperparams V3 (gardes identiques pour comparaison clean V3 vs V3.5)

Output :
- bot_v2/ml_model_XAUUSD_admiral_v3_5.pkl
- bot_v2/ml_features_XAUUSD_admiral_v3_5.json
- ml_metrics_XAUUSD_admiral_v3_5.txt
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


PARTIAL_DIR = Path("c:/Users/Shadow/TradingBot/data/ml_partial_M1")
MODEL_PATH = Path("c:/Users/Shadow/TradingBot/bot_v2/ml_model_XAUUSD_admiral_v3_5.pkl")
FEATURES_PATH = Path("c:/Users/Shadow/TradingBot/bot_v2/ml_features_XAUUSD_admiral_v3_5.json")
METRICS_PATH = Path("c:/Users/Shadow/TradingBot/ml_metrics_XAUUSD_admiral_v3_5.txt")

INSTRUMENT = "XAUUSD"

# Split custom user 2026-05-19 - train jusqu'a 2025
TRAIN_END = pd.Timestamp("2025-01-01", tz="UTC")
VAL_END = pd.Timestamp("2025-05-01", tz="UTC")

NON_FEATURES = {
    "instrument", "ts", "direction",
    "outcome", "pnl_usd", "bars_to_exit",
}


def load_all_chunks():
    """Charge tous les chunks V3.5 disponibles."""
    chunks = sorted(PARTIAL_DIR.glob("XAUUSD_M1_*.parquet"))
    print(f"Chargement {len(chunks)} chunks V3.5...")
    all_dfs = [pd.read_parquet(c) for c in chunks]
    df = pd.concat(all_dfs, ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").reset_index(drop=True)
    return df


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


def main():
    print(f"=== TRAIN XAUUSD V3.5 (split custom) ===\n")
    df = load_all_chunks()
    print(f"  Total setups : {len(df):,}")
    print(f"  Periode : {df['ts'].min()} -> {df['ts'].max()}")

    df_closed = df[df["outcome"].isin(["WIN", "LOSS"])].copy()
    df_closed = df_closed.sort_values("ts").reset_index(drop=True)
    print(f"  Trades fermes : {len(df_closed):,}")
    wr_brut = (df_closed["outcome"] == "WIN").mean() * 100
    print(f"  WR brut       : {wr_brut:.1f}%")

    # Split CUSTOM par dates
    train_df = df_closed[df_closed["ts"] < TRAIN_END].copy()
    val_df = df_closed[(df_closed["ts"] >= TRAIN_END) & (df_closed["ts"] < VAL_END)].copy()
    test_df = df_closed[df_closed["ts"] >= VAL_END].copy()

    print(f"\nSplit temporel CUSTOM :")
    print(f"  TRAIN : {len(train_df):>6,} trades ({train_df['ts'].min().date()} -> {train_df['ts'].max().date()})")
    print(f"  VAL   : {len(val_df):>6,} trades ({val_df['ts'].min().date()} -> {val_df['ts'].max().date()})")
    print(f"  TEST  : {len(test_df):>6,} trades ({test_df['ts'].min().date()} -> {test_df['ts'].max().date()})")

    train_wr = (train_df["outcome"] == "WIN").mean() * 100
    val_wr = (val_df["outcome"] == "WIN").mean() * 100
    test_wr = (test_df["outcome"] == "WIN").mean() * 100
    print(f"\n  WR baseline TRAIN : {train_wr:.1f}%")
    print(f"  WR baseline VAL   : {val_wr:.1f}%")
    print(f"  WR baseline TEST  : {test_wr:.1f}%")

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
    print(f"  {feat_cols}")

    print("\nTraining LightGBM V3.5...")
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

    train_proba = model.predict_proba(X_train)[:, 1]
    val_proba = model.predict_proba(X_val)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]

    metrics_lines = []

    def report(name, y_true, y_proba):
        auc = roc_auc_score(y_true, y_proba)
        wr_baseline = y_true.mean() * 100
        n_days_test = 365  # approximation pour trades/jour
        for threshold in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
            mask = y_proba >= threshold
            if mask.sum() < 5:
                continue
            kept_wr = y_true[mask].mean() * 100
            kept_n = mask.sum()
            line = f"  {name:6s} seuil={threshold:.2f} : {kept_n:>5} trades, WR={kept_wr:.1f}% (vs baseline {wr_baseline:.1f}%)"
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
    metrics_lines.append(f"\nTEST OOS ({len(y_test)} trades) - 2025+")
    print(f"\nTEST OOS ({len(y_test)} trades) - 2025+")
    report("test", y_test.values, test_proba)

    print("\n=== TOP 20 FEATURES IMPORTANCE ===")
    metrics_lines.append("\n=== TOP 20 FEATURES IMPORTANCE ===")
    importance = pd.Series(model.feature_importances_, index=feat_cols).sort_values(ascending=False)
    for fname, imp in importance.head(20).items():
        line = f"  {fname:30s} {int(imp)}"
        print(line)
        metrics_lines.append(line)

    print("\n=== FEATURES PEU UTILISEES (importance <= 3) ===")
    metrics_lines.append("\n=== FEATURES PEU UTILISEES ===")
    low_imp = importance[importance <= 3]
    for fname, imp in low_imp.items():
        line = f"  {fname:30s} {int(imp)}"
        print(line)
        metrics_lines.append(line)

    # Profit estime sur TEST OOS
    print("\n=== PROFIT ESTIME TEST OOS @ seuil 0.70 (compte 100, risk 5%, RR=2) ===")
    seuil = 0.70
    mask = test_proba >= seuil
    if mask.sum() > 5:
        n_wins = (y_test.values[mask] == 1).sum()
        n_loss = (y_test.values[mask] == 0).sum()
        wr_seuil = n_wins / (n_wins + n_loss) * 100
        # +10% par WIN, -5% par LOSS (compound non inclus)
        profit_pct = n_wins * 10 - n_loss * 5
        n_days_test = (test_df["ts"].max() - test_df["ts"].min()).days
        line = (
            f"  Trades : {n_wins + n_loss} | WIN={n_wins} | LOSS={n_loss}\n"
            f"  WR @ 0.70 : {wr_seuil:.1f}%\n"
            f"  Profit theorique sur {n_days_test} jours OOS : +{profit_pct}%\n"
            f"  Moyenne : +{profit_pct/n_days_test:.2f}% par jour"
        )
        print(line)
        metrics_lines.append("\n" + line)

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
