"""Train ML v4 = modele qui apprend le STYLE du user (pas WIN/LOSS).

Difference avec ml_train.py :
- Target = label manuel "valid" du user (0/1)
- Joint le dataset Vizion (ml_dataset.parquet) avec les labels manuels
- Sauve dans ml_model_v4_user_style.pkl
"""
from __future__ import annotations

import pickle
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score

from bot_v2.manual_labels import load_labels


DATASET_PATH = Path("c:/Users/Shadow/TradingBot/data/ml_dataset.parquet")
V4_MODEL_PATH = Path("c:/Users/Shadow/TradingBot/bot_v2/ml_model_v4_user_style.pkl")


NON_FEATURES = {
    "instrument", "ts", "direction",
    "outcome", "pnl_usd", "bars_to_exit",
    "target",
}


def train_v4() -> dict:
    """Train LightGBM sur les labels manuels du user. Return metrics."""
    labels = load_labels()
    valid_labels = [lb for lb in labels if lb.label in ("valid", "invalid")]
    if len(valid_labels) < 20:
        return {"error": f"Pas assez de labels ({len(valid_labels)}<20)"}

    df_ds = pd.read_parquet(DATASET_PATH)

    # Index dataset par (ts iso, direction)
    df_ds["ts_iso"] = df_ds["ts"].apply(lambda t: pd.Timestamp(t).isoformat())

    # Map (ts, direction) -> label
    lbl_map = {(lb.ts, lb.direction): (1 if lb.label == "valid" else 0)
               for lb in valid_labels}

    df_ds["target"] = df_ds.apply(
        lambda row: lbl_map.get((row["ts_iso"], row["direction"])),
        axis=1,
    )
    df_train = df_ds[df_ds["target"].notna()].copy()
    df_train["target"] = df_train["target"].astype(int)

    if len(df_train) < 20:
        return {"error": f"Pas assez de matches dans dataset ({len(df_train)})"}

    feature_cols = [c for c in df_train.columns
                    if c not in NON_FEATURES and c != "ts_iso"]

    X = df_train[feature_cols].copy()
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
    y = df_train["target"]

    # Sur peu de labels on n'a pas de luxe de split, train sur tout
    # avec validation OOB via cross_val_score
    from sklearn.model_selection import cross_val_score

    n_valid = int(y.sum())
    n_invalid = int(len(y) - n_valid)

    if n_valid < 5 or n_invalid < 5:
        return {
            "error": f"Classes desequilibrees : {n_valid} valid / {n_invalid} invalid",
            "n_valid": n_valid,
            "n_invalid": n_invalid,
        }

    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=15,
        min_child_samples=3,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbosity=-1,
    )

    # Cross-val AUC (5-fold si possible)
    try:
        cv_auc = cross_val_score(model, X, y, cv=min(5, n_valid, n_invalid),
                                  scoring="roc_auc").mean()
    except Exception:
        cv_auc = None

    model.fit(X, y)

    # AUC train (info, sans valeur predictive)
    train_auc = float(roc_auc_score(y, model.predict_proba(X)[:, 1]))

    payload = {
        "model": model,
        "features": feature_cols,
        "n_labels_total": len(valid_labels),
        "n_valid": n_valid,
        "n_invalid": n_invalid,
        "train_auc": train_auc,
        "cv_auc": float(cv_auc) if cv_auc is not None else None,
    }
    V4_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(V4_MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)

    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    top_features = importance.head(10).to_dict()

    return {
        "n_labels": len(valid_labels),
        "n_valid": n_valid,
        "n_invalid": n_invalid,
        "train_auc": train_auc,
        "cv_auc": float(cv_auc) if cv_auc is not None else None,
        "top_features": top_features,
        "model_path": str(V4_MODEL_PATH),
    }


if __name__ == "__main__":
    import json
    result = train_v4()
    print(json.dumps(result, indent=2, default=str))
