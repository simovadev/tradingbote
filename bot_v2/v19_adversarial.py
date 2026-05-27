"""V19 ADVERSARIAL VALIDATION : check drift train vs OOS sur les features V19.

On entraine un classifieur a distinguer "trade vient de TRAIN" vs "trade vient de OOS".
Si AUC > 0.65 -> drift severe -> mauvais signe pour live.
Si AUC < 0.55 -> distributions similaires -> bon signe.

Difference avec V18 adversarial : on inclut les sequences brutes (m1/m15/h1)
en plus des features ICT (en aplatissant) pour voir si le DL peut aussi distinguer
les seqs train/OOS.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = "c:/Users/Shadow/TradingBot" if sys.platform == "win32" else "/workspace/TradingBot"
sys.path.insert(0, ROOT)


TRAIN_END = pd.Timestamp("2025-05-20").to_datetime64()
OOS_START = pd.Timestamp("2025-11-22").to_datetime64()


def main():
    print("=== V19 ADVERSARIAL VALIDATION ===")
    print(f"Train period : ... -> {TRAIN_END}")
    print(f"OOS period   : {OOS_START} -> ...")
    print()

    # Charge sequences
    npz = np.load(Path(f"{ROOT}/data/v19_sequences.npz"), allow_pickle=True)
    m1 = npz["m1_seqs"]
    ict = npz["ict_feats"]
    aid = npz["asset_ids"]
    ts = np.array(npz["timestamps"], dtype="datetime64[ns]")

    train_mask = ts < TRAIN_END
    oos_mask = ts >= OOS_START

    print(f"Train : {train_mask.sum()} samples")
    print(f"OOS   : {oos_mask.sum()} samples")

    # Construit features pour le classifieur adversarial :
    # - ICT features (50)
    # - Statistiques sur m1_seqs : mean/std/min/max par channel = 20 features
    m1_train = m1[train_mask]
    m1_oos = m1[oos_mask]
    ict_train = ict[train_mask]
    ict_oos = ict[oos_mask]
    aid_train = aid[train_mask]
    aid_oos = aid[oos_mask]

    def seq_stats(seqs):
        # seqs : (N, 240, 5)
        return np.concatenate([
            seqs.mean(axis=1),  # (N, 5)
            seqs.std(axis=1),   # (N, 5)
            seqs.min(axis=1),   # (N, 5)
            seqs.max(axis=1),   # (N, 5)
        ], axis=1)  # (N, 20)

    stats_train = seq_stats(m1_train)
    stats_oos = seq_stats(m1_oos)

    # Concat ICT + stats + asset_id one-hot
    n_assets = int(max(aid.max(), 27)) + 1
    aid_onehot_train = np.eye(n_assets)[aid_train]
    aid_onehot_oos = np.eye(n_assets)[aid_oos]

    X_train = np.concatenate([ict_train, stats_train, aid_onehot_train], axis=1)
    X_oos = np.concatenate([ict_oos, stats_oos, aid_onehot_oos], axis=1)

    print(f"Features adversarial : {X_train.shape[1]}")

    # Sous-echantillonne train pour equilibrer (max 3:1 ratio)
    n_oos = len(X_oos)
    if len(X_train) > n_oos * 3:
        idx_sample = np.random.RandomState(42).choice(len(X_train), n_oos * 3, replace=False)
        X_train = X_train[idx_sample]

    y_train_lbl = np.zeros(len(X_train))
    y_oos_lbl = np.ones(len(X_oos))

    X_all = np.concatenate([X_train, X_oos])
    y_all = np.concatenate([y_train_lbl, y_oos_lbl])

    X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.3,
                                                random_state=42, stratify=y_all)

    model = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.02,
        max_depth=6, num_leaves=31, min_child_samples=50,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=-1, n_jobs=8,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)],
              callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)])
    pred = model.predict_proba(X_te)[:, 1]
    auc = float(roc_auc_score(y_te, pred))

    if auc < 0.55:
        diag = "OK"
        comment = "Distributions train/OOS similaires - V19 va probablement bien generaliser"
    elif auc < 0.65:
        diag = "WARN"
        comment = "Drift modere - monitoring necessaire en live"
    elif auc < 0.75:
        diag = "DRIFT"
        comment = "Drift significatif - shadow demo obligatoire avant live"
    else:
        diag = "SEVERE"
        comment = "Drift severe - NE PAS DEPLOYER (donnees train/OOS trop differentes)"

    print(f"\nAdversarial AUC : {auc:.4f}")
    print(f"Diagnostic      : {diag}")
    print(f"Comment         : {comment}")

    # Top features qui distinguent
    n_ict = ict.shape[1]
    n_stats = 20
    feat_names = (
        [f"ict_{i}" for i in range(n_ict)] +
        [f"stat_{i}" for i in range(n_stats)] +
        [f"asset_{i}" for i in range(n_assets)]
    )
    imp = pd.DataFrame({"feature": feat_names, "importance": model.feature_importances_})
    imp = imp.sort_values("importance", ascending=False).head(15)
    print(f"\nTop 15 features qui DRIFT le plus :")
    for _, row in imp.iterrows():
        print(f"  {row['feature']:<25} {row['importance']}")

    result = {
        "adversarial_auc": auc,
        "diagnostic": diag,
        "comment": comment,
        "n_train": int(train_mask.sum()),
        "n_oos": int(oos_mask.sum()),
        "top_drift": imp.to_dict("records"),
    }
    out = Path(f"{ROOT}/v19_adversarial_recap.json")
    out.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nSaved : {out.name}")


if __name__ == "__main__":
    main()
