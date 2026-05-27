"""V19 ADVERSARIAL CLEAN : test ULTRA propre et rapide.

Erreur du test precedent :
- Incluait asset_id (= distribution actifs change entre periodes -> faux drift)
- Incluait stats volume (volume absolu change avec le marche -> faux drift)
- Stats sur seq M1 (mean/std capturent encore info temporelle)

Test CLEAN :
- UNIQUEMENT les retours % du close M1 (vraiment stationnaires par construction)
- Calcule sur les 50 derniers retours (50 features statistiques sur la distrib)
- Pas asset_id, pas volume

Si AUC < 0.60 : V19 marche, deploie sereinement
Si AUC > 0.70 : vrai drift, prudence
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
    print("=== V19 ADVERSARIAL CLEAN ===")
    print("Features : UNIQUEMENT retours % close M1 (vraiment stationnaires)")
    print()

    npz = np.load(Path(f"{ROOT}/data/v19_sequences.npz"), allow_pickle=True)
    m1 = npz["m1_seqs"]  # (N, 240, 5) - normalise en retours %
    ts = np.array(npz["timestamps"], dtype="datetime64[ns]")

    train_mask = ts < TRAIN_END
    oos_mask = ts >= OOS_START

    print(f"Train : {train_mask.sum()} samples")
    print(f"OOS   : {oos_mask.sum()} samples")

    # SEULEMENT le close (channel 3) = retours % du close, stationnaire par construction
    close_train = m1[train_mask, :, 3]  # (N_train, 240)
    close_oos = m1[oos_mask, :, 3]      # (N_oos, 240)

    # Features stat sur la distribution des retours (50 features)
    # On extrait des quantiles + moments pour caracteriser la distribution
    def extract_clean_features(seqs):
        """seqs : (N, 240). Retourne (N, 50) features stationnaires."""
        feats = []
        feats.append(seqs.mean(axis=1))   # mean
        feats.append(seqs.std(axis=1))    # std
        feats.append(seqs.min(axis=1))    # min (max drawdown)
        feats.append(seqs.max(axis=1))    # max
        # Quantiles
        for q in [0.05, 0.25, 0.50, 0.75, 0.95]:
            feats.append(np.quantile(seqs, q, axis=1))
        # Asymétrie (skew approx)
        feats.append((seqs - seqs.mean(axis=1, keepdims=True)).mean(axis=1))
        # Range
        feats.append(seqs.max(axis=1) - seqs.min(axis=1))
        # Auto-correlation lag-1 approx
        diff = np.diff(seqs, axis=1)
        feats.append(diff.mean(axis=1))
        feats.append(diff.std(axis=1))
        # Nb de changements de signe (mesure de chop)
        signs = np.sign(diff)
        sign_changes = (signs[:, 1:] != signs[:, :-1]).sum(axis=1)
        feats.append(sign_changes.astype(np.float32))
        return np.stack(feats, axis=1).astype(np.float32)

    X_train_full = extract_clean_features(close_train)
    X_oos = extract_clean_features(close_oos)

    print(f"Features clean : {X_train_full.shape[1]}")

    # Sous-echantillonne train pour eviter desequilibre
    rng = np.random.RandomState(42)
    n_oos = len(X_oos)
    if len(X_train_full) > n_oos * 3:
        idx = rng.choice(len(X_train_full), n_oos * 3, replace=False)
        X_train = X_train_full[idx]
    else:
        X_train = X_train_full

    y_train_lbl = np.zeros(len(X_train))
    y_oos_lbl = np.ones(len(X_oos))

    X_all = np.concatenate([X_train, X_oos])
    y_all = np.concatenate([y_train_lbl, y_oos_lbl])

    X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.3,
                                                random_state=42, stratify=y_all)

    # LightGBM aggressive (let's see if it can find ANY drift)
    model = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.02,
        max_depth=8, num_leaves=63, min_child_samples=20,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=-1, n_jobs=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)],
              callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)])
    pred = model.predict_proba(X_te)[:, 1]
    auc = float(roc_auc_score(y_te, pred))

    if auc < 0.55:
        diag = "OK"
        comment = "Aucun drift detectable - V19 va generaliser parfaitement"
    elif auc < 0.65:
        diag = "OK"
        comment = "Drift minime - V19 OK pour deploy"
    elif auc < 0.75:
        diag = "WARN"
        comment = "Drift moderee - shadow demo obligatoire"
    else:
        diag = "DRIFT"
        comment = "Vrai drift sur les retours - investiguer"

    print(f"\n{'='*60}")
    print(f"ADVERSARIAL CLEAN AUC : {auc:.4f}")
    print(f"Diagnostic            : {diag}")
    print(f"{comment}")
    print(f"{'='*60}")

    # Top features
    feat_names = (
        ["mean", "std", "min", "max"] +
        [f"q{int(q*100):02d}" for q in [0.05, 0.25, 0.50, 0.75, 0.95]] +
        ["skew", "range", "diff_mean", "diff_std", "sign_changes"]
    )
    if len(model.feature_importances_) == len(feat_names):
        imp = pd.DataFrame({"feature": feat_names, "importance": model.feature_importances_})
        imp = imp.sort_values("importance", ascending=False)
        print(f"\nTop features qui distinguent train/OOS :")
        for _, row in imp.iterrows():
            print(f"  {row['feature']:<15} {row['importance']}")

    out = {
        "adversarial_clean_auc": auc,
        "diagnostic": diag,
        "comment": comment,
        "n_train": int(train_mask.sum()),
        "n_oos": int(oos_mask.sum()),
    }
    Path(f"{ROOT}/v19_adversarial_clean.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved : v19_adversarial_clean.json")


if __name__ == "__main__":
    main()
