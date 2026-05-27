"""V20 ADVERSARIAL : valide qu'il n'y a pas de drift entre train et OOS.

Test : un classifieur peut-il distinguer train vs OOS uniquement avec les features
(sans regarder le label) ? Si oui -> les distributions divergent dans le temps
-> le model peut tricher en reconnaissant l'epoque.

- AUC < 0.55 : CLEAN (distributions identiques, ICT vraiment intemporel)
- 0.55-0.65 : LEGER DRIFT (peu d'impact)
- > 0.65 : SEVERE (le model peut sur-apprendre l'epoque)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path("/workspace/TradingBot") if sys.platform == "linux" else Path("c:/Users/Shadow/TradingBot")

TRAIN_END = pd.Timestamp("2025-05-20").to_datetime64()
OOS_START = pd.Timestamp("2025-11-22").to_datetime64()


def load_chunks(chunks_dir):
    """Concatene tous les chunks."""
    parts = {"ict_feats": [], "labels": [], "timestamps": []}
    for c in sorted(chunks_dir.glob("*.npz")):
        d = np.load(c, allow_pickle=True)
        for k in parts:
            parts[k].append(d[k])
    return (np.concatenate(parts["ict_feats"]),
            np.concatenate(parts["labels"]),
            np.array(np.concatenate(parts["timestamps"]), dtype="datetime64[ns]"))


def main():
    chunks_dir = Path("/dev/shm/v20_chunks") if Path("/dev/shm/v20_chunks").exists() else ROOT / "data" / "v20_chunks"
    print(f"Chunks dir : {chunks_dir}")
    ict, lbl, ts = load_chunks(chunks_dir)
    print(f"Total samples : {len(lbl):,}")

    train_mask = ts < TRAIN_END
    oos_mask = ts >= OOS_START
    print(f"Train : {train_mask.sum():,}")
    print(f"OOS   : {oos_mask.sum():,}")

    # Sample pour rapidite (50k train + 50k oos)
    rng = np.random.default_rng(42)
    n_sample = 50000
    train_idx = rng.choice(np.where(train_mask)[0], size=min(n_sample, train_mask.sum()), replace=False)
    oos_idx = rng.choice(np.where(oos_mask)[0], size=min(n_sample, oos_mask.sum()), replace=False)

    X = np.concatenate([ict[train_idx], ict[oos_idx]])
    # Label adversarial : 0 = train, 1 = oos
    y_adv = np.concatenate([np.zeros(len(train_idx)), np.ones(len(oos_idx))])

    print(f"\nAdversarial sample : {len(X):,}")
    print(f"Features : {X.shape[1]}")

    # Shuffle + split
    order = rng.permutation(len(X))
    X = X[order]
    y_adv = y_adv[order]
    split = int(0.7 * len(X))
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y_adv[:split], y_adv[split:]

    # Train GBM rapide
    print("\nTraining adversarial classifier (GBM)...")
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)[:, 1]
    adv_auc = roc_auc_score(y_te, proba)

    if adv_auc < 0.55:
        verdict = "CLEAN (distributions train/OOS identiques, ICT intemporel)"
    elif adv_auc < 0.65:
        verdict = "LEGER DRIFT (peu d'impact, acceptable)"
    else:
        verdict = "SEVERE DRIFT (model peut cheat sur l'epoque)"

    print(f"\n=== ADVERSARIAL AUC : {adv_auc:.4f} ===")
    print(f"Verdict : {verdict}")

    # Top features qui distinguent train/oos
    importances = clf.feature_importances_
    top5 = np.argsort(importances)[-5:][::-1]
    print(f"\nTop 5 features qui distinguent train vs OOS :")
    for i in top5:
        print(f"  Feature #{i} : importance={importances[i]:.4f}")

    recap = {
        "adversarial_auc": float(adv_auc),
        "verdict": verdict,
        "n_train_sample": len(train_idx),
        "n_oos_sample": len(oos_idx),
        "n_features": int(X.shape[1]),
        "top5_feature_indices": [int(i) for i in top5],
        "top5_feature_importances": [float(importances[i]) for i in top5],
    }
    (ROOT / "v20_adversarial_recap.json").write_text(json.dumps(recap, indent=2))
    print(f"\nSaved : v20_adversarial_recap.json")


if __name__ == "__main__":
    main()
