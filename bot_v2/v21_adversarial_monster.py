"""V21 MONSTER : adversarial validation.

Test : un classifieur peut-il distinguer train vs OOS uniquement avec les bougies
brutes (focus M1) ? Si oui -> drift = le model peut tricher en reconnaissant
l'epoque (pas vraiment "ICT universel intemporel").

Cible : AUC < 0.65 (acceptable)
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


def main():
    chunks_dir = Path("/dev/shm/v21_chunks")
    print(f"Loading chunks {chunks_dir}", flush=True)
    chunks = sorted(chunks_dir.glob("*.npz"))
    m1_list, ts_list = [], []
    for c in chunks:
        d = np.load(c, allow_pickle=True)
        m1_list.append(d["m1_seqs"])
        ts_list.append(d["timestamps"])
    m1 = np.concatenate(m1_list)
    ts = np.array(np.concatenate(ts_list), dtype="datetime64[ns]")
    print(f"Total: {len(ts):,} samples")

    train_mask = ts < TRAIN_END
    oos_mask = ts >= OOS_START
    print(f"Train : {train_mask.sum():,}")
    print(f"OOS   : {oos_mask.sum():,}")

    # Features : on flatten les bougies M1 (240 * 5 = 1200 features)
    # Trop pour GBM rapide -> on garde juste close (240 features)
    # En plus on subsample 30k de chaque
    rng = np.random.default_rng(42)
    n_sample = 30000
    train_idx = rng.choice(np.where(train_mask)[0], size=min(n_sample, train_mask.sum()), replace=False)
    oos_idx = rng.choice(np.where(oos_mask)[0], size=min(n_sample, oos_mask.sum()), replace=False)

    # On utilise les closes M1 (240 points par sample) + stats
    closes_train = m1[train_idx, :, 3]  # (N, 240)
    closes_oos = m1[oos_idx, :, 3]

    # Features compactes : moyennes par bucket de 24 + std + min/max
    def compact(x):
        # x: (N, 240)
        b = x.reshape(x.shape[0], 10, 24).mean(axis=2)  # (N, 10) moyennes par bucket de 24
        s = x.std(axis=1, keepdims=True)
        mn = x.min(axis=1, keepdims=True)
        mx = x.max(axis=1, keepdims=True)
        return np.concatenate([b, s, mn, mx], axis=1)

    X_train = compact(closes_train)
    X_oos = compact(closes_oos)

    X = np.concatenate([X_train, X_oos])
    y_adv = np.concatenate([np.zeros(len(X_train)), np.ones(len(X_oos))])
    print(f"Adversarial sample : {len(X):,}, features = {X.shape[1]}")

    order = rng.permutation(len(X))
    X = X[order]
    y_adv = y_adv[order]
    split = int(0.7 * len(X))
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y_adv[:split], y_adv[split:]

    print("Training adversarial GBM...", flush=True)
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)[:, 1]
    adv_auc = roc_auc_score(y_te, proba)

    if adv_auc < 0.55:
        verdict = "CLEAN (train/OOS identiques, model genuine universel)"
    elif adv_auc < 0.65:
        verdict = "LEGER DRIFT (acceptable)"
    else:
        verdict = "SEVERE DRIFT (model peut cheat sur l'epoque)"

    print(f"\n=== ADVERSARIAL AUC : {adv_auc:.4f} ===")
    print(f"Verdict : {verdict}")

    recap = {
        "adversarial_auc": float(adv_auc),
        "verdict": verdict,
        "n_train_sample": int(len(X_train)),
        "n_oos_sample": int(len(X_oos)),
        "n_features": int(X.shape[1]),
    }
    (ROOT / "v21_monster_adversarial_recap.json").write_text(json.dumps(recap, indent=2))
    print(f"\nRecap : v21_monster_adversarial_recap.json")


if __name__ == "__main__":
    main()
