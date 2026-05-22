"""Train V10 TEST : compare configs hyperparametres LightGBM sur dataset test.

Charge les datasets V10 test (3 actifs, 1 an), split train/OOS, et teste
plusieurs configs pour voir laquelle donne le plus de trades a WR maintenu.

Usage : python tools/train_v10_test.py [suffix_dataset]
  suffix = rr20 ou rr15 (defaut rr20)
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TEST_ASSETS = ["XAUUSD", "EURUSD", "NAS100"]
NON_FEATURES = {"instrument", "ts", "direction", "outcome", "pnl_usd", "bars_to_exit", "target"}

# Configs a tester
CONFIGS = {
    "V9_baseline": dict(n_estimators=300, learning_rate=0.05, max_depth=6,
                        num_leaves=31, min_child_samples=20, subsample=0.8,
                        colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1),
    "deeper": dict(n_estimators=500, learning_rate=0.03, max_depth=9,
                   num_leaves=63, min_child_samples=30, subsample=0.8,
                   colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1),
    "balanced": dict(n_estimators=400, learning_rate=0.04, max_depth=7,
                     num_leaves=47, min_child_samples=25, subsample=0.8,
                     colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
                     class_weight="balanced"),
    "more_trees": dict(n_estimators=800, learning_rate=0.02, max_depth=7,
                       num_leaves=47, min_child_samples=20, subsample=0.85,
                       colsample_bytree=0.85, reg_alpha=0.05, reg_lambda=0.05),
}


def load_ds(asset, suffix):
    p = ROOT / f"data/ml_dataset_{asset}_v10test_{suffix}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df[df["outcome"].isin(["WIN", "LOSS"])].sort_values("ts").reset_index(drop=True)


def main():
    suffix = sys.argv[1] if len(sys.argv) > 1 else "rr20"
    print(f"=== TRAIN V10 TEST | dataset suffix={suffix} ===\n")

    for asset in TEST_ASSETS:
        df = load_ds(asset, suffix)
        if df is None or len(df) < 200:
            print(f"{asset} : dataset absent ou trop court, skip\n")
            continue

        # Split 75% train / 25% OOS temporel
        cut = df["ts"].quantile(0.75)
        tr = df[df["ts"] < cut].copy()
        oos = df[df["ts"] >= cut].copy()
        feats = [c for c in df.columns if c not in NON_FEATURES]
        for sub in (tr, oos):
            for c in feats:
                if sub[c].dtype == bool:
                    sub[c] = sub[c].astype(int)
        Xtr, ytr = tr[feats], (tr["outcome"] == "WIN").astype(int)
        Xoos, yoos = oos[feats], (oos["outcome"] == "WIN").astype(int)

        print(f"### {asset}  (train {len(tr)}, OOS {len(oos)}, {len(feats)} features)")
        print(f"  {'config':14s} {'AUC':>7s} {'>=0.6 n/WR':>14s} {'>=0.65 n/WR':>14s} {'>=0.70 n/WR':>14s}")
        for name, params in CONFIGS.items():
            m = lgb.LGBMClassifier(**params, random_state=42, verbosity=-1)
            m.fit(Xtr, ytr)
            proba = m.predict_proba(Xoos)[:, 1]
            auc = roc_auc_score(yoos, proba)
            cells = []
            for thr in (0.60, 0.65, 0.70):
                sel = proba >= thr
                n = int(sel.sum())
                wr = float(yoos[sel].mean() * 100) if n else 0
                cells.append(f"{n:>4}/{wr:>5.1f}%")
            print(f"  {name:14s} {auc:>7.3f} {cells[0]:>14s} {cells[1]:>14s} {cells[2]:>14s}")
        print()


if __name__ == "__main__":
    main()
