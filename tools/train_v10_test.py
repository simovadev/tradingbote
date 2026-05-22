"""Train V10 TEST : compare 4 datasets (RR x SWS) x 4 configs hyperparametres.

Pour chaque combo : split train/OOS, AUC, et surtout nb de trades + WR
a chaque seuil. But : trouver la combinaison qui maximise le VOLUME de
trades a WR maintenu (>= ~65%).

Usage : python tools/train_v10_test.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TEST_ASSETS = ["XAUUSD", "EURUSD", "NAS100"]
NON_FEATURES = {"instrument", "ts", "direction", "outcome", "pnl_usd", "bars_to_exit", "target"}

# 4 datasets : RR x SWS
DATASETS = {
    "RR2.0/swsDEF": "rr20_swsdef",
    "RR2.0/sws1":   "rr20_sws1",
    "RR1.5/swsDEF": "rr15_swsdef",
    "RR1.5/sws1":   "rr15_sws1",
}

CONFIGS = {
    "V9_base":   dict(n_estimators=300, learning_rate=0.05, max_depth=6,
                      num_leaves=31, min_child_samples=20, subsample=0.8,
                      colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1),
    "deeper":    dict(n_estimators=500, learning_rate=0.03, max_depth=9,
                      num_leaves=63, min_child_samples=30, subsample=0.8,
                      colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1),
    "balanced":  dict(n_estimators=400, learning_rate=0.04, max_depth=7,
                      num_leaves=47, min_child_samples=25, subsample=0.8,
                      colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
                      class_weight="balanced"),
    "more_tree": dict(n_estimators=800, learning_rate=0.02, max_depth=7,
                      num_leaves=47, min_child_samples=20, subsample=0.85,
                      colsample_bytree=0.85, reg_alpha=0.05, reg_lambda=0.05),
}


def load_combined(suffix):
    """Charge et concatene les 3 actifs pour un dataset donne."""
    parts = []
    for a in TEST_ASSETS:
        p = ROOT / f"data/ml_dataset_{a}_v10test_{suffix}.parquet"
        if not p.exists():
            return None
        df = pd.read_parquet(p)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df = df[df["outcome"].isin(["WIN", "LOSS"])]
        parts.append(df)
    return pd.concat(parts, ignore_index=True).sort_values("ts").reset_index(drop=True)


def evaluate(df, params):
    """Train sur 75%, eval sur 25% OOS. Retourne (auc, {seuil: (n, wr, pnl)})."""
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
    m = lgb.LGBMClassifier(**params, random_state=42, verbosity=-1)
    m.fit(Xtr, ytr)
    proba = m.predict_proba(Xoos)[:, 1]
    auc = roc_auc_score(yoos, proba)
    oos = oos.assign(proba=proba, win=yoos.values)
    res = {}
    n_months = (oos["ts"].max() - oos["ts"].min()).days / 30.0
    for thr in (0.55, 0.60, 0.65, 0.70):
        sel = oos[oos["proba"] >= thr]
        n = len(sel)
        wr = sel["win"].mean() * 100 if n else 0
        pnl = sel["pnl_usd"].sum() if n else 0
        tpm = n / n_months if n_months > 0 else 0
        res[thr] = (n, wr, pnl, tpm)
    return auc, res, len(feats)


def main():
    print("=" * 95)
    print("TRAIN V10 TEST - 4 datasets x 4 configs (3 actifs, 1 an, OOS 25%)")
    print("=" * 95)

    best = None
    for ds_label, suffix in DATASETS.items():
        df = load_combined(suffix)
        if df is None:
            print(f"\n### {ds_label} : dataset manquant, skip")
            continue
        print(f"\n### DATASET {ds_label}  ({len(df):,} trades fermes)")
        print(f"  {'config':12s} {'AUC':>6s} {'feats':>6s} | "
              f"{'@0.60 tpm/WR':>15s} {'@0.65 tpm/WR':>15s} {'@0.70 tpm/WR':>15s}")
        for cfg_name, params in CONFIGS.items():
            try:
                auc, res, nfeat = evaluate(df, params)
            except Exception as e:
                print(f"  {cfg_name:12s} ERREUR {e}")
                continue
            cells = []
            for thr in (0.60, 0.65, 0.70):
                n, wr, pnl, tpm = res[thr]
                cells.append(f"{tpm:>5.1f}/{wr:>5.1f}%")
            print(f"  {cfg_name:12s} {auc:>6.3f} {nfeat:>6d} | "
                  f"{cells[0]:>15s} {cells[1]:>15s} {cells[2]:>15s}")
            # Score : volume a 0.65 x WR (compromis)
            n65, wr65, pnl65, tpm65 = res[0.65]
            if wr65 >= 63 and tpm65 >= 10:
                score = tpm65 * wr65
                if best is None or score > best[0]:
                    best = (score, ds_label, cfg_name, auc, tpm65, wr65, pnl65)

    print("\n" + "=" * 95)
    if best:
        print(f"MEILLEUR COMBO (volume x WR @0.65, WR>=63%) :")
        print(f"  Dataset : {best[1]}")
        print(f"  Config  : {best[2]}")
        print(f"  AUC={best[3]:.3f} | {best[4]:.1f} trades/mois @0.65 | WR {best[5]:.1f}% | PnL {best[6]:+.0f}")
    else:
        print("Aucun combo avec WR>=63% et volume>=10/mois")


if __name__ == "__main__":
    main()
