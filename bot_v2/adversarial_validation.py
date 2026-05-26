"""ADVERSARIAL VALIDATION pour V18.6 — check critique avant deploy live.

Principe :
- On train un modele qui essaie de distinguer TRAIN data vs OOS data
- Si AUC > 0.6 : les data ont change significativement entre train et OOS
  -> le ML va probablement DERIVER en live
- Si AUC < 0.55 : train et OOS sont statistiquement similaires
  -> confiance haute pour le live

C'est LE check ultime avant deploy. Une AUC > 0.7 = drift severe = NO-GO live.

Strategie :
- Label = 1 si trade vient de OOS, 0 si TRAIN
- Si le modele peut predire le label avec haute AUC -> les distributions sont DIFFERENTES
- Si le modele rame (AUC ~ 0.5) -> distributions IDENTIQUES = bon signe pour live

Output : adversarial_recap_v18_6.json

Usage :
    python -m bot_v2.adversarial_validation --all
"""
from __future__ import annotations

import argparse
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

ALL_ASSETS = [
    "BTCUSD", "ETHUSD", "USDMXN", "XAUUSD", "USDJPY", "USDZAR", "USDCAD", "NZDUSD",
    "GBPUSD", "CL-OIL", "EURUSD", "AUDUSD", "USDCHF", "NAS100", "XAGUSD",
    "DJ30", "GAS-C", "HK50", "GER40", "FRA40", "UK100", "Nikkei225",
    "BVSPX", "SP500", "Coffee-C", "Cocoa-C", "Wheat-C", "Sugar-C",
]

EMBARGO = pd.Timedelta(days=2)
TRAIN_END = pd.Timestamp("2025-05-22", tz="UTC")
VAL_START = TRAIN_END + EMBARGO
VAL_END = pd.Timestamp("2025-11-22", tz="UTC")
OOS_START = VAL_END + EMBARGO

NON_FEATURES = {
    "instrument", "ts", "direction",
    "outcome", "pnl_usd", "bars_to_exit",
    "pnl_R", "pnl_R_clipped",
}

V18_4_NEW_FEATURES = {
    "dist_round_atr", "liq_asymmetry", "adr_consumed_pct",
    "bias_x_fvg", "bias_x_parent_x_fvg", "kz_ny_am_x_bias",
    "ob_x_sweep_strength", "unicorn_x_ny",
}


def load_dataset(asset: str) -> pd.DataFrame | None:
    for suffix in ("v18_4", "v18_3"):
        p = Path(f"{ROOT}/data/ml_dataset_{asset}_vantage_{suffix}.parquet")
        if p.exists():
            df = pd.read_parquet(p)
            df["ts"] = pd.to_datetime(df["ts"], utc=True)
            df = df.sort_values("ts").reset_index(drop=True)
            return df
    return None


def load_v18_6_feature_set(asset: str) -> str:
    """Charge le choix CORE/FULL du modele V18.6 entraine pour cet actif."""
    try:
        p = Path(f"{ROOT}/ml_features_{asset}_vantage_v18_6.json")
        # Fallback : essayer le path bot_v2/
        if not p.exists():
            p = Path(f"{ROOT}/bot_v2/ml_features_{asset}_vantage_v18_6.json")
        if p.exists():
            r = json.loads(p.read_text())
            feats = r.get("features", [])
            # Si contient les features V18.4 = FULL, sinon CORE
            has_v18_4 = any(f in V18_4_NEW_FEATURES for f in feats)
            return "full" if has_v18_4 else "core"
    except Exception:
        pass
    return "core"  # defaut


def prepare_features(df: pd.DataFrame, feature_set: str = "core"):
    df = df[df["outcome"].isin(["WIN", "LOSS"])].copy()
    all_feats = [c for c in df.columns if c not in NON_FEATURES]
    if feature_set == "core":
        feature_cols = [c for c in all_feats if c not in V18_4_NEW_FEATURES]
    else:
        feature_cols = all_feats
    X = df[feature_cols].copy()
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
    return X, feature_cols


def adversarial_one(asset: str) -> dict:
    """Run adversarial validation : train vs OOS distinguishability."""
    print(f"\n=== {asset} ===")
    df = load_dataset(asset)
    if df is None:
        return {"asset": asset, "status": "no_data"}

    df_closed = df[df["outcome"].isin(["WIN", "LOSS"])].copy()
    train_df = df_closed[df_closed["ts"] < TRAIN_END].copy()
    oos_df = df_closed[df_closed["ts"] >= OOS_START].copy()

    if len(train_df) < 200 or len(oos_df) < 50:
        return {"asset": asset, "status": "not_enough",
                "n_train": len(train_df), "n_oos": len(oos_df)}

    # Determine features : on utilise le choix V18.6 si dispo
    feature_set = load_v18_6_feature_set(asset)

    # Sous-echantillonner train pour equilibrer si trop grosse difference
    n_oos = len(oos_df)
    n_train_sample = min(len(train_df), n_oos * 3)  # max 3:1 ratio
    train_sample = train_df.sample(n=n_train_sample, random_state=42)

    X_train, feat_cols = prepare_features(train_sample, feature_set)
    X_oos, _ = prepare_features(oos_df, feature_set)

    # Aligner colonnes
    common = [c for c in feat_cols if c in X_train.columns and c in X_oos.columns]
    X_train = X_train[common]
    X_oos = X_oos[common]

    # Label : 0 = train, 1 = oos
    y_train_lbl = np.zeros(len(X_train))
    y_oos_lbl = np.ones(len(X_oos))

    X_all = pd.concat([X_train, X_oos], ignore_index=True)
    y_all = np.concatenate([y_train_lbl, y_oos_lbl])

    # Train/test split aleatoire (pas temporel, on cherche a distinguer les sources)
    X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.3,
                                                random_state=42, stratify=y_all)

    # LightGBM adversarial - on veut un modele qui essaye fort
    model = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.02,
        max_depth=6, num_leaves=31,
        min_child_samples=50,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)],
              callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)])

    pred = model.predict_proba(X_te)[:, 1]
    auc = float(roc_auc_score(y_te, pred))

    # Top features qui distinguent train/oos
    imp = pd.DataFrame({"feature": common, "importance": model.feature_importances_})
    imp = imp.sort_values("importance", ascending=False).head(10)
    top_drift = imp.to_dict("records")

    # Diagnostic
    if auc < 0.55:
        diag = "OK"
        comment = "Train et OOS statistiquement similaires - confiance HAUTE pour live"
    elif auc < 0.65:
        diag = "WARN"
        comment = "Drift moderee - live possible mais monitoring serre"
    elif auc < 0.75:
        diag = "DRIFT"
        comment = "Drift significatif - live risque, peut nicht generalise"
    else:
        diag = "SEVERE"
        comment = "Drift severe - DO NOT DEPLOY, donnees differentes"

    print(f"  Feature set : {feature_set} ({len(common)} feats)")
    print(f"  N train sample : {len(X_train)} | N OOS : {len(X_oos)}")
    print(f"  Adversarial AUC : {auc:.4f}  ==> {diag}")
    print(f"  {comment}")
    print(f"  Top 5 features drift :")
    for row in top_drift[:5]:
        print(f"    {row['feature']:<30} importance={row['importance']:>5}")

    return {
        "asset": asset,
        "status": "ok",
        "feature_set": feature_set,
        "n_features": len(common),
        "n_train_sample": len(X_train),
        "n_oos": len(X_oos),
        "adversarial_auc": auc,
        "diagnostic": diag,
        "comment": comment,
        "top_drift_features": top_drift[:10],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("asset", nargs="?")
    p.add_argument("--all", action="store_true")
    args = p.parse_args()

    assets = ALL_ASSETS if (args.all or args.asset == "--all") else [args.asset]
    if not assets[0]:
        p.print_help()
        sys.exit(1)

    print(f"=== ADVERSARIAL VALIDATION V18.6 ===")
    print(f"Critere : si AUC adversarial > 0.65 -> train et OOS sont DIFFERENTS")
    print(f"         si AUC < 0.55 -> train et OOS sont SIMILAIRES (bon pour live)")
    print(f"         si AUC > 0.75 -> SEVERE drift, DO NOT DEPLOY")
    print(f"Train period : ... -> {TRAIN_END.date()}")
    print(f"OOS period   : {OOS_START.date()} -> ...")
    print()

    results = []
    for a in assets:
        try:
            r = adversarial_one(a)
        except Exception as e:
            print(f"!! FAIL {a} : {e}")
            import traceback; traceback.print_exc()
            r = {"asset": a, "status": "exception", "error": str(e)[:200]}
        results.append(r)

    print("\n" + "=" * 90)
    print("=== RECAP ADVERSARIAL VALIDATION V18.6 ===")
    print("=" * 90)
    print(f"{'Asset':<11} {'FeatSet':<8} {'AdvAUC':<8} {'Diagnostic':<10} {'Verdict pour live':<40}")
    n_ok = 0
    n_warn = 0
    n_drift = 0
    n_severe = 0
    for r in results:
        if r.get("status") != "ok":
            print(f"{r['asset']:<11} {r.get('status', '?'):<8}")
            continue
        d = r.get("diagnostic", "?")
        if d == "OK": n_ok += 1
        elif d == "WARN": n_warn += 1
        elif d == "DRIFT": n_drift += 1
        elif d == "SEVERE": n_severe += 1
        comment = r.get("comment", "")[:38]
        print(f"{r['asset']:<11} {r.get('feature_set','?'):<8} "
              f"{r.get('adversarial_auc',0):<8.4f} {d:<10} {comment:<40}")

    print()
    print(f"OK     ({n_ok:>2}) : deploy live confiance haute")
    print(f"WARN   ({n_warn:>2}) : deploy avec monitoring serre")
    print(f"DRIFT  ({n_drift:>2}) : eviter live, ou shadow 1 semaine d'abord")
    print(f"SEVERE ({n_severe:>2}) : NE PAS DEPLOYER en live")

    recap = Path(f"{ROOT}/adversarial_recap_v18_6.json")
    recap.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nRecap : {recap.name}")


if __name__ == "__main__":
    main()
