"""Adversarial validation V18.8 : verifie que ICT pur reduit le drift.

Cible : adversarial AUC < 0.65 (vs 0.85-0.99 V18.6/V18.7)
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

# MEMES features V18.8 dropped
NON_STATIONARY_DROP = {
    "atr_at_setup", "risk_points",
    "dist_to_pdh_pct", "dist_to_pdl_pct", "dist_to_d1_open_pct",
    "dist_pdh_atr", "dist_pdl_atr", "atr_regime",
    "dist_pdh_log", "dist_pdl_log",
    "dist_round_atr", "liq_asymmetry", "adr_consumed_pct",
    "bias_x_fvg", "bias_x_parent_x_fvg", "kz_ny_am_x_bias",
    "ob_x_sweep_strength", "unicorn_x_ny",
    "hour_of_day",
}

NON_FEATURES = {
    "instrument", "ts", "direction",
    "outcome", "pnl_usd", "bars_to_exit",
    "pnl_R", "pnl_R_clipped",
}

TRAIN_END = pd.Timestamp("2025-05-22", tz="UTC")
OOS_START = pd.Timestamp("2025-11-24", tz="UTC")


def load_dataset(asset: str):
    for suffix in ("v18_7", "v18_4", "v18_3"):
        p = Path(f"{ROOT}/data/ml_dataset_{asset}_vantage_{suffix}.parquet")
        if p.exists():
            df = pd.read_parquet(p)
            df["ts"] = pd.to_datetime(df["ts"], utc=True)
            df = df.sort_values("ts").reset_index(drop=True)
            return df
    return None


def adversarial_one(asset: str):
    df = load_dataset(asset)
    if df is None:
        return {"asset": asset, "status": "no_data"}

    df_closed = df[df["outcome"].isin(["WIN", "LOSS"])].copy()
    train_df = df_closed[df_closed["ts"] < TRAIN_END].copy()
    oos_df = df_closed[df_closed["ts"] >= OOS_START].copy()

    if len(train_df) < 200 or len(oos_df) < 50:
        return {"asset": asset, "status": "not_enough"}

    # Features V18.8 ICT PUR uniquement
    all_feats = [c for c in df.columns if c not in NON_FEATURES]
    feature_cols = [c for c in all_feats if c not in NON_STATIONARY_DROP]

    n_oos = len(oos_df)
    n_train_sample = min(len(train_df), n_oos * 3)
    train_sample = train_df.sample(n=n_train_sample, random_state=42)

    X_train = train_sample[feature_cols].copy()
    X_oos = oos_df[feature_cols].copy()
    for c in feature_cols:
        if X_train[c].dtype == bool:
            X_train[c] = X_train[c].astype(int)
            X_oos[c] = X_oos[c].astype(int)

    y_all = np.concatenate([np.zeros(len(X_train)), np.ones(len(X_oos))])
    X_all = pd.concat([X_train, X_oos], ignore_index=True)

    X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.3,
                                                random_state=42, stratify=y_all)

    m = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.02,
                            max_depth=6, num_leaves=31, min_child_samples=50,
                            subsample=0.8, colsample_bytree=0.8,
                            random_state=42, verbosity=-1, n_jobs=4)
    m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)],
          callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)])
    pred = m.predict_proba(X_te)[:, 1]
    auc = float(roc_auc_score(y_te, pred))

    imp = pd.DataFrame({"feature": feature_cols, "importance": m.feature_importances_})
    imp = imp.sort_values("importance", ascending=False).head(10)

    diag = "OK" if auc < 0.55 else ("WARN" if auc < 0.65 else ("DRIFT" if auc < 0.75 else "SEVERE"))

    return {
        "asset": asset, "status": "ok",
        "adversarial_auc": auc,
        "n_features": len(feature_cols),
        "diagnostic": diag,
        "top_drift": imp.to_dict("records"),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true")
    p.add_argument("asset", nargs="?")
    args = p.parse_args()

    if args.all or args.asset == "--all":
        assets = ALL_ASSETS
    elif args.asset:
        assets = [args.asset]
    else:
        p.print_help()
        return

    print(f"=== ADVERSARIAL V18.8 (ICT PUR) ===")
    print(f"Cible : AUC < 0.65 (vs 0.85-0.99 V18.6/V18.7)")
    print()

    results = []
    for a in assets:
        try:
            r = adversarial_one(a)
        except Exception as e:
            r = {"asset": a, "status": "exception", "error": str(e)[:200]}
        results.append(r)
        if r.get("status") == "ok":
            print(f"  {a:<11} feats={r['n_features']:<3} AUC={r['adversarial_auc']:.4f} ==> {r['diagnostic']}")
        else:
            print(f"  {a:<11} {r.get('status', '?')}")

    print()
    print(f"=== RECAP V18.8 ADVERSARIAL ===")
    ok = sum(1 for r in results if r.get("diagnostic") == "OK")
    warn = sum(1 for r in results if r.get("diagnostic") == "WARN")
    drift = sum(1 for r in results if r.get("diagnostic") == "DRIFT")
    severe = sum(1 for r in results if r.get("diagnostic") == "SEVERE")
    print(f"OK     ({ok:>2}) : deploy confiance haute")
    print(f"WARN   ({warn:>2}) : deploy avec monitoring")
    print(f"DRIFT  ({drift:>2}) : shadow 1 semaine d'abord")
    print(f"SEVERE ({severe:>2}) : NE PAS DEPLOYER")

    recap = Path(f"{ROOT}/adversarial_v18_8_recap.json")
    recap.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nRecap : {recap.name}")


if __name__ == "__main__":
    main()
