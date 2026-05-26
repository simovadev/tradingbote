"""V18.4 TRAIN : leviers AUC apres fixes V18.3.

Ameliorations vs V18.2 :
- Charge dataset V18.3 (avec score/quality OUT, hour_sin/cos IN, AMBIGUOUS filtre)
- NEW B+F : 9 features ICT supplementaires (round_atr, liq_asymmetry, adr_consumed,
  bias_x_fvg, bias_x_parent_x_fvg, kz_ny_am_x_bias, ob_x_sweep_strength, unicorn_x_ny)
  -> deja calculees dans ml_filter._features_from_result via V18.3 build
- NEW C : mode REGRESSION sur pnl_R (active si TRAIN_OBJ=reg, defaut classification)
  Le ML predit le R-multiple attendu (1.5R, -1R, 2R), pas juste WIN/LOSS binaire.
  Conserve plus d'info -> meilleure discrimination des bons setups.
- NEW D : walk-forward CV 5 folds rolling (active si TRAIN_CV=wf, defaut split fixe)
  Mesure AUC plus robuste, sans dependre du hasard d'1 seul split.
- Conserve : calibration isotonique, embargo 2j, regularisation forte LightGBM

Usage :
    python -m bot_v2.train_v18_4_vantage XAUUSD
    python -m bot_v2.train_v18_4_vantage --all
    TRAIN_OBJ=reg python -m bot_v2.train_v18_4_vantage XAUUSD  # mode regression
    TRAIN_CV=wf python -m bot_v2.train_v18_4_vantage XAUUSD    # walk-forward
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, mean_squared_error

ROOT = "c:/Users/Shadow/TradingBot" if sys.platform == "win32" else "/workspace/TradingBot"
sys.path.insert(0, ROOT)

ALL_ASSETS = [
    "BTCUSD", "ETHUSD", "USDMXN", "XAUUSD", "USDJPY", "USDZAR", "USDCAD", "NZDUSD",
    "GBPUSD", "CL-OIL", "EURUSD", "AUDUSD", "USDCHF", "NAS100", "XAGUSD",
    "DJ30", "GAS-C", "HK50", "GER40", "FRA40", "UK100", "Nikkei225",
    "BVSPX", "SP500", "Coffee-C", "Cocoa-C", "Wheat-C", "Sugar-C",
]

TRAIN_OBJ = os.environ.get("TRAIN_OBJ", "cls")   # "cls" ou "reg"
TRAIN_CV = os.environ.get("TRAIN_CV", "fixed")   # "fixed" ou "wf"

EMBARGO = pd.Timedelta(days=2)
TRAIN_END = pd.Timestamp("2025-05-22", tz="UTC")
VAL_START = TRAIN_END + EMBARGO
VAL_END = pd.Timestamp("2025-11-22", tz="UTC")
OOS_START = VAL_END + EMBARGO

NON_FEATURES = {
    "instrument", "ts", "direction",
    "outcome", "pnl_usd", "bars_to_exit",
    "pnl_R", "pnl_R_clipped",  # target regression : exclu
}


def load_dataset(asset: str) -> pd.DataFrame | None:
    """Charge dataset V18.4 (priorite, 9 new features ICT) puis V18.3/V18.2."""
    for suffix in ("v18_4", "v18_3", "v18_2"):
        p = Path(f"{ROOT}/data/ml_dataset_{asset}_vantage_{suffix}.parquet")
        if p.exists():
            print(f"  Chargement {p.name}")
            df = pd.read_parquet(p)
            df["ts"] = pd.to_datetime(df["ts"], utc=True)
            df = df.sort_values("ts").reset_index(drop=True)
            return df
    print(f"  Aucun dataset V18.4/V18.3/V18.2 pour {asset}")
    return None


def add_pnl_R(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute la colonne pnl_R (R-multiple) pour le mode regression.

    WIN : +rr (le RR du trade, typiquement 1.5 a 2.5)
    LOSS : -1
    PENDING/AMBIGUOUS : exclus (filtre amont)
    """
    df = df.copy()
    df["pnl_R"] = 0.0
    win_mask = df["outcome"] == "WIN"
    loss_mask = df["outcome"] == "LOSS"
    if "rr" in df.columns:
        df.loc[win_mask, "pnl_R"] = df.loc[win_mask, "rr"].astype(float)
    else:
        df.loc[win_mask, "pnl_R"] = 2.0  # defaut RR=2
    df.loc[loss_mask, "pnl_R"] = -1.0
    # Clip pour robustesse (regression sensible aux outliers)
    df["pnl_R_clipped"] = df["pnl_R"].clip(-1.5, 3.5)
    return df


def prepare_xy(df: pd.DataFrame, target_col: str = "target"):
    df = df.sort_values("ts").reset_index(drop=True)
    feature_cols = [c for c in df.columns
                    if c not in NON_FEATURES and c != "target" and c != "pnl_R" and c != "pnl_R_clipped"]
    X = df[feature_cols].copy()
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
    y = df[target_col]
    return X, y, feature_cols


def _make_lgbm_classifier():
    return lgb.LGBMClassifier(
        n_estimators=2000, learning_rate=0.01,
        max_depth=5, num_leaves=31, min_child_samples=200,
        subsample=0.7, colsample_bytree=0.6,
        reg_alpha=1.0, reg_lambda=5.0, min_split_gain=0.01,
        random_state=42, verbosity=-1,
    )


def _make_lgbm_regressor():
    return lgb.LGBMRegressor(
        n_estimators=2000, learning_rate=0.01,
        max_depth=5, num_leaves=31, min_child_samples=200,
        subsample=0.7, colsample_bytree=0.6,
        reg_alpha=1.0, reg_lambda=5.0, min_split_gain=0.01,
        random_state=42, verbosity=-1,
        objective="regression",
    )


def _eval_thresholds(name, y_true, y_pred, metrics):
    """Eval AUC + WR@seuil. y_pred = probas calibrees (0..1)."""
    try:
        auc = roc_auc_score(y_true, y_pred)
    except Exception:
        auc = float("nan")
    metrics[f"auc_{name}"] = float(auc)
    print(f"\n  {name.upper()} ({len(y_true)} trades) AUC={auc:.3f}")
    for thr in (0.55, 0.60, 0.65, 0.70, 0.75):
        mask = y_pred >= thr
        n = int(mask.sum())
        if n < 5:
            print(f"    seuil={thr:.2f} : {n:>3} trades, WR=-- (trop peu)")
            continue
        wr = float(y_true[mask].mean() * 100)
        metrics[f"{name}_n@{thr:.2f}"] = n
        metrics[f"{name}_wr@{thr:.2f}"] = wr
        print(f"    seuil={thr:.2f} : {n:>3} trades, WR={wr:.1f}%")


def _eval_reg_thresholds(name, y_true_bin, y_true_R, y_pred_R, metrics):
    """Eval mode regression : y_pred = R-multiple attendu. Seuils en R."""
    try:
        auc = roc_auc_score(y_true_bin, y_pred_R)
    except Exception:
        auc = float("nan")
    metrics[f"auc_{name}"] = float(auc)
    mse = mean_squared_error(y_true_R, y_pred_R)
    metrics[f"mse_{name}"] = float(mse)
    print(f"\n  {name.upper()} ({len(y_true_R)} trades) AUC={auc:.3f} MSE={mse:.3f}")
    # Seuils en R-multiple attendu
    for thr in (0.0, 0.3, 0.5, 0.7, 1.0):
        mask = y_pred_R >= thr
        n = int(mask.sum())
        if n < 5:
            print(f"    pred_R>={thr:+.2f} : {n:>3} trades, WR=-- (trop peu)")
            continue
        wr = float(y_true_bin[mask].mean() * 100)
        avg_R = float(y_true_R[mask].mean())
        metrics[f"{name}_n@R{thr:.1f}"] = n
        metrics[f"{name}_wr@R{thr:.1f}"] = wr
        metrics[f"{name}_avgR@R{thr:.1f}"] = avg_R
        print(f"    pred_R>={thr:+.2f} : {n:>3} trades, WR={wr:.1f}% avgR={avg_R:+.2f}")


def _walk_forward_cv(df_closed, feat_cols, n_folds=5):
    """Walk-forward 5 folds : train sur [0..t], test sur [t..t+1/n].
    Retourne AUC moyen + std.
    """
    print(f"\n  Walk-forward CV ({n_folds} folds)...")
    n = len(df_closed)
    fold_size = n // (n_folds + 1)  # 1er fold = warmup
    aucs = []
    for k in range(n_folds):
        train_end = fold_size * (k + 1)
        val_start = train_end
        val_end = train_end + fold_size
        if val_end > n:
            break
        tr = df_closed.iloc[:train_end]
        vl = df_closed.iloc[val_start:val_end]
        if len(tr) < 100 or len(vl) < 30:
            continue
        X_tr, y_tr, _ = prepare_xy(tr)
        X_vl, y_vl, _ = prepare_xy(vl)
        try:
            m = _make_lgbm_classifier()
            m.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)],
                  callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
            pp = m.predict_proba(X_vl)[:, 1]
            a = roc_auc_score(y_vl, pp)
            aucs.append(a)
            print(f"    fold {k+1}/{n_folds} : train={len(tr)} val={len(vl)} AUC={a:.3f}")
        except Exception as e:
            print(f"    fold {k+1} fail : {e}")
    if not aucs:
        return float("nan"), float("nan")
    return float(np.mean(aucs)), float(np.std(aucs))


def train_one(asset: str) -> dict:
    print(f"\n{'='*60}")
    print(f"=== TRAIN {asset} V18.4 (OBJ={TRAIN_OBJ}, CV={TRAIN_CV}) ===")
    print(f"{'='*60}")

    df = load_dataset(asset)
    if df is None:
        return {"asset": asset, "status": "no_data"}

    df = add_pnl_R(df)
    df_closed = df[df["outcome"].isin(["WIN", "LOSS"])].copy()
    n_total = len(df_closed)
    if n_total < 200:
        return {"asset": asset, "status": "not_enough", "n": n_total}

    wr_brut = (df_closed["outcome"] == "WIN").mean() * 100
    avgR_brut = df_closed["pnl_R"].mean()
    print(f"  N={n_total} | WR brut {wr_brut:.1f}% | avgR brut {avgR_brut:+.3f}")
    print(f"  Periode : {df_closed['ts'].min()} -> {df_closed['ts'].max()}")

    # Target : binaire (cls) ou continu (reg)
    df_closed["target"] = (df_closed["outcome"] == "WIN").astype(int)

    train_df = df_closed[df_closed["ts"] < TRAIN_END].copy()
    val_df = df_closed[(df_closed["ts"] >= VAL_START) & (df_closed["ts"] < VAL_END)].copy()
    oos_df = df_closed[df_closed["ts"] >= OOS_START].copy()
    print(f"  Train={len(train_df)} Val={len(val_df)} OOS={len(oos_df)}")

    if len(train_df) < 100 or len(val_df) < 20:
        return {"asset": asset, "status": "bad_split",
                "n_train": len(train_df), "n_val": len(val_df), "n_oos": len(oos_df)}

    # Walk-forward CV optionnel (sur train+val concatenes)
    if TRAIN_CV == "wf":
        cv_train = df_closed[df_closed["ts"] < OOS_START].copy()
        feat_cols_wf = [c for c in cv_train.columns if c not in NON_FEATURES and c != "target"]
        auc_wf_mean, auc_wf_std = _walk_forward_cv(cv_train, feat_cols_wf, n_folds=5)
        print(f"\n  Walk-forward AUC : {auc_wf_mean:.3f} +/- {auc_wf_std:.3f}")

    X_train, y_train, feat_cols = prepare_xy(train_df)
    X_val, y_val, _ = prepare_xy(val_df)
    has_oos = len(oos_df) >= 10
    if has_oos:
        X_oos, y_oos, _ = prepare_xy(oos_df)
    else:
        X_oos, y_oos = X_val.head(0).copy(), y_val.head(0).copy()

    cat_cols = []
    for c in feat_cols:
        if X_train[c].dtype == "object" or X_train[c].dtype.name == "category":
            X_train[c] = X_train[c].astype("category")
            X_val[c] = X_val[c].astype("category")
            if has_oos:
                X_oos[c] = X_oos[c].astype("category")
            cat_cols.append(c)

    print(f"\n  Features : {len(feat_cols)} (cat: {cat_cols})")

    metrics = {"asset": asset, "status": "trained",
               "n_train": len(X_train), "n_val": len(X_val), "n_oos": len(X_oos),
               "wr_brut_oos": float((y_oos.mean()) * 100) if has_oos else 0.0,
               "obj": TRAIN_OBJ, "cv": TRAIN_CV}
    if TRAIN_CV == "wf":
        metrics["auc_wf_mean"] = auc_wf_mean
        metrics["auc_wf_std"] = auc_wf_std

    if TRAIN_OBJ == "reg":
        # ===== Mode REGRESSION sur pnl_R =====
        r_train = train_df["pnl_R_clipped"].values
        r_val = val_df["pnl_R_clipped"].values
        r_oos = oos_df["pnl_R_clipped"].values if has_oos else None
        print(f"  Training LGBMRegressor sur pnl_R_clipped (range [-1.5, +3.5])...")
        model = _make_lgbm_regressor()
        model.fit(X_train, r_train, eval_set=[(X_val, r_val)],
                  categorical_feature=cat_cols if cat_cols else "auto",
                  callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
        pred_tr = model.predict(X_train)
        pred_vl = model.predict(X_val)
        pred_oo = model.predict(X_oos) if has_oos else None
        _eval_reg_thresholds("train", y_train.values, r_train, pred_tr, metrics)
        _eval_reg_thresholds("val", y_val.values, r_val, pred_vl, metrics)
        if has_oos:
            _eval_reg_thresholds("oos", y_oos.values, r_oos, pred_oo, metrics)
        model_path = Path(f"{ROOT}/bot_v2/ml_model_{asset}_vantage_v18_4_reg.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
    else:
        # ===== Mode CLASSIFICATION binaire (defaut) avec calibration =====
        model = _make_lgbm_classifier()
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  categorical_feature=cat_cols if cat_cols else "auto",
                  callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
        print(f"  Calibration isotonique sur VAL ({len(X_val)})...")
        # sklearn>=1.6 : cv="prefit" deprecated, utiliser FrozenEstimator
        try:
            from sklearn.frozen import FrozenEstimator
            calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
            calibrated.fit(X_val, y_val)
        except ImportError:
            # Fallback sklearn ancien
            calibrated = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
            calibrated.fit(X_val, y_val)
        pred_tr = calibrated.predict_proba(X_train)[:, 1]
        pred_vl = calibrated.predict_proba(X_val)[:, 1]
        pred_oo = calibrated.predict_proba(X_oos)[:, 1] if has_oos else None
        _eval_thresholds("train", y_train, pred_tr, metrics)
        _eval_thresholds("val", y_val, pred_vl, metrics)
        if has_oos:
            _eval_thresholds("oos", y_oos, pred_oo, metrics)
        model_path = Path(f"{ROOT}/bot_v2/ml_model_{asset}_vantage_v18_4.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(calibrated, f)

    features_path = Path(f"{ROOT}/bot_v2/ml_features_{asset}_vantage_v18_4.json")
    features_path.write_text(json.dumps({"features": feat_cols}, indent=2))
    print(f"\n  Saved : {model_path.name} + {features_path.name}")

    # Verdict
    verdict = "PASS"
    reasons = []
    if has_oos:
        auc_oos = metrics.get("auc_oos", 0)
        if auc_oos < 0.65:
            verdict = "FAIL"
            reasons.append(f"AUC OOS {auc_oos:.3f} < 0.65")
        if TRAIN_OBJ == "cls":
            wr_oos = metrics.get("oos_wr@0.65", 0)
            n_oos = metrics.get("oos_n@0.65", 0)
            if n_oos >= 5 and wr_oos < 60:
                verdict = "FAIL"
                reasons.append(f"WR OOS @0.65 {wr_oos:.1f}% < 60%")
    metrics["verdict"] = verdict
    metrics["reasons"] = reasons
    print(f"\n  ===> {verdict} {' | '.join(reasons) if reasons else 'OK'}")
    return metrics


def main():
    p = argparse.ArgumentParser(description="Train V18.4 (V18.3 dataset + R-mult + walk-forward + features ICT)")
    p.add_argument("asset", nargs="?", help="Asset ou --all")
    p.add_argument("--all", action="store_true")
    args = p.parse_args()

    if args.all or args.asset == "--all":
        all_metrics = []
        for a in ALL_ASSETS:
            try:
                m = train_one(a)
            except Exception as e:
                print(f"!! FAIL {a} : {e}")
                import traceback; traceback.print_exc()
                m = {"asset": a, "status": "exception", "error": str(e)[:200]}
            all_metrics.append(m)

        print("\n" + "=" * 80)
        print(f"=== RECAP V18.4 (OBJ={TRAIN_OBJ}, CV={TRAIN_CV}) ===")
        print("=" * 80)
        print(f"{'Asset':<10} {'Status':<12} {'AUC OOS':<8} {'WR':<8} {'N':<7} {'Verdict':<6}")
        for m in all_metrics:
            status = m.get("status", "?")
            auc = f"{m.get('auc_oos', 0):.3f}" if "auc_oos" in m else "--"
            if TRAIN_OBJ == "reg":
                wr = f"{m.get('oos_wr@R0.5', 0):.1f}%" if "oos_wr@R0.5" in m else "--"
                n = m.get("oos_n@R0.5", "--")
            else:
                wr = f"{m.get('oos_wr@0.65', 0):.1f}%" if "oos_wr@0.65" in m else "--"
                n = m.get("oos_n@0.65", "--")
            verdict = m.get("verdict", "--")
            print(f"{m['asset']:<10} {status:<12} {auc:<8} {wr:<8} {n!s:<7} {verdict:<6}")

        recap_path = Path(f"{ROOT}/ml_metrics_v18_4_{TRAIN_OBJ}_recap.json")
        recap_path.write_text(json.dumps(all_metrics, indent=2, default=str))
        print(f"\nRecap : {recap_path.name}")

    elif args.asset:
        if args.asset not in ALL_ASSETS:
            print(f"Actif inconnu : {args.asset}")
            sys.exit(1)
        train_one(args.asset)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
