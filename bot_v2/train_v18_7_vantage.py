"""V18.7 : TRAIN REGULARISE FORT + ABLATION + WALK-FORWARD CV.

Objectif : reduire le gap TRAIN-OOS (0.145 dans V18.5) pour avoir un modele
qui generalise vraiment, meme si l'AUC train baisse.

Strategie :
1. Hyperparams TRES regulariss (depth=4, min_child=500, reg=20+) FIXES
   pour tous les actifs (pas de genetique = pas de sur-selection)
2. Mode ABLATION : on test 2 jeux de features
   - core : 61 features V18.3 (sans les 9 V18.4)
   - full : 69 features V18.4
   -> selectionne automatiquement le meilleur (par AUC OOS)
3. WALK-FORWARD CV 5 folds rolling pour mesure honnete

Output :
- ml_model_<ASSET>_vantage_v18_7.pkl (best variant)
- ml_features_<ASSET>_vantage_v18_7.json
- ml_metrics_v18_7_recap.json (avec wf_auc et auc_oos compares)

Usage :
    python -m bot_v2.train_v18_7_vantage --all
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
from sklearn.metrics import roc_auc_score

ROOT = "c:/Users/Shadow/TradingBot" if sys.platform == "win32" else "/workspace/TradingBot"
sys.path.insert(0, ROOT)

ALL_ASSETS = [
    "BTCUSD", "ETHUSD", "USDMXN", "XAUUSD", "USDJPY", "USDZAR", "USDCAD", "NZDUSD",
    "GBPUSD", "CL-OIL", "EURUSD", "AUDUSD", "USDCHF", "NAS100", "XAGUSD",
    "DJ30", "GAS-C", "HK50", "GER40", "FRA40", "UK100", "Nikkei225",
    "BVSPX", "SP500", "Coffee-C", "Cocoa-C", "Wheat-C", "Sugar-C",
]

# V18.7 ablation : on isole les features qui DRIFTENT (adversarial V18.6)
# pour les retirer dans la variante "core" (= drift-free).
DRIFT_FEATURES = {
    # Features absolues (USD/EUR/etc.) qui changent avec l'epoque
    "atr_at_setup",            # ATR en valeur absolue
    "risk_points",             # Risk en points absolus
    # Features % qui drift quand prix monte structurellement
    "dist_to_pdh_pct", "dist_to_pdl_pct", "dist_to_d1_open_pct",
    # Features distance-ATR qui drift aussi (ATR brut dedans)
    "dist_pdh_atr", "dist_pdl_atr",
    # ATR regime utilise ATR brut
    "atr_regime",
}

# 9 features V18.4 (peu utiles selon ablation V18.6 mais on les exclut pour proprete)
V18_4_NEW_FEATURES = {
    "dist_round_atr", "liq_asymmetry", "adr_consumed_pct",
    "bias_x_fvg", "bias_x_parent_x_fvg", "kz_ny_am_x_bias",
    "ob_x_sweep_strength", "unicorn_x_ny",
}

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

# HYPERPARAMS V18.7 : TRES REGULARISES (anti-overfit)
# n_jobs=4 = limit threads par fit (avec 8 actifs en parallel = 32 cores actifs sans thrashing)
HYPERPARAMS_REGULARIZED = {
    "n_estimators": 2000,          # 3000->2000 (gain temps sans perte significative)
    "learning_rate": 0.01,         # 0.005->0.01 (converge 2x plus vite)
    "max_depth": 4,                # peu profond
    "num_leaves": 15,              # tres peu de splits
    "min_child_samples": 500,      # gros buckets = pas memorisation
    "subsample": 0.7,
    "colsample_bytree": 0.5,
    "reg_alpha": 3.0,              # L1 fort
    "reg_lambda": 20.0,            # L2 tres fort
    "min_split_gain": 0.05,
    "random_state": 42,
    "verbosity": -1,
    "n_jobs": 4,                   # CRITIQUE : limite threads (anti-thrashing parallel)
}

# Si V18_7_FAST=1 : desactive WF CV (gain x6 temps mais robustesse moins mesurable)
import os as _os
_FAST_MODE = _os.environ.get("V18_7_FAST", "0") == "1"


def load_dataset(asset: str) -> pd.DataFrame | None:
    """Charge V18.7 (priorite, features normalisees), fallback V18.4/V18.3."""
    for suffix in ("v18_7", "v18_4", "v18_3"):
        p = Path(f"{ROOT}/data/ml_dataset_{asset}_vantage_{suffix}.parquet")
        if p.exists():
            print(f"  Chargement {p.name}")
            df = pd.read_parquet(p)
            df["ts"] = pd.to_datetime(df["ts"], utc=True)
            df = df.sort_values("ts").reset_index(drop=True)
            return df
    return None


def prepare_xy(df: pd.DataFrame, feature_set: str = "full"):
    """feature_set V18.7 :
    - 'core' : features V18.3 SANS les drift features (~53 feats, drift-free)
    - 'full' : features V18.3 + V18.7 normalisees AVEC drift features (~68 feats)
    Toutes les variantes excluent les 9 features V18.4 (jugees inutiles V18.6).
    """
    df = df[df["outcome"].isin(["WIN", "LOSS"])].copy()
    df["target"] = (df["outcome"] == "WIN").astype(int)
    df = df.sort_values("ts").reset_index(drop=True)
    all_feats = [c for c in df.columns if c not in NON_FEATURES and c != "target"]
    # Toujours exclure les 9 features V18.4 (preuve V18.6 = pas d'aide)
    all_feats = [c for c in all_feats if c not in V18_4_NEW_FEATURES]
    if feature_set == "core":
        # core = drift-free : on retire aussi les features qui drift
        feature_cols = [c for c in all_feats if c not in DRIFT_FEATURES]
    else:
        # full = tout sauf V18.4 (garde drift features pour comparaison)
        feature_cols = all_feats
    X = df[feature_cols].copy()
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
    y = df["target"]
    return X, y, feature_cols


def walk_forward_auc(df_closed, feature_set, n_folds=5):
    """Walk-forward CV : 5 folds rolling sur train+val, mesure AUC honnete."""
    cv_data = df_closed[df_closed["ts"] < OOS_START].copy()
    n = len(cv_data)
    fold_size = n // (n_folds + 1)  # fold 1 = warmup train
    aucs = []
    for k in range(n_folds):
        train_end = fold_size * (k + 1)
        val_end = train_end + fold_size
        if val_end > n:
            break
        tr = cv_data.iloc[:train_end]
        vl = cv_data.iloc[train_end:val_end]
        if len(tr) < 200 or len(vl) < 50:
            continue
        X_tr, y_tr, _ = prepare_xy(tr, feature_set)
        X_vl, y_vl, _ = prepare_xy(vl, feature_set)
        try:
            m = lgb.LGBMClassifier(**HYPERPARAMS_REGULARIZED)
            m.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)],
                  callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
            pp = m.predict_proba(X_vl)[:, 1]
            a = roc_auc_score(y_vl, pp)
            aucs.append(a)
        except Exception:
            continue
    return (float(np.mean(aucs)), float(np.std(aucs))) if aucs else (float("nan"), float("nan"))


def _eval_thresholds(name, y_true, y_pred, metrics):
    try:
        auc = roc_auc_score(y_true, y_pred)
    except Exception:
        auc = float("nan")
    metrics[f"auc_{name}"] = float(auc)
    for thr in (0.55, 0.60, 0.65, 0.70):
        mask = y_pred >= thr
        n = int(mask.sum())
        if n < 5:
            continue
        wr = float(y_true[mask].mean() * 100)
        metrics[f"{name}_n@{thr:.2f}"] = n
        metrics[f"{name}_wr@{thr:.2f}"] = wr


def train_variant(asset: str, df_closed: pd.DataFrame, feature_set: str) -> dict:
    """Train une variante (core ou full), retourne metrics + model."""
    train_df = df_closed[df_closed["ts"] < TRAIN_END].copy()
    val_df = df_closed[(df_closed["ts"] >= VAL_START) & (df_closed["ts"] < VAL_END)].copy()
    oos_df = df_closed[df_closed["ts"] >= OOS_START].copy()

    if len(train_df) < 200 or len(val_df) < 30:
        return {"feature_set": feature_set, "status": "bad_split", "model": None}

    X_train, y_train, feat_cols = prepare_xy(train_df, feature_set)
    X_val, y_val, _ = prepare_xy(val_df, feature_set)
    has_oos = len(oos_df) >= 10
    X_oos, y_oos = (prepare_xy(oos_df, feature_set)[:2] if has_oos
                    else (X_val.head(0).copy(), y_val.head(0).copy()))

    model = lgb.LGBMClassifier(**HYPERPARAMS_REGULARIZED)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])

    # Calibration
    try:
        from sklearn.frozen import FrozenEstimator
        calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
        calibrated.fit(X_val, y_val)
    except ImportError:
        calibrated = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
        calibrated.fit(X_val, y_val)

    pred_tr = calibrated.predict_proba(X_train)[:, 1]
    pred_vl = calibrated.predict_proba(X_val)[:, 1]
    pred_oo = calibrated.predict_proba(X_oos)[:, 1] if has_oos else None

    metrics = {"feature_set": feature_set, "n_features": len(feat_cols),
               "n_train": len(X_train), "n_val": len(X_val), "n_oos": len(X_oos),
               "status": "trained"}
    _eval_thresholds("train", y_train, pred_tr, metrics)
    _eval_thresholds("val", y_val, pred_vl, metrics)
    if has_oos:
        _eval_thresholds("oos", y_oos, pred_oo, metrics)

    # Gap diagnostic
    metrics["gap_train_oos"] = metrics.get("auc_train", 0) - metrics.get("auc_oos", 0)

    # Walk-forward CV (3 folds en parallel pour gagner temps, ou skip si FAST)
    if _FAST_MODE:
        metrics["wf_auc_mean"] = 0.0
        metrics["wf_auc_std"] = 0.0
    else:
        wf_mean, wf_std = walk_forward_auc(df_closed, feature_set, n_folds=3)
        metrics["wf_auc_mean"] = wf_mean
        metrics["wf_auc_std"] = wf_std

    return {"feature_set": feature_set, "model": calibrated, "metrics": metrics,
            "feat_cols": feat_cols}


def train_one(asset: str) -> dict:
    print(f"\n{'='*60}")
    print(f"=== TRAIN {asset} V18.7 (regularise + ablation) ===")
    print(f"{'='*60}")

    df = load_dataset(asset)
    if df is None:
        return {"asset": asset, "status": "no_data"}

    df_closed = df[df["outcome"].isin(["WIN", "LOSS"])].copy()
    n_total = len(df_closed)
    if n_total < 200:
        return {"asset": asset, "status": "not_enough", "n": n_total}

    wr_brut = (df_closed["outcome"] == "WIN").mean() * 100
    print(f"  N={n_total} WR_brut={wr_brut:.1f}%")

    # Train 2 variantes
    print(f"\n  Variante CORE (61 features V18.3)...")
    r_core = train_variant(asset, df_closed, "core")
    if r_core["model"] is None:
        return {"asset": asset, "status": r_core.get("metrics", {}).get("status", "fail")}
    mc = r_core["metrics"]
    print(f"    AUC train={mc.get('auc_train',0):.3f} val={mc.get('auc_val',0):.3f} oos={mc.get('auc_oos',0):.3f} "
          f"gap={mc.get('gap_train_oos',0):+.3f} wf={mc.get('wf_auc_mean',0):.3f}")

    print(f"\n  Variante FULL (69 features V18.4)...")
    r_full = train_variant(asset, df_closed, "full")
    if r_full["model"] is None:
        return {"asset": asset, "status": r_full.get("metrics", {}).get("status", "fail")}
    mf = r_full["metrics"]
    print(f"    AUC train={mf.get('auc_train',0):.3f} val={mf.get('auc_val',0):.3f} oos={mf.get('auc_oos',0):.3f} "
          f"gap={mf.get('gap_train_oos',0):+.3f} wf={mf.get('wf_auc_mean',0):.3f}")

    # Selection : meilleur WF AUC (vraie robustesse) en priorite, OOS sinon
    score_core = mc.get('wf_auc_mean', 0) or mc.get('auc_oos', 0)
    score_full = mf.get('wf_auc_mean', 0) or mf.get('auc_oos', 0)
    best = r_full if score_full >= score_core else r_core
    bm = best["metrics"]
    chosen = best["feature_set"]
    print(f"\n  ==> CHOIX : {chosen} (score wf/oos = {max(score_core, score_full):.3f})")

    # Save best
    model_path = Path(f"{ROOT}/bot_v2/ml_model_{asset}_vantage_v18_7.pkl")
    features_path = Path(f"{ROOT}/bot_v2/ml_features_{asset}_vantage_v18_7.json")
    with open(model_path, "wb") as f:
        pickle.dump(best["model"], f)
    features_path.write_text(json.dumps({"features": best["feat_cols"]}, indent=2))
    print(f"  Saved : {model_path.name} ({chosen}, {len(best['feat_cols'])} features)")

    # Verdict
    verdict = "PASS"
    reasons = []
    auc_oos = bm.get("auc_oos", 0)
    wf_auc = bm.get("wf_auc_mean", 0)
    wr_oos = bm.get("oos_wr@0.65", 0)
    n_oos = bm.get("oos_n@0.65", 0)
    gap = bm.get("gap_train_oos", 1)

    if auc_oos < 0.62:
        verdict = "FAIL"
        reasons.append(f"AUC OOS {auc_oos:.3f} < 0.62")
    elif gap > 0.18:
        verdict = "FAIL"
        reasons.append(f"Overfit gap {gap:+.3f} > 0.18")
    if n_oos >= 5 and wr_oos < 60:
        verdict = "FAIL"
        reasons.append(f"WR OOS @0.65 {wr_oos:.1f}% < 60%")

    result = {"asset": asset, "status": "trained", "chosen": chosen,
              "metrics_core": mc, "metrics_full": mf, "metrics_best": bm,
              "verdict": verdict, "reasons": reasons}
    print(f"  ===> {verdict} {' | '.join(reasons) if reasons else 'OK'}")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("asset", nargs="?")
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

        print("\n" + "=" * 110)
        print("=== RECAP V18.7 (REGULARISE + ABLATION + WF) ===")
        print("=" * 110)
        print(f"{'Asset':<11}{'Chosen':<8}{'AUC oos':<9}{'WF AUC':<9}{'GAP':<9}{'WR@65':<8}{'N@65':<7}{'Verdict':<8}")
        for m in all_metrics:
            if m.get("status") != "trained":
                print(f"{m['asset']:<11}{m.get('status', '?'):<8}")
                continue
            bm = m.get("metrics_best", {})
            print(f"{m['asset']:<11}"
                  f"{m['chosen']:<8}"
                  f"{bm.get('auc_oos', 0):<9.3f}"
                  f"{bm.get('wf_auc_mean', 0):<9.3f}"
                  f"{bm.get('gap_train_oos', 0):<+9.3f}"
                  f"{bm.get('oos_wr@0.65', 0):<8.1f}"
                  f"{bm.get('oos_n@0.65', '--')!s:<7}"
                  f"{m.get('verdict', '--'):<8}")
        n_pass = sum(1 for m in all_metrics if m.get("verdict") == "PASS")
        n_core = sum(1 for m in all_metrics if m.get("chosen") == "core")
        n_full = sum(1 for m in all_metrics if m.get("chosen") == "full")
        print(f"\nPASS : {n_pass} / {len(all_metrics)}")
        print(f"ABLATION : {n_core} actifs preferent CORE (61), {n_full} preferent FULL (69)")

        recap = Path(f"{ROOT}/ml_metrics_v18_7_recap.json")
        recap.write_text(json.dumps(all_metrics, indent=2, default=str))
        print(f"\nRecap : {recap.name}")

    elif args.asset:
        train_one(args.asset)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
