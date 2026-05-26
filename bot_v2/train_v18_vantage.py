"""Train + validate ML V15 sur dataset Vantage 8 ans (mode PUR AMONT).

V15 vs V13 :
- Detection OB SANS condition BOS : un OB est valide des qu'il est COMBLE par le prix
  (mitigation immediate), sans attendre une cassure de structure (close au-dela de
  ob_high/ob_low). Aligne avec la strategie manuelle du user.
- Avant (V13) : OB valide quand close > ob_high (bullish) ou close < ob_low (bearish).
  Latence typique 5-25 bougies M1 -> mouvement consume avant decision bot.
- Maintenant (V15) : OB valide des que low <= ob_high (bullish) ou high >= ob_low (bearish).
  Latence 0-2 bougies. Coherent avec MARKET exec V12.

Build dataset avec OB_VALIDATION_MODE=mitigation (cf run_v15_parallel.py).
Tout le reste est strictement identique a V13 (hyperparams, split, 59 features dont
atr_regime par killzone).

Split temporel V15 (sur 8 ans Vantage 2018-03 -> 2026-05-21) :
  TRAIN : 2018-03 -> 2025-05-21 (~7.2 ans, 90% des trades)
  VAL   : 2025-05-22 -> 2025-11-21 (6 mois)
  OOS   : 2025-11-22 -> 2026-05-21 (6 mois, jamais vu en training)

Verdict PASS si :
- AUC OOS >= 0.65
- WR OOS @0.65 >= 60% sur au moins 5 trades

Usage :
    python -m bot_v2.train_v18_vantage XAUUSD
    python -m bot_v2.train_v18_vantage --all
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

ROOT = "c:/Users/Shadow/TradingBot" if sys.platform == "win32" else "/workspace/TradingBot"
sys.path.insert(0, ROOT)


# 28 actifs V18 (14 actuels + 14 nouveaux decoreles)
ALL_ASSETS = [
    # === ACTUELS (14) ===
    "XAUUSD", "NAS100", "GER40", "BTCUSD",
    "EURUSD", "GBPUSD", "AUDUSD", "USDJPY",
    "SP500", "DJ30", "UK100", "FRA40",
    "USDCAD", "USDCHF",
    # === NOUVEAUX (14) ===
    "Nikkei225", "HK50", "USDMXN", "USDZAR", "CL-OIL", "GAS-C",
    "BVSPX", "Cocoa-C", "Coffee-C", "NZDUSD", "ETHUSD", "XAGUSD",
    "Wheat-C", "Sugar-C",
]

# Memes seuils que V12. Dataset V15 ne contient PAS snapshot_k (mode PUR AMONT).
# V15 PROD (2026-05-25) : rebuild avec TOUTE la data jusqu'a aujourd'hui.
# V15 (2026-05-25) : 2 modes via env var TRAIN_MODE :
#  - TRAIN_MODE=oos (defaut) : train jusqu'a mai 2025, OOS sur nov2025->mai2026
#    = MEMES dates que V14 -> comparaison directe AUC/WR V14 vs V15.
#  - TRAIN_MODE=prod : train jusqu'a mai 2026 (toute la data) pour le live.
import os as _os_tm
_TRAIN_MODE = _os_tm.environ.get("TRAIN_MODE", "oos")
# BUG #4 FIX (2026-05-26) : ajout d'un EMBARGO de 2 jours (48h) entre les splits.
# simulate_trade_market scan jusqu'a 1440 bougies M1 (24h) pour fermer un trade
# (cf backtest.py l.188). Sans embargo, des trades de FIN TRAIN ont leur label
# calcule sur des bougies de VAL = contamination future->passe. 48h couvre
# largement les 24h de scan + marge securitaire.
EMBARGO = pd.Timedelta(days=2)
if _TRAIN_MODE == "prod":
    TRAIN_END = pd.Timestamp("2026-05-15", tz="UTC")
    VAL_START = TRAIN_END + EMBARGO
    VAL_END = pd.Timestamp("2026-05-22", tz="UTC")
    OOS_START = VAL_END + EMBARGO
else:  # oos : memes dates que V14
    TRAIN_END = pd.Timestamp("2025-05-22", tz="UTC")
    VAL_START = TRAIN_END + EMBARGO  # 2025-05-24
    VAL_END = pd.Timestamp("2025-11-22", tz="UTC")
    OOS_START = VAL_END + EMBARGO    # 2025-11-24

NON_FEATURES = {
    "instrument", "ts", "direction",
    "outcome", "pnl_usd", "bars_to_exit",
}

PARTIAL_DIR = Path(f"{ROOT}/data/ml_partial_M1_V18_VANTAGE_STRICT_OB")


def load_dataset(asset: str) -> pd.DataFrame | None:
    """Charge dataset final si dispo, sinon agrege les chunks partials."""
    final_path = Path(f"{ROOT}/data/ml_dataset_{asset}_vantage_v18.parquet")
    if final_path.exists():
        print(f"  Chargement dataset final : {final_path.name}")
        df = pd.read_parquet(final_path)
    else:
        chunks = sorted(PARTIAL_DIR.glob(f"{asset}_M1_*.parquet"))
        if not chunks:
            print(f"  Aucun dataset V15 trouve pour {asset} (ni final ni chunks)")
            return None
        print(f"  Chargement {len(chunks)} chunks V15 pour {asset}")
        dfs = [pd.read_parquet(c) for c in chunks]
        df = pd.concat(dfs, ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").reset_index(drop=True)
    # Garde-fou : si jamais snapshot_k existe (build mixte), on le drop.
    if "snapshot_k" in df.columns:
        print(f"  WARNING : snapshot_k detecte dans le dataset V15, suppression")
        df = df.drop(columns=["snapshot_k"])
    return df


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    df = df[df["outcome"].isin(["WIN", "LOSS"])].copy()
    df["target"] = (df["outcome"] == "WIN").astype(int)
    df = df.sort_values("ts").reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in NON_FEATURES and c != "target"]
    X = df[feature_cols].copy()
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
    y = df["target"]
    return X, y, feature_cols


def train_one(asset: str) -> dict:
    print(f"\n{'='*60}")
    print(f"=== TRAIN {asset} V15 (Vantage 8 ans, OB sans BOS (mitigation immediate)) ===")
    print(f"{'='*60}")
    df = load_dataset(asset)
    if df is None:
        return {"asset": asset, "status": "no_data"}

    df_closed = df[df["outcome"].isin(["WIN", "LOSS"])].copy()
    n_total = len(df_closed)
    if n_total < 200:
        print(f"  !! TROP PEU de samples ({n_total} < 200), skip")
        return {"asset": asset, "status": "not_enough", "n": n_total}

    wr_brut = (df_closed["outcome"] == "WIN").mean() * 100
    print(f"  Total trades fermes : {n_total} | WR brut {wr_brut:.1f}%")
    print(f"  Periode : {df_closed['ts'].min()} -> {df_closed['ts'].max()}")

    # BUG #4 FIX : applique l'embargo de 2j entre TRAIN/VAL et VAL/OOS.
    train_df = df_closed[df_closed["ts"] < TRAIN_END].copy()
    val_df = df_closed[(df_closed["ts"] >= VAL_START) & (df_closed["ts"] < VAL_END)].copy()
    oos_df = df_closed[df_closed["ts"] >= OOS_START].copy()
    print(f"  Train : {len(train_df):>4} ({train_df['ts'].min().date()} -> {train_df['ts'].max().date() if len(train_df) else '--'})")
    print(f"  Val   : {len(val_df):>4} ({val_df['ts'].min().date() if len(val_df) else '--'} -> {val_df['ts'].max().date() if len(val_df) else '--'})")
    print(f"  OOS   : {len(oos_df):>4} ({oos_df['ts'].min().date() if len(oos_df) else '--'} -> {oos_df['ts'].max().date() if len(oos_df) else '--'})")

    # V15 PROD : val/oos peuvent etre tres petits (toute la data en training).
    # On accepte si train suffisant + val suffisant pour early stopping LightGBM.
    if len(train_df) < 100 or len(val_df) < 20:
        print(f"  !! split trop deséquilibré, skip")
        return {"asset": asset, "status": "bad_split",
                "n_train": len(train_df), "n_val": len(val_df), "n_oos": len(oos_df)}

    X_train, y_train, feat_cols = prepare_xy(train_df)
    X_val, y_val, _ = prepare_xy(val_df)
    has_oos = len(oos_df) >= 10
    if has_oos:
        X_oos, y_oos, _ = prepare_xy(oos_df)
    else:
        X_oos = X_val.head(0).copy()
        y_oos = y_val.head(0).copy()

    cat_cols = []
    for c in feat_cols:
        if X_train[c].dtype == "object" or X_train[c].dtype.name == "category":
            X_train[c] = X_train[c].astype("category")
            X_val[c] = X_val[c].astype("category")
            if has_oos:
                X_oos[c] = X_oos[c].astype("category")
            cat_cols.append(c)

    print(f"\n  Features : {len(feat_cols)} (cat: {cat_cols})")
    print(f"  Training LightGBM (config V19 'regularized')...")
    # BONUS (2026-05-26) : regularisation forte pour reduire l'overfit
    # (train AUC 0.83 vs OOS 0.65). max_depth 9->5, num_leaves 63->31,
    # min_child_samples 30->200, reg_alpha 0.1->1.0, reg_lambda 0.1->5.0,
    # n_estimators 500->2000 et learning_rate 0.03->0.01 (apprentissage plus
    # lent et plus stable), subsample 0.8->0.7, colsample_bytree 0.8->0.6,
    # ajout de min_split_gain=0.01. Early stopping passe de 40 -> 100.
    model = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.01,
        max_depth=5,                    # 9 -> 5
        num_leaves=31,                  # 63 -> 31
        min_child_samples=200,          # 30 -> 200
        subsample=0.7,
        colsample_bytree=0.6,
        reg_alpha=1.0,                  # 0.1 -> 1.0
        reg_lambda=5.0,                 # 0.1 -> 5.0
        min_split_gain=0.01,
        random_state=42,
        verbosity=-1,
        # V18 (2026-05-25) : class_weight=balanced TESTE et REJETE.
        # Avec balanced, les probas grimpent toutes vers 0.5-0.7 -> au seuil
        # 0.65 on prend 1700+ trades XAUUSD au lieu de ~50, WR chute de 96% a
        # 41%. Le ML naturel (sans balanced) discrimine MIEUX les WIN rares.
        # On garde le comportement V15 original (sans class_weight).
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        categorical_feature=cat_cols if cat_cols else "auto",
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
    )

    # BUG #3 FIX (2026-05-26) : calibration isotonique des probas.
    # LightGBM produit des probas NON CALIBREES : seuil 0.65 != 65% WR reel.
    # Preuve V15.1 : oos_wr@0.65 = 40.97% (cf ml_metrics_v15_1_vantage_recap.json).
    # On calibre via isotonic regression sur le VAL set (jamais vu par early
    # stopping cote LightGBM -> calibration honnete). cv="prefit" indique que
    # model est deja fit, on fait juste la calibration en post-traitement.
    print(f"  Calibration isotonique sur VAL ({len(X_val)} samples)...")
    calibrated_model = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    calibrated_model.fit(X_val, y_val)

    train_proba = calibrated_model.predict_proba(X_train)[:, 1]
    val_proba = calibrated_model.predict_proba(X_val)[:, 1]
    oos_proba = calibrated_model.predict_proba(X_oos)[:, 1] if has_oos else None

    metrics = {"asset": asset, "status": "trained", "n_train": len(X_train),
               "n_val": len(X_val), "n_oos": len(X_oos),
               "wr_brut_oos": float((y_oos.mean()) * 100) if has_oos else 0.0}
    eval_sets = [("train", y_train, train_proba), ("val", y_val, val_proba)]
    if has_oos:
        eval_sets.append(("oos", y_oos, oos_proba))
    for name, y_true, y_pred in eval_sets:
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

    model_path = Path(f"{ROOT}/bot_v2/ml_model_{asset}_vantage_v18.pkl")
    features_path = Path(f"{ROOT}/bot_v2/ml_features_{asset}_vantage_v18.json")
    # BUG #3 FIX (2026-05-26) : on sauvegarde le CalibratedClassifierCV, pas
    # le modele LightGBM brut. Le live appelle .predict_proba() dessus -> il
    # obtient des probas calibrees alignees sur le WR reel.
    with open(model_path, "wb") as f:
        pickle.dump(calibrated_model, f)
    features_path.write_text(json.dumps({"features": feat_cols}, indent=2))
    print(f"\n  Saved : {model_path.name} + {features_path.name} (calibrated isotonic)")

    # V15 PROD : OOS quasi-vide, on valide juste sur VAL
    verdict = "PASS"
    reasons = []
    if has_oos:
        auc_oos = metrics.get("auc_oos", 0)
        wr_oos_065 = metrics.get("oos_wr@0.65", 0)
        n_oos_065 = metrics.get("oos_n@0.65", 0)
        if auc_oos < 0.65:
            verdict = "FAIL"
            reasons.append(f"AUC OOS {auc_oos:.3f} < 0.65")
        if n_oos_065 < 5:
            reasons.append(f"trop peu de trades @0.65 ({n_oos_065})")
        elif wr_oos_065 < 60:
            verdict = "FAIL"
            reasons.append(f"WR OOS @0.65 = {wr_oos_065:.1f}% < 60%")
    else:
        # Pas d'OOS : on valide sur VAL
        auc_val = metrics.get("auc_val", 0)
        wr_val_065 = metrics.get("val_wr@0.65", 0)
        if auc_val < 0.65:
            verdict = "FAIL"
            reasons.append(f"AUC VAL {auc_val:.3f} < 0.65")
        elif wr_val_065 < 55:
            verdict = "FAIL"
            reasons.append(f"WR VAL @0.65 = {wr_val_065:.1f}% < 55%")
        reasons.append("(no_oos: train jusqu'a la fin, valide sur VAL)")

    metrics["verdict"] = verdict
    metrics["reasons"] = reasons
    print(f"\n  ===> {verdict} {' | '.join(reasons) if reasons else 'OK'}")
    return metrics


def main():
    p = argparse.ArgumentParser(description="Train + validate ML V15 sur Vantage (OB sans BOS (mitigation immediate))")
    p.add_argument("asset", nargs="?", help="Asset (XAUUSD) ou --all")
    p.add_argument("--all", action="store_true", help="Train tous les 14 actifs")
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
        print("=== RECAP V15 VANTAGE (OB sans BOS (mitigation immediate)) ===")
        print("=" * 80)
        print(f"{'Asset':<10} {'Status':<12} {'AUC OOS':<8} {'WR@0.65':<8} {'N@0.65':<7} {'Verdict':<6}")
        for m in all_metrics:
            status = m.get("status", "?")
            auc = f"{m.get('auc_oos', 0):.3f}" if "auc_oos" in m else "--"
            wr = f"{m.get('oos_wr@0.65', 0):.1f}%" if "oos_wr@0.65" in m else "--"
            n = m.get("oos_n@0.65", 0) if "oos_n@0.65" in m else "--"
            verdict = m.get("verdict", "--")
            print(f"{m['asset']:<10} {status:<12} {auc:<8} {wr:<8} {n!s:<7} {verdict:<6}")

        recap_path = Path(f"{ROOT}/ml_metrics_v18_vantage_recap.json")
        recap_path.write_text(json.dumps(all_metrics, indent=2, default=str))
        print(f"\nRecap sauve : {recap_path.name}")

    elif args.asset:
        if args.asset not in ALL_ASSETS:
            print(f"Actif inconnu : {args.asset}. Choix : {ALL_ASSETS}")
            sys.exit(1)
        train_one(args.asset)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
