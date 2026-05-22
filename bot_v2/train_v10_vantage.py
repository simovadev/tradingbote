"""Train + validate ML V10 sur dataset Vantage 8 ans.

Pour chaque actif :
1. Charge le dataset V10 (data/ml_dataset_{asset}_vantage_v10.parquet ou chunks
   dans data/ml_partial_M1_V10_VANTAGE/).
2. Filtre WIN/LOSS, split temporel TRAIN / VAL / OOS.
3. Entraine LightGBM (memes hyperparams que V5).
4. Calcule AUC sur les 3 sets + WR aux seuils 0.55/0.60/0.65/0.70/0.75.
5. Sauve modele + features + metriques.
6. Imprime rapport pass/fail par actif (AUC OOS >= 0.65 + WR OOS @0.65 >= 60%).

Split temporel V10 (sur 8 ans Vantage 2018-03 -> 2026-05-21) :
  TRAIN : 2018-03 -> 2025-05-22 (~7.2 ans)
  VAL   : 2025-05-22 -> 2025-11-22 (6 mois)
  OOS   : 2025-11-22 -> 2026-05-21 (6 mois, jamais vu en training)

Usage :
    python -m bot_v2.train_v10_vantage XAUUSD
    python -m bot_v2.train_v10_vantage --all
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = "c:/Users/Shadow/TradingBot" if sys.platform == "win32" else "/workspace/TradingBot"
sys.path.insert(0, ROOT)


ALL_ASSETS = [
    "XAUUSD", "NAS100", "GER40", "BTCUSD",
    "EURUSD", "GBPUSD", "AUDUSD", "USDJPY",
    "SP500", "DJ30", "UK100", "FRA40",
    "USDCAD", "USDCHF",
]

# V10 (2026-05-22) : split temporel 7 ans / 6 mois / 6 mois (8 ans Vantage)
# Le dataset couvre ~2018-03 -> 2026-05-21.
#   TRAIN : 2018-03 -> 2025-05-21 (~7.2 ans, 90% des trades)
#   VAL   : 2025-05-22 -> 2025-11-21 (6 mois)
#   OOS   : 2025-11-22 -> 2026-05-21 (6 mois, jamais vu en training)
TRAIN_END = pd.Timestamp("2025-05-22", tz="UTC")
VAL_END = pd.Timestamp("2025-11-22", tz="UTC")
# OOS = >= 2025-11-22

NON_FEATURES = {
    "instrument", "ts", "direction",
    "outcome", "pnl_usd", "bars_to_exit",
}

PARTIAL_DIR = Path(f"{ROOT}/data/ml_partial_M1_V10_VANTAGE")


def load_dataset(asset: str) -> pd.DataFrame | None:
    """Charge dataset final si dispo, sinon agrege les chunks partials."""
    final_path = Path(f"{ROOT}/data/ml_dataset_{asset}_vantage_v10.parquet")
    if final_path.exists():
        print(f"  Chargement dataset final : {final_path.name}")
        df = pd.read_parquet(final_path)
    else:
        chunks = sorted(PARTIAL_DIR.glob(f"{asset}_M1_*.parquet"))
        if not chunks:
            print(f"  Aucun dataset V10 trouve pour {asset} (ni final ni chunks)")
            return None
        print(f"  Chargement {len(chunks)} chunks V10 pour {asset}")
        dfs = [pd.read_parquet(c) for c in chunks]
        df = pd.concat(dfs, ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").reset_index(drop=True)
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
    """Train + evaluate. Retourne un dict de metriques."""
    print(f"\n{'='*60}")
    print(f"=== TRAIN {asset} V10 (Vantage 8 ans, fix DATA LEAKAGES) ===")
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

    # Split par dates
    train_df = df_closed[df_closed["ts"] < TRAIN_END].copy()
    val_df = df_closed[(df_closed["ts"] >= TRAIN_END) & (df_closed["ts"] < VAL_END)].copy()
    oos_df = df_closed[df_closed["ts"] >= VAL_END].copy()
    print(f"  Train : {len(train_df):>4} ({train_df['ts'].min().date()} -> {train_df['ts'].max().date() if len(train_df) else '--'})")
    print(f"  Val   : {len(val_df):>4} ({val_df['ts'].min().date() if len(val_df) else '--'} -> {val_df['ts'].max().date() if len(val_df) else '--'})")
    print(f"  OOS   : {len(oos_df):>4} ({oos_df['ts'].min().date() if len(oos_df) else '--'} -> {oos_df['ts'].max().date() if len(oos_df) else '--'})")

    if len(train_df) < 100 or len(val_df) < 20 or len(oos_df) < 20:
        print(f"  !! split trop deséquilibré, skip")
        return {"asset": asset, "status": "bad_split",
                "n_train": len(train_df), "n_val": len(val_df), "n_oos": len(oos_df)}

    X_train, y_train, feat_cols = prepare_xy(train_df)
    X_val, y_val, _ = prepare_xy(val_df)
    X_oos, y_oos, _ = prepare_xy(oos_df)

    # Categorical detection
    cat_cols = []
    for c in feat_cols:
        if X_train[c].dtype == "object" or X_train[c].dtype.name == "category":
            X_train[c] = X_train[c].astype("category")
            X_val[c] = X_val[c].astype("category")
            X_oos[c] = X_oos[c].astype("category")
            cat_cols.append(c)

    print(f"\n  Features : {len(feat_cols)} (cat: {cat_cols})")
    print(f"  Training LightGBM (config V10 'deeper')...")
    # V10 : config 'deeper' - gagnante au tuning (4 datasets x 4 configs).
    # Plus profond + plus d'arbres pour exploiter les 15 nouvelles features.
    # Resultat tuning : AUC 0.737 (vs 0.72 V9), x2.6 volume @0.65 WR 79%.
    model = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.03, max_depth=9, num_leaves=63,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbosity=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        categorical_feature=cat_cols if cat_cols else "auto",
        callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)],
    )

    # Predictions
    train_proba = model.predict_proba(X_train)[:, 1]
    val_proba = model.predict_proba(X_val)[:, 1]
    oos_proba = model.predict_proba(X_oos)[:, 1]

    # Metrics
    metrics = {"asset": asset, "status": "trained", "n_train": len(X_train),
               "n_val": len(X_val), "n_oos": len(X_oos),
               "wr_brut_oos": float((y_oos.mean()) * 100)}
    for name, y_true, y_pred in [("train", y_train, train_proba),
                                  ("val", y_val, val_proba),
                                  ("oos", y_oos, oos_proba)]:
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

    # Save model + features
    model_path = Path(f"{ROOT}/bot_v2/ml_model_{asset}_vantage_v10.pkl")
    features_path = Path(f"{ROOT}/bot_v2/ml_features_{asset}_vantage_v10.json")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    features_path.write_text(json.dumps({"features": feat_cols}, indent=2))
    print(f"\n  Saved : {model_path.name} + {features_path.name}")

    # Verdict
    auc_oos = metrics.get("auc_oos", 0)
    wr_oos_065 = metrics.get("oos_wr@0.65", 0)
    n_oos_065 = metrics.get("oos_n@0.65", 0)
    verdict = "PASS"
    reasons = []
    if auc_oos < 0.65:
        verdict = "FAIL"
        reasons.append(f"AUC OOS {auc_oos:.3f} < 0.65")
    if n_oos_065 < 5:
        verdict = "FAIL"
        reasons.append(f"trop peu de trades @0.65 ({n_oos_065})")
    elif wr_oos_065 < 60:
        verdict = "FAIL"
        reasons.append(f"WR OOS @0.65 = {wr_oos_065:.1f}% < 60%")

    metrics["verdict"] = verdict
    metrics["reasons"] = reasons
    print(f"\n  ===> {verdict} {' | '.join(reasons) if reasons else 'OK'}")
    return metrics


def main():
    p = argparse.ArgumentParser(description="Train + validate ML V10 sur Vantage")
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

        # Recap final
        print("\n" + "=" * 80)
        print("=== RECAP V10 VANTAGE ===")
        print("=" * 80)
        print(f"{'Asset':<10} {'Status':<12} {'AUC OOS':<8} {'WR@0.65':<8} {'N@0.65':<7} {'Verdict':<6}")
        for m in all_metrics:
            status = m.get("status", "?")
            auc = f"{m.get('auc_oos', 0):.3f}" if "auc_oos" in m else "--"
            wr = f"{m.get('oos_wr@0.65', 0):.1f}%" if "oos_wr@0.65" in m else "--"
            n = m.get("oos_n@0.65", 0) if "oos_n@0.65" in m else "--"
            verdict = m.get("verdict", "--")
            print(f"{m['asset']:<10} {status:<12} {auc:<8} {wr:<8} {n!s:<7} {verdict:<6}")

        # Save recap JSON
        recap_path = Path(f"{ROOT}/ml_metrics_v10_vantage_recap.json")
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
