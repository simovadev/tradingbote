"""Check les probas + trades sur la derniere semaine (7 derniers jours)
pour TOUS les actifs depuis les datasets V5 8 ans.

Affiche par actif et global :
- Combien de candidats par jour
- Combien de trades a chaque seuil (0.55, 0.65, 0.70, 0.75, 0.80)
- WR par seuil
- Distribution probas
"""
import os
import sys
import json
import pickle
from pathlib import Path

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# Periode : derniere semaine
END_TS = pd.Timestamp("2026-05-19", tz="UTC")
START_TS = END_TS - pd.Timedelta(days=7)

ASSETS = [
    "XAUUSD", "NAS100", "GER40", "BTCUSD",
    "EURUSD", "GBPUSD", "AUDUSD", "USDJPY",
    "SP500", "DJ30", "UK100", "FRA40",
    "USDCAD", "USDCHF",
]

THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


def check_asset(asset):
    dataset_path = Path(f"{ROOT}/data/ml_dataset_{asset}_admiral_8ans_V5.parquet")
    model_path = Path(f"{ROOT}/bot_v2/ml_model_{asset}_admiral_v5.pkl")
    features_path = Path(f"{ROOT}/bot_v2/ml_features_{asset}_admiral_v5.json")

    if not dataset_path.exists():
        return None

    df = pd.read_parquet(dataset_path)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    # Filtre semaine
    df_week = df[(df["ts"] >= START_TS) & (df["ts"] <= END_TS)].copy()
    if len(df_week) == 0:
        return None

    # Load modele
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    features = json.loads(features_path.read_text())["features"]

    # Predict
    X = df_week[features].copy()
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
    X = X.fillna(0)
    probas = model.predict_proba(X)[:, 1]
    df_week["proba"] = probas

    # Stats par seuil
    result = {
        "asset": asset,
        "candidats": len(df_week),
        "max_proba": probas.max(),
        "mean_proba": probas.mean(),
    }
    for thr in THRESHOLDS:
        mask = df_week["proba"] >= thr
        n_trades = mask.sum()
        if n_trades > 0:
            closed = df_week[mask & df_week["outcome"].isin(["WIN", "LOSS"])]
            n_win = (closed["outcome"] == "WIN").sum()
            n_loss = (closed["outcome"] == "LOSS").sum()
            wr = n_win / (n_win + n_loss) * 100 if (n_win + n_loss) > 0 else 0
            pnl = df_week[mask]["pnl_usd"].sum() if "pnl_usd" in df_week else 0
            result[f"trades_{thr}"] = n_trades
            result[f"wr_{thr}"] = wr
            result[f"pnl_{thr}"] = pnl
        else:
            result[f"trades_{thr}"] = 0
            result[f"wr_{thr}"] = 0
            result[f"pnl_{thr}"] = 0

    return result


def main():
    print(f"\n{'='*100}")
    print(f"CHECK SEMAINE {START_TS.date()} -> {END_TS.date()} - TOUS ACTIFS")
    print(f"{'='*100}\n")

    results = []
    for asset in ASSETS:
        r = check_asset(asset)
        if r:
            results.append(r)

    # Tableau par actif
    print(f"{'ASSET':<8} {'CAND':>5} {'MaxP':>6} {'MeanP':>6} | " +
          " | ".join([f"{'@'+str(t):>15}" for t in THRESHOLDS]))
    print("-" * 130)

    totaux = {f"trades_{t}": 0 for t in THRESHOLDS}
    totaux.update({f"wins_{t}": 0 for t in THRESHOLDS})
    totaux.update({f"losses_{t}": 0 for t in THRESHOLDS})
    totaux.update({f"pnl_{t}": 0.0 for t in THRESHOLDS})

    for r in results:
        line = f"{r['asset']:<8} {r['candidats']:>5} {r['max_proba']:>5.3f} {r['mean_proba']:>5.3f} |"
        for thr in THRESHOLDS:
            n = r[f"trades_{thr}"]
            wr = r[f"wr_{thr}"]
            pnl = r[f"pnl_{thr}"]
            if n > 0:
                line += f" {n:>3}T WR={wr:>4.0f}% {pnl:>+5.0f}\$ |"
                totaux[f"trades_{thr}"] += n
                totaux[f"pnl_{thr}"] += pnl
                # Recompute wins/losses
                # On reload pour avoir le breakdown WIN/LOSS
            else:
                line += f" {'-':>15} |"
        print(line)

    print("-" * 130)

    # Total
    print(f"\n{'TOTAL 14 ACTIFS (7 jours)':<8}")
    for thr in THRESHOLDS:
        n = totaux[f"trades_{thr}"]
        pnl = totaux[f"pnl_{thr}"]
        if n > 0:
            print(f"  Seuil {thr:.2f} : {n:>4} trades sur 7j ({n/7:.1f}/jour) | PnL total: {pnl:+.2f}$")
        else:
            print(f"  Seuil {thr:.2f} : 0 trades")


if __name__ == "__main__":
    main()
