"""WR global V8 sur OOS a chaque seuil — pour choisir le seuil de prod."""
import pandas as pd, pickle, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ["XAUUSD","NAS100","GER40","BTCUSD","EURUSD","GBPUSD","AUDUSD",
          "USDJPY","SP500","DJ30","UK100","FRA40","USDCAD","USDCHF"]
VAL_END = pd.Timestamp("2025-11-22", tz="UTC")
NON_FEAT = {"instrument","ts","direction","outcome","pnl_usd","bars_to_exit"}

rows = []
for a in ASSETS:
    dsp = ROOT / f"data/ml_dataset_{a}_vantage_v8.parquet"
    mp = ROOT / f"bot_v2/ml_model_{a}_vantage_v8.pkl"
    fp = ROOT / f"bot_v2/ml_features_{a}_vantage_v8.json"
    if not dsp.exists():
        continue
    df = pd.read_parquet(dsp)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df[df["outcome"].isin(["WIN","LOSS"])].copy()
    df["win"] = (df["outcome"]=="WIN").astype(int)
    oos = df[df["ts"] >= VAL_END].copy()
    if len(oos) < 20:
        continue
    model = pickle.load(open(mp,"rb"))
    feats = json.load(open(fp))["features"]
    X = oos[feats].copy()
    for c in X.columns:
        if X[c].dtype == bool: X[c] = X[c].astype(int)
        elif X[c].dtype == "object": X[c] = X[c].astype("category")
    oos["proba"] = model.predict_proba(X)[:,1]
    rows.append(oos[["proba","win"]])

allp = pd.concat(rows, ignore_index=True)
n_days = 180  # OOS ~ 6 mois

print("=" * 60)
print("WR GLOBAL V8 sur OOS 6 mois (14 actifs) par seuil")
print("=" * 60)
print(f"{'Seuil':<8} {'Trades':<9} {'Trades/jour':<13} {'WR':<8}")
print("-" * 45)
for thr in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
    sub = allp[allp["proba"] >= thr]
    n = len(sub)
    wr = sub["win"].mean()*100 if n > 0 else 0
    print(f"{thr:<8.2f} {n:<9} {n/n_days:<13.1f} {wr:.1f}%")
print("-" * 45)
print()
print("Rappel : strategie RR 2-3.")
print("  - RR 2.0 -> rentable si WR > 33%")
print("  - RR 2.5 -> rentable si WR > 29%")
print("  - RR 3.0 -> rentable si WR > 25%")
