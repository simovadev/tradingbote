"""Check les probas du 17 fev DIRECTEMENT depuis le dataset training V5 3 mois.

On charge :
- Le dataset V5 3 mois (deja calcule lors du rebuild)
- Le modele V5 8 ans

Pour chaque OB du 17 fev, on applique le modele et on regarde la proba.
Si max proba > 0.75 -> le code marche, juste le simulate_live_24h donnait differents resultats
Si max proba == 0.68 (comme simulate) -> identique, c'est un mauvais jour
"""
import os
import sys
import json
import pickle
from pathlib import Path

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

DAY = "2026-02-17"
ASSET = "XAUUSD"

dataset_path = Path(f"{ROOT}/data/ml_dataset_{ASSET}_admiral_3mois_V5.parquet")
model_path = Path(f"{ROOT}/bot_v2/ml_model_{ASSET}_admiral_v5.pkl")
features_path = Path(f"{ROOT}/bot_v2/ml_features_{ASSET}_admiral_v5.json")

print(f"\n=== CHECK {ASSET} {DAY} depuis dataset training ===")
print(f"Dataset : {dataset_path.name}")
print(f"Modele  : {model_path.name}")

# Load
df = pd.read_parquet(dataset_path)
df["ts"] = pd.to_datetime(df["ts"], utc=True)
print(f"Dataset total : {len(df):,} candidats")
print(f"Periode : {df['ts'].min()} -> {df['ts'].max()}")

# Filtre jour
day_ts = pd.Timestamp(DAY, tz="UTC")
day_end = day_ts + pd.Timedelta(days=1)
df_day = df[(df["ts"] >= day_ts) & (df["ts"] < day_end)].copy()
print(f"\nCandidats jour {DAY} : {len(df_day)}")

if len(df_day) == 0:
    print("AUCUN candidat ce jour-la dans le dataset")
    sys.exit(0)

# Load modele
with open(model_path, "rb") as f:
    model = pickle.load(f)
features = json.loads(features_path.read_text())["features"]
print(f"Modele V5 : {len(features)} features")

# Predict
X = df_day[features].copy()
for c in X.columns:
    if X[c].dtype == bool:
        X[c] = X[c].astype(int)
X = X.fillna(0)

probas = model.predict_proba(X)[:, 1]
df_day["proba_v5"] = probas

# Stats
print(f"\nProbas du {DAY} (n={len(df_day)}) :")
print(f"  min  = {probas.min():.3f}")
print(f"  max  = {probas.max():.3f}")
print(f"  mean = {probas.mean():.3f}")
print(f"  median = {pd.Series(probas).median():.3f}")

for thr in [0.30, 0.50, 0.65, 0.70, 0.75, 0.80, 0.85]:
    n = (probas >= thr).sum()
    print(f"  >= {thr:.2f} : {n} ({n/len(probas)*100:.1f}%)")

# Top 10 probas avec outcome
print(f"\nTop 10 probas (avec outcome) :")
top = df_day.nlargest(10, "proba_v5")[["ts", "direction", "proba_v5", "outcome", "pnl_usd"]]
print(top.to_string())
