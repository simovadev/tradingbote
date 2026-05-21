"""Validation V8 : applique chaque modele V8 sur la periode OOS de son dataset.

Pas de re-backtest : on reutilise les datasets V8 deja construits
(data/ml_dataset_{asset}_vantage_v8.parquet) qui contiennent deja les
features + le label WIN/LOSS de chaque OB.

Pour chaque actif :
- Charge le dataset V8
- Filtre la periode OOS (>= VAL_END, jamais vue en training)
- Applique le modele V8 -> proba ML
- Calcule WR reel + nb trades a chaque seuil 0.55/0.60/0.65/0.70/0.75

Si V8 est bien calibre, le WR a 0.75 doit etre eleve (>75%) et il doit
rester un nombre raisonnable de trades.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ALL_ASSETS = [
    "XAUUSD", "NAS100", "GER40", "BTCUSD",
    "EURUSD", "GBPUSD", "AUDUSD", "USDJPY",
    "SP500", "DJ30", "UK100", "FRA40",
    "USDCAD", "USDCHF",
]

# Meme split que train_v8_vantage.py
VAL_END = pd.Timestamp("2025-11-22", tz="UTC")  # OOS = >= cette date

NON_FEATURES = {"instrument", "ts", "direction", "outcome", "pnl_usd", "bars_to_exit"}


def validate_asset(asset: str) -> dict:
    ds_path = ROOT / f"data/ml_dataset_{asset}_vantage_v8.parquet"
    model_path = ROOT / f"bot_v2/ml_model_{asset}_vantage_v8.pkl"
    feat_path = ROOT / f"bot_v2/ml_features_{asset}_vantage_v8.json"

    if not ds_path.exists() or not model_path.exists():
        return {"asset": asset, "status": "missing"}

    df = pd.read_parquet(ds_path)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df[df["outcome"].isin(["WIN", "LOSS"])].copy()
    df["target"] = (df["outcome"] == "WIN").astype(int)

    # Periode OOS uniquement
    oos = df[df["ts"] >= VAL_END].copy()
    if len(oos) < 20:
        return {"asset": asset, "status": "oos_too_small", "n": len(oos)}

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    features = json.loads(feat_path.read_text())["features"]

    X = oos[features].copy()
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
        elif X[c].dtype == "object":
            X[c] = X[c].astype("category")
    y = oos["target"]
    proba = model.predict_proba(X)[:, 1]

    res = {"asset": asset, "status": "ok", "n_oos": len(oos),
           "wr_brut_oos": float(y.mean() * 100),
           "oos_period": f"{oos['ts'].min().date()} -> {oos['ts'].max().date()}"}
    for thr in (0.55, 0.60, 0.65, 0.70, 0.75):
        mask = proba >= thr
        n = int(mask.sum())
        res[f"n@{thr}"] = n
        res[f"wr@{thr}"] = float(y[mask].mean() * 100) if n >= 5 else None
    return res


def main():
    print("=" * 88)
    print("=== VALIDATION V8 sur OOS (periode jamais vue en training, >= 2025-11-22) ===")
    print("=" * 88)
    print(f"{'Asset':<9} {'N_OOS':<7} {'WRbrut':<8} {'WR@.65':<9} {'N@.65':<7} "
          f"{'WR@.75':<9} {'N@.75':<7}")
    print("-" * 70)

    rows = []
    for a in ALL_ASSETS:
        try:
            r = validate_asset(a)
        except Exception as e:
            print(f"{a:<9} ERREUR : {e}")
            continue
        rows.append(r)
        if r["status"] != "ok":
            print(f"{a:<9} {r['status']}")
            continue
        wr65 = f"{r['wr@0.65']:.1f}%" if r['wr@0.65'] is not None else "--"
        wr75 = f"{r['wr@0.75']:.1f}%" if r['wr@0.75'] is not None else "--"
        print(f"{a:<9} {r['n_oos']:<7} {r['wr_brut_oos']:.1f}%   {wr65:<9} "
              f"{r['n@0.65']:<7} {wr75:<9} {r['n@0.75']:<7}")

    ok = [r for r in rows if r["status"] == "ok"]
    if ok:
        # Moyennes ponderees
        tot_n75 = sum(r["n@0.75"] for r in ok)
        tot_win75 = sum((r["wr@0.75"] or 0) / 100 * r["n@0.75"] for r in ok if r["wr@0.75"])
        tot_n65 = sum(r["n@0.65"] for r in ok)
        tot_win65 = sum((r["wr@0.65"] or 0) / 100 * r["n@0.65"] for r in ok if r["wr@0.65"])
        print("-" * 70)
        print(f"GLOBAL OOS : {len(ok)} actifs")
        if tot_n65:
            print(f"  Seuil 0.65 : {tot_n65} trades, WR global {tot_win65/tot_n65*100:.1f}%")
        if tot_n75:
            print(f"  Seuil 0.75 : {tot_n75} trades, WR global {tot_win75/tot_n75*100:.1f}%")
        # Trades/jour estimes (OOS ~ 6 mois = 180 jours)
        if tot_n75:
            print(f"  ~{tot_n75/180:.1f} trades/jour @0.75 sur les 14 actifs")
        if tot_n65:
            print(f"  ~{tot_n65/180:.1f} trades/jour @0.65 sur les 14 actifs")


if __name__ == "__main__":
    main()
