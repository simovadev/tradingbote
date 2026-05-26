"""Check si les features qui DRIFT (adversarial) sont les memes que celles
qui PREDISENT (V18.6 ML importance).

Si overlap : V18.6 utilise des features non-stationnaires -> probleme en live.
Si pas d'overlap : V18.6 ignore les features drift -> OK pour live.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = "c:/Users/Shadow/TradingBot" if sys.platform == "win32" else "/workspace/TradingBot"
sys.path.insert(0, ROOT)


def analyse_one(asset: str) -> dict:
    # Top drift features depuis adversarial
    adv = Path(f"{ROOT}/adversarial_recap_v18_6.json")
    if not adv.exists():
        return {"asset": asset, "status": "no_adv"}
    adv_data = json.loads(adv.read_text())
    asset_adv = next((r for r in adv_data if r.get("asset") == asset), None)
    if not asset_adv or asset_adv.get("status") != "ok":
        return {"asset": asset, "status": "adv_fail"}

    top_drift = asset_adv.get("top_drift_features", [])
    drift_set = {r["feature"] for r in top_drift[:10]}
    adv_auc = asset_adv.get("adversarial_auc", 0)

    # V18.6 model importance
    model_path = Path(f"{ROOT}/bot_v2/ml_model_{asset}_vantage_v18_6.pkl")
    feat_path = Path(f"{ROOT}/bot_v2/ml_features_{asset}_vantage_v18_6.json")
    if not model_path.exists() or not feat_path.exists():
        return {"asset": asset, "status": "no_v18_6"}

    with open(model_path, "rb") as f:
        m = pickle.load(f)
    # CalibratedClassifierCV -> .estimator -> .feature_importances_
    try:
        if hasattr(m, "calibrated_classifiers_"):
            base = m.calibrated_classifiers_[0].estimator
        elif hasattr(m, "estimator"):
            base = m.estimator
        else:
            base = m
        importances = base.feature_importances_
    except Exception as e:
        return {"asset": asset, "status": f"model_fail: {e}"}

    feats = json.loads(feat_path.read_text())["features"]
    if len(importances) != len(feats):
        return {"asset": asset, "status": f"mismatch: {len(importances)} vs {len(feats)}"}

    imp_df = pd.DataFrame({"feature": feats, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False)
    top_predict = imp_df.head(10)
    predict_set = set(top_predict["feature"].tolist())

    overlap = drift_set & predict_set
    safe_in_predict = predict_set - drift_set

    return {
        "asset": asset,
        "status": "ok",
        "adv_auc": adv_auc,
        "top_drift": list(drift_set),
        "top_predict": top_predict.to_dict("records"),
        "overlap": list(overlap),
        "n_overlap": len(overlap),
        "safe_features_in_top10_predict": list(safe_in_predict),
        "n_safe_in_predict": len(safe_in_predict),
    }


def main():
    assets = [
        "BTCUSD", "ETHUSD", "USDMXN", "XAUUSD", "USDJPY", "USDZAR", "USDCAD", "NZDUSD",
        "GBPUSD", "CL-OIL", "EURUSD", "USDCHF", "NAS100", "XAGUSD",
        "GAS-C", "HK50", "GER40", "UK100", "SP500", "Coffee-C",
    ]

    print(f"=== CHECK OVERLAP DRIFT vs PREDICT (V18.6) ===")
    print()
    print(f"{'Asset':<11} {'Adv AUC':<9} {'Overlap':<8} {'Top drift (top3)':<60}")
    print(f"{'':>11} {'':>9} {'':>8} {'Top predict (top3)':<60}")
    print("-" * 100)
    results = []
    for a in assets:
        r = analyse_one(a)
        if r.get("status") != "ok":
            print(f"{a:<11} {r.get('status', '?')}")
            continue
        overlap_n = r["n_overlap"]
        drift_top3 = ", ".join(r["top_drift"][:3])
        predict_top3 = ", ".join(p["feature"] for p in r["top_predict"][:3])
        print(f"{a:<11} {r['adv_auc']:<9.4f} {overlap_n:<8} {drift_top3:<60}")
        print(f"{'':>11} {'':>9} {'':>8} {'->' + predict_top3:<60}")
        if overlap_n > 0:
            print(f"           OVERLAP : {', '.join(r['overlap'])}")
        results.append(r)

    # Summary
    print()
    print(f"=== SUMMARY ===")
    n_zero = sum(1 for r in results if r["n_overlap"] == 0)
    n_one = sum(1 for r in results if r["n_overlap"] == 1)
    n_two = sum(1 for r in results if r["n_overlap"] == 2)
    n_three_plus = sum(1 for r in results if r["n_overlap"] >= 3)
    print(f"0 overlap (SAFE)   : {n_zero}")
    print(f"1 overlap (warn)   : {n_one}")
    print(f"2 overlaps (drift) : {n_two}")
    print(f"3+ overlaps (RISK) : {n_three_plus}")

    out = Path(f"{ROOT}/check_overlap_v18_6.json")
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nDetails : {out.name}")


if __name__ == "__main__":
    main()
