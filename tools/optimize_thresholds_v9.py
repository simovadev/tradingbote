"""Optimisation du seuil ML par actif - V9.

Pour chaque actif :
1. Charge le dataset V9 (data/ml_dataset_{asset}_vantage_v9.parquet).
2. Filtre la periode OOS (2025-11-22 -> 2026-05-21, jamais vue en training).
3. Recalcule les probas du modele V9 sur l'OOS.
4. Pour chaque seuil (0.50/0.55/0.60/0.65/0.70/0.75) : nb trades, WR, PnL.
5. Determine le seuil optimal (meilleur compromis WR x volume x PnL).

Le but : trouver le seuil qui maximise le PnL tout en gardant assez de trades.
Un actif qui a WR 85% a 0.75 mais seulement 3 trades n'est pas exploitable.

Usage : python tools/optimize_thresholds_v9.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ALL_ASSETS = [
    "XAUUSD", "NAS100", "GER40", "BTCUSD",
    "EURUSD", "GBPUSD", "AUDUSD", "USDJPY",
    "SP500", "DJ30", "UK100", "FRA40",
    "USDCAD", "USDCHF",
]

# Periode OOS = jamais vue pendant le training V9
OOS_START = pd.Timestamp("2025-11-22", tz="UTC")

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]

NON_FEATURES = {
    "instrument", "ts", "direction",
    "outcome", "pnl_usd", "bars_to_exit", "target",
}


def analyse_asset(asset: str) -> dict | None:
    ds_path = ROOT / f"data/ml_dataset_{asset}_vantage_v9.parquet"
    model_path = ROOT / f"bot_v2/ml_model_{asset}_vantage_v9.pkl"
    feat_path = ROOT / f"bot_v2/ml_features_{asset}_vantage_v9.json"

    if not ds_path.exists() or not model_path.exists():
        print(f"  {asset:8s} : dataset ou modele manquant, skip")
        return None

    df = pd.read_parquet(ds_path)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df[df["outcome"].isin(["WIN", "LOSS"])].copy()

    # OOS uniquement
    oos = df[df["ts"] >= OOS_START].copy()
    if len(oos) < 30:
        print(f"  {asset:8s} : OOS trop court ({len(oos)} trades), skip")
        return None

    model = joblib.load(model_path)
    features = json.load(open(feat_path))["features"]

    X = oos[features].copy()
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
    proba = model.predict_proba(X)[:, 1]
    oos = oos.assign(proba=proba)
    oos["win"] = (oos["outcome"] == "WIN").astype(int)

    # PnL : si la colonne pnl_usd existe, on l'utilise. Sinon WR-based.
    has_pnl = "pnl_usd" in oos.columns

    n_months = (oos["ts"].max() - oos["ts"].min()).days / 30.0
    results = {"asset": asset, "n_oos": len(oos), "n_months": n_months, "by_threshold": {}}

    for thr in THRESHOLDS:
        sel = oos[oos["proba"] >= thr]
        n = len(sel)
        if n == 0:
            results["by_threshold"][thr] = {"n": 0, "wr": 0, "trades_per_month": 0, "pnl": 0}
            continue
        wr = sel["win"].mean() * 100
        tpm = n / n_months if n_months > 0 else 0
        pnl = float(sel["pnl_usd"].sum()) if has_pnl else None
        results["by_threshold"][thr] = {
            "n": n, "wr": wr, "trades_per_month": tpm, "pnl": pnl,
        }
    return results


# Cible : ~1 trade/heure tous actifs confondus.
# Bot tradable ~14h/jour (killzones) -> ~14 trades/jour -> ~420/mois.
# Reparti sur 14 actifs -> cible ~30 trades/mois/actif.
TARGET_TPM = 30.0       # trades/mois vise par actif
TPM_MIN = 12.0          # en-dessous : volume trop faible, on rejette le seuil
WR_FLOOR = 60.0         # WR plancher : un seuil sous 60% n'est pas exploitable


def pick_optimal(res: dict) -> tuple[float, str]:
    """Choisit le seuil : vise ~1 trade/h (TARGET_TPM/actif) avec le meilleur WR.

    Logique :
    - On ne garde que les seuils avec WR >= WR_FLOOR et volume >= TPM_MIN.
    - Parmi eux, on prend celui dont le volume est le plus proche de TARGET_TPM
      (pour ne pas surcharger ni sous-utiliser le bot).
    - A volume egal, on prefere le meilleur WR.
    """
    bt = res["by_threshold"]
    candidates = []
    for thr, m in bt.items():
        if m["n"] < 10:
            continue
        if m["wr"] < WR_FLOOR:
            continue
        if m["trades_per_month"] < TPM_MIN:
            continue
        # Score : penalise l'ecart a la cible de volume, bonus WR
        dist = abs(m["trades_per_month"] - TARGET_TPM)
        score = m["wr"] - dist * 0.5
        candidates.append((score, thr, m))

    if not candidates:
        # Aucun seuil ne respecte WR>=60 ET volume>=12 : on prend le seuil
        # avec le meilleur WR parmi ceux a volume >= TPM_MIN.
        fallback = [(m["wr"], thr, m) for thr, m in bt.items()
                    if m["trades_per_month"] >= TPM_MIN and m["n"] >= 10]
        if fallback:
            fallback.sort(reverse=True)
            _, thr, m = fallback[0]
            return thr, f"fallback WR={m['wr']:.0f}% vol={m['trades_per_month']:.0f}/m"
        return 0.65, "defaut (donnees insuffisantes)"

    candidates.sort(reverse=True)
    _, best_thr, m = candidates[0]
    return best_thr, f"WR={m['wr']:.0f}% vol={m['trades_per_month']:.0f}/m PnL={m['pnl']:+.0f}"


def main():
    print("=" * 90)
    print("OPTIMISATION SEUIL ML V9 - OOS 6 mois (2025-11-22 -> 2026-05-21)")
    print("=" * 90)
    print()

    all_res = []
    optimal = {}
    for asset in ALL_ASSETS:
        res = analyse_asset(asset)
        if res is None:
            continue
        all_res.append(res)

        print(f"### {asset}  (OOS {res['n_oos']} trades, ~{res['n_months']:.1f} mois)")
        print(f"  {'seuil':>6} {'trades':>7} {'/mois':>7} {'WR':>7} {'PnL':>10}")
        for thr in THRESHOLDS:
            m = res["by_threshold"][thr]
            pnl_str = f"{m['pnl']:+.0f}" if m.get("pnl") is not None else "--"
            print(f"  {thr:>6.2f} {m['n']:>7} {m['trades_per_month']:>7.1f} "
                  f"{m['wr']:>6.1f}% {pnl_str:>10}")
        opt_thr, opt_reason = pick_optimal(res)
        optimal[asset] = opt_thr
        print(f"  >>> SEUIL OPTIMAL : {opt_thr:.2f}  ({opt_reason})")
        print()

    # Recap final
    print("=" * 90)
    print("RECAP - SEUIL OPTIMAL PAR ACTIF")
    print("=" * 90)
    print(f"{'Asset':<10} {'Seuil opt':>10} {'Trades/mois':>13} {'WR':>8} {'PnL OOS':>12}")
    total_tpm = 0
    for res in all_res:
        a = res["asset"]
        thr = optimal[a]
        m = res["by_threshold"][thr]
        total_tpm += m["trades_per_month"]
        pnl_str = f"{m['pnl']:+.0f}" if m.get("pnl") is not None else "--"
        print(f"{a:<10} {thr:>10.2f} {m['trades_per_month']:>13.1f} "
              f"{m['wr']:>7.1f}% {pnl_str:>12}")
    print("-" * 90)
    print(f"{'TOTAL':<10} {'':<10} {total_tpm:>13.1f} trades/mois (tous actifs au seuil optimal)")

    # Export JSON pour config
    out = {a: round(thr, 2) for a, thr in optimal.items()}
    out_path = ROOT / "ml_thresholds_v9_optimal.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSeuils optimaux exportes : {out_path.name}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
