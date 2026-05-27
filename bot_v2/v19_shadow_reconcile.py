"""V19 SHADOW RECONCILE : remplit les outcomes des predictions PENDING.

Pour chaque prediction PENDING dans shadow_predictions.jsonl :
- Charge les bougies M1 du broker (live ou Vantage historique)
- Verifie si TP ou SL touche dans la suite
- Met a jour outcome = WIN/LOSS/PENDING

Usage :
    python -m bot_v2.v19_shadow_reconcile  # met a jour outcomes
    python -m bot_v2.v19_shadow_reconcile --summary  # imprime stats apres reconcile
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = "c:/Users/Shadow/TradingBot" if sys.platform == "win32" else "/workspace/TradingBot"
sys.path.insert(0, ROOT)


SHADOW_LOG = Path(f"{ROOT}/shadow_predictions.jsonl")


def load_m1_data(instrument: str) -> pd.DataFrame | None:
    """Charge les data M1 pour determiner outcome."""
    for cand in [
        Path(f"{ROOT}/data_vantage/{instrument}_M1.parquet"),
        Path(f"{ROOT}/data/data_vantage/{instrument}_M1.parquet"),
    ]:
        if cand.exists():
            df = pd.read_parquet(cand)
            if "ts" in df.columns:
                df["ts"] = pd.to_datetime(df["ts"], utc=True)
                df = df.set_index("ts")
            df.index = pd.to_datetime(df.index, utc=True)
            return df.sort_index()
    return None


def determine_outcome(record: dict, df_m1: pd.DataFrame, max_bars: int = 1440) -> str:
    """Verifie si TP ou SL touche dans les max_bars bougies suivant ob_validation_ts.

    Returns : "WIN", "LOSS", "PENDING", "AMBIGUOUS"
    """
    ob_ts = pd.to_datetime(record.get("ob_validation_ts") or record["ts"], utc=True)
    entry = record["entry"]
    sl = record["sl"]
    tp = record["tp"]
    direction = record["direction"]

    future = df_m1[df_m1.index > ob_ts].head(max_bars)
    if len(future) == 0:
        return "PENDING"

    # Premiere bougie d'entry : entry est dans range -> on enter
    # Puis on cherche TP/SL touche
    entered = False
    for ts, row in future.iterrows():
        low, high = row["low"], row["high"]

        if not entered:
            # Entry au prix entry (pending order)
            if direction == "bullish" and low <= entry <= high:
                entered = True
            elif direction == "bearish" and low <= entry <= high:
                entered = True
            else:
                continue

        # On a entered. Verifie TP/SL touche dans cette bougie
        if direction == "bullish":
            sl_touched = low <= sl
            tp_touched = high >= tp
        else:
            sl_touched = high >= sl
            tp_touched = low <= tp

        if sl_touched and tp_touched:
            return "AMBIGUOUS"  # les deux dans la meme bougie
        if sl_touched:
            return "LOSS"
        if tp_touched:
            return "WIN"

    return "PENDING"  # max_bars depasse, pas de close


def reconcile():
    if not SHADOW_LOG.exists():
        print("Pas de shadow log a reconcilier.")
        return

    records = []
    with open(SHADOW_LOG) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        return

    # Cache par instrument pour eviter de recharger
    df_cache: dict[str, pd.DataFrame | None] = {}

    n_updated = 0
    for r in records:
        if r.get("outcome") not in ("PENDING", None):
            continue  # deja resolu
        inst = r["instrument"]
        if inst not in df_cache:
            df_cache[inst] = load_m1_data(inst)
        if df_cache[inst] is None:
            continue
        new_outcome = determine_outcome(r, df_cache[inst])
        if new_outcome != "PENDING":
            r["outcome"] = new_outcome
            n_updated += 1

    # Rewrite jsonl
    with open(SHADOW_LOG, "w") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")

    print(f"Reconcile : {n_updated} predictions resolues.")
    n_pending = sum(1 for r in records if r.get("outcome") == "PENDING")
    n_win = sum(1 for r in records if r.get("outcome") == "WIN")
    n_loss = sum(1 for r in records if r.get("outcome") == "LOSS")
    n_amb = sum(1 for r in records if r.get("outcome") == "AMBIGUOUS")
    print(f"Total : {len(records)} | WIN={n_win} LOSS={n_loss} PENDING={n_pending} AMB={n_amb}")
    if n_win + n_loss > 0:
        wr = n_win / (n_win + n_loss) * 100
        print(f"WR brut : {wr:.1f}%")


def summary():
    """Stats par seuil ML, separement V19 et V18."""
    from bot_v2.v19_shadow import load_predictions
    records = load_predictions()
    closed = [r for r in records if r.get("outcome") in ("WIN", "LOSS")]

    print(f"=== SHADOW SUMMARY ({len(records)} predictions, {len(closed)} closed) ===\n")

    for model_key in ("proba_v19", "proba_v18"):
        print(f"### {model_key} ###")
        for thr in (0.55, 0.60, 0.65, 0.70, 0.75):
            picks = [r for r in closed
                     if r.get(model_key) is not None and r[model_key] >= thr]
            if not picks:
                continue
            wins = sum(1 for r in picks if r["outcome"] == "WIN")
            wr = wins / len(picks) * 100
            print(f"  thr={thr:.2f} : N={len(picks):<5} WR={wr:.1f}%")
        print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", action="store_true")
    args = p.parse_args()

    if args.summary:
        summary()
    else:
        reconcile()
        summary()


if __name__ == "__main__":
    main()
