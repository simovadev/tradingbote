"""Backtest V4 sur les 14 derniers jours - tous trades par jour.

V4 = 14 actifs avec modeles V3.5 + seuil ML 0.70.
Compte 100€, risk 5%, RR=2, MAX_CONCURRENT=8.

Affiche par jour :
  - DATE
  - Heure | Actif | DIR | proba | OUT | PnL
  - Total jour : Trades / WIN / LOSS / PnL
"""
from __future__ import annotations

import sys
sys.path.insert(0, 'c:/Users/Shadow/TradingBot')

import pickle
import json
from pathlib import Path
from collections import defaultdict

import pandas as pd

from bot_v2.data_loader import load
from bot_v2.ml_dataset import _process_instrument


# Config V4
ASSETS = [
    "XAUUSD", "NAS100", "GER40", "BTCUSD",
    "EURUSD", "GBPUSD", "AUDUSD", "USDJPY",
    "SP500", "DJ30", "UK100", "FRA40",
    "USDCAD", "USDCHF",
]

SEUIL_ML = 0.70
RISK_PCT = 0.05
BALANCE_START = 100.0
MAX_CONCURRENT = 8

MODEL_DIR = Path("c:/Users/Shadow/TradingBot/bot_v2")


def load_model_features(asset: str):
    """Charge le model + features V3.5 pour un actif."""
    pkl = MODEL_DIR / f"ml_model_{asset}_admiral_v3_5.pkl"
    feat = MODEL_DIR / f"ml_features_{asset}_admiral_v3_5.json"
    if not pkl.exists() or not feat.exists():
        return None, None
    with open(pkl, "rb") as f:
        model = pickle.load(f)
    with open(feat) as f:
        features = json.load(f)["features"]
    return model, features


def main():
    print("=" * 80)
    print("  BACKTEST V4 - 14 DERNIERS JOURS")
    print("=" * 80)
    print(f"  Actifs : {len(ASSETS)}")
    print(f"  Seuil ML : {SEUIL_ML}")
    print(f"  Risk/trade : {RISK_PCT*100:.0f}%")
    print(f"  RR : 2")
    print(f"  Balance start : {BALANCE_START}€")
    print(f"  MAX_CONCURRENT : {MAX_CONCURRENT}")
    print("=" * 80)
    print()

    # Determine la periode : 14 derniers jours dispo
    df_test = load("XAUUSD", "M1")
    end = df_test.index[-1]
    start = end - pd.Timedelta(days=14)
    print(f"Periode : {start.date()} -> {end.date()}\n")

    # Collecte tous les setups V3.5 sur 14j
    all_trades = []
    for asset in ASSETS:
        model, features = load_model_features(asset)
        if model is None:
            print(f"  [SKIP] {asset} : modele V3.5 non trouve")
            continue

        try:
            rows, _, err = _process_instrument((asset, start, end, "M1"))
            if err or not rows:
                print(f"  [{asset}] 0 setup")
                continue
            df = pd.DataFrame(rows)

            # Filtre WIN/LOSS (NO_FILL exclu)
            df_closed = df[df["outcome"].isin(["WIN", "LOSS"])].copy()
            if len(df_closed) == 0:
                print(f"  [{asset}] 0 trade ferme")
                continue

            # Predict proba ML
            df_closed["ts"] = pd.to_datetime(df_closed["ts"], utc=True)
            X = df_closed[features].copy()
            for c in X.columns:
                if X[c].dtype == bool:
                    X[c] = X[c].astype(int)
            df_closed["proba_ml"] = model.predict_proba(X)[:, 1]

            # Filtre seuil 0.70
            df_keep = df_closed[df_closed["proba_ml"] >= SEUIL_ML].copy()
            print(f"  [{asset}] {len(df_closed)} setups total -> {len(df_keep)} passes seuil {SEUIL_ML}")

            for _, row in df_keep.iterrows():
                all_trades.append({
                    "ts": row["ts"],
                    "asset": asset,
                    "direction": row["direction"],
                    "proba": float(row["proba_ml"]),
                    "outcome": row["outcome"],
                    "rr": float(row["rr"]) if pd.notna(row["rr"]) else 2.0,
                })
        except Exception as e:
            print(f"  [{asset}] ERREUR : {e}")

    if not all_trades:
        print("\nAucun trade selectionne.")
        return

    # Tri par timestamp
    all_trades.sort(key=lambda x: x["ts"])

    # Group by jour
    trades_by_day = defaultdict(list)
    for t in all_trades:
        day = t["ts"].date()
        trades_by_day[day].append(t)

    # Simulation compound : on prend les trades dans l'ordre, max MAX_CONCURRENT en parallele
    balance = BALANCE_START
    print()
    print("=" * 80)
    print(f"  TRADES PAR JOUR (compound, balance start {BALANCE_START}€)")
    print("=" * 80)

    total_trades = 0
    total_wins = 0
    for day in sorted(trades_by_day.keys()):
        day_trades = trades_by_day[day]
        # On limite MAX_CONCURRENT par jour (simplification : pas de gestion fine slots)
        day_trades_capped = day_trades[:MAX_CONCURRENT * 3]  # max 24/jour par jour

        print(f"\n{'='*80}")
        print(f"  {day} - {len(day_trades)} setups (cap {len(day_trades_capped)})")
        print(f"{'='*80}")

        day_pnl = 0.0
        day_wins = 0
        day_loss = 0
        balance_start_day = balance

        for t in day_trades_capped:
            heure = t["ts"].strftime("%H:%M")
            risk_eur = balance * RISK_PCT
            if t["outcome"] == "WIN":
                pnl = risk_eur * t["rr"]
                day_wins += 1
                total_wins += 1
                symbol = "+"
            else:
                pnl = -risk_eur
                day_loss += 1
                symbol = "-"
            balance += pnl
            day_pnl += pnl
            total_trades += 1

            print(f"  {heure} | {t['asset']:7s} | {t['direction'][:4]:4s} | "
                  f"proba={t['proba']:.3f} | {t['outcome']:5s} | "
                  f"{symbol}{abs(pnl):>6.2f}€ | balance={balance:>8.2f}€")

        wr_day = day_wins / max(day_wins + day_loss, 1) * 100
        print(f"\n  >>> Jour : {day_wins + day_loss} trades, WR={wr_day:.1f}%, "
              f"PnL={day_pnl:+.2f}€ ({day_pnl/max(balance_start_day,1)*100:+.1f}%), "
              f"balance={balance:.2f}€")

    # Recap total
    wr_total = total_wins / max(total_trades, 1) * 100
    pnl_total = balance - BALANCE_START
    pnl_pct = (balance / BALANCE_START - 1) * 100
    n_days = len(trades_by_day)

    print()
    print("=" * 80)
    print("  RECAP TOTAL")
    print("=" * 80)
    print(f"  Jours actifs    : {n_days}")
    print(f"  Total trades    : {total_trades}")
    print(f"  WIN / LOSS      : {total_wins} / {total_trades - total_wins}")
    print(f"  WR              : {wr_total:.1f}%")
    print(f"  Trades/jour     : {total_trades / max(n_days, 1):.1f}")
    print(f"  Balance start   : {BALANCE_START:.2f}€")
    print(f"  Balance end     : {balance:.2f}€")
    print(f"  PnL total       : {pnl_total:+.2f}€ ({pnl_pct:+.1f}%)")
    print(f"  PnL/jour moyen  : {pnl_total / max(n_days, 1):+.2f}€/jour")
    print("=" * 80)


if __name__ == "__main__":
    main()
