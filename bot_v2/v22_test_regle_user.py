"""v22_test_regle_user.py - Test de LA regle ICT du user :

Acheter (LONG) UNIQUEMENT si :
- D1 bias est haussier
- H1 trend est haussier
- Prix est en DISCOUNT daily (sous le mid du range PDH-PDL)

Vendre (SHORT) UNIQUEMENT si :
- D1 bias est baissier
- H1 trend est baissier
- Prix est en PREMIUM daily (au-dessus du mid)

Compare au baseline (tous les OBs).
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

# Charge le CSV deja genere par l'analyseur
CSV = ROOT / "v22_features_XAUUSD.csv"


def main():
    print("=" * 80)
    print("TEST REGLE ICT USER : D1 + H1 alignes + Discount/Premium daily")
    print("=" * 80)

    df = pd.read_csv(CSV)
    print(f"\nDataset : {len(df)} OBs")
    print(f"WR baseline : {df['label'].mean()*100:.1f}%")

    # Reconstruit la regle
    # d1_aligned = D1 bias va dans le sens du trade
    # h1_aligned = H1 trend va dans le sens du trade
    # Discount/Premium : pour bullish il faut etre en discount (pct < 0.5)
    #                    pour bearish il faut etre en premium (pct > 0.5)

    df["pd_aligned"] = (
        ((df["direction_bullish"] == 1) & (df["price_in_range_pct"] < 0.5)) |
        ((df["direction_bullish"] == 0) & (df["price_in_range_pct"] > 0.5))
    ).astype(int)

    df["regle_user_complete"] = (
        (df["d1_aligned"] == 1) &
        (df["h1_aligned"] == 1) &
        (df["pd_aligned"] == 1)
    ).astype(int)

    # ============ STATS DETAIL ============
    print("\n=== FILTRE PAR FILTRE ===\n")
    for col, label in [
        ("d1_aligned", "D1 aligne uniquement"),
        ("h1_aligned", "H1 aligne uniquement"),
        ("pd_aligned", "Discount/Premium aligne uniquement"),
    ]:
        sel = df[col] == 1
        wr = df[sel]["label"].mean() * 100 if sel.sum() > 0 else 0
        print(f"  {label:<40} n={sel.sum():<5}  WR={wr:.1f}%")

    # ============ COMBINAISONS ============
    print("\n=== COMBINAISONS ===\n")
    combinations = [
        ("D1 + H1", ["d1_aligned", "h1_aligned"]),
        ("D1 + PD daily", ["d1_aligned", "pd_aligned"]),
        ("H1 + PD daily", ["h1_aligned", "pd_aligned"]),
        ("D1 + H1 + PD daily (REGLE COMPLETE USER)", ["d1_aligned", "h1_aligned", "pd_aligned"]),
        ("D1 + H1 + ATR vol high", ["d1_aligned", "h1_aligned", "atr_high_vol"]),
        ("D1 + H1 + PD + NOT lundi", ["d1_aligned", "h1_aligned", "pd_aligned"]),
        ("D1 + H1 + PD + ATR vol", ["d1_aligned", "h1_aligned", "pd_aligned", "atr_high_vol"]),
    ]

    print(f"{'config':<55} {'n':<6} {'WR':<8} {'gain_vs_BL':<11} {'exp_R(RR2)'}")
    print("-" * 100)
    BL = df["label"].mean() * 100
    for label, cols in combinations:
        mask = pd.Series([True] * len(df))
        for c in cols:
            mask &= (df[c] == 1)
        # Filtre NOT lundi si dans label
        if "NOT lundi" in label:
            mask &= (df["is_monday"] == 0)
        n = mask.sum()
        if n < 10:
            print(f"{label:<55} n={n:<6} (trop peu)")
            continue
        wins = df[mask]["label"].sum()
        wr = wins / n * 100
        gain = wr - BL
        # exp R avec RR 2 : WIN = +2R, LOSS = -1R
        losses = n - wins
        exp_r = (wins * 2 + losses * (-1)) / n
        marker = " ***EDGE***" if exp_r > 0 else ""
        print(f"{label:<55} n={n:<6} WR={wr:<7.1f}% {gain:+<10.1f}% exp_R={exp_r:+.3f}R{marker}")

    # ============ DECOMPOSITION : ta regle complete par sens ============
    print("\n=== DECOMPOSITION REGLE COMPLETE PAR SENS ===\n")
    # Long uniquement
    long_mask = (df["direction_bullish"] == 1) & (df["d1_aligned"] == 1) & (df["h1_aligned"] == 1) & (df["pd_aligned"] == 1)
    short_mask = (df["direction_bullish"] == 0) & (df["d1_aligned"] == 1) & (df["h1_aligned"] == 1) & (df["pd_aligned"] == 1)
    print(f"LONG (D1+H1 haussier, prix en discount)   n={long_mask.sum():<4} WR={df[long_mask]['label'].mean()*100 if long_mask.sum() else 0:.1f}%")
    print(f"SHORT (D1+H1 baissier, prix en premium)   n={short_mask.sum():<4} WR={df[short_mask]['label'].mean()*100 if short_mask.sum() else 0:.1f}%")

    # ============ FREQUENCE / EVOLUTION TEMPORELLE ============
    print("\n=== EVOLUTION DANS LE TEMPS (par 6 mois) ===\n")
    df["dt"] = pd.to_datetime(df["ts"], unit="s")
    df["period"] = df["dt"].dt.to_period("Q")
    user_mask = df["regle_user_complete"] == 1
    print(f"{'periode':<10} {'n_user':<8} {'WR_user':<10} {'n_total':<10} {'WR_total'}")
    for period in sorted(df["period"].unique()):
        sub = df[df["period"] == period]
        sub_user = sub[sub["regle_user_complete"] == 1]
        n_total = len(sub); n_user = len(sub_user)
        wr_total = sub["label"].mean() * 100 if n_total else 0
        wr_user = sub_user["label"].mean() * 100 if n_user else 0
        print(f"{str(period):<10} {n_user:<8} {wr_user:<10.1f} {n_total:<10} {wr_total:.1f}")

    # ============ AVEC AUTRES PLANS TRADE ============
    print("\n=== TA REGLE AVEC DIFFERENTS RR ===")
    print("(meme entry/SL mais TP plus eleve)")
    # On a deja le label pour RR=2. On simule RR3 et RR5 mentalement :
    # Comme on n'a pas les valeurs exactes du SL/TP, on ne peut pas refaire la simu ici.
    # On peut juste constater : avec WR fixe X, BE_WR(RR) = 100/(RR+1)
    user_n = user_mask.sum()
    user_wins = df[user_mask]["label"].sum()
    user_wr = user_wins / user_n * 100 if user_n else 0
    print(f"Ta regle complete : n={user_n}, WR={user_wr:.1f}% (sur trades RR2)")
    for rr in [2.0, 3.0, 5.0]:
        be = 100 / (rr + 1)
        verdict = "PROFITABLE" if user_wr > be else "perdant"
        margin = user_wr - be
        # esperance si WR garde meme valeur (approximation)
        exp = user_wr/100 * rr + (1 - user_wr/100) * (-1)
        print(f"  RR={rr}  BE_WR={be:.1f}%  ecart={margin:+.1f}%  exp_R={exp:+.3f}R  -> {verdict}")


if __name__ == "__main__":
    main()
