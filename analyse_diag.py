"""Analyse le CSV diagnostic ultra-logge pour trouver les patterns de LOSS.

Pour chaque dimension, on compare WIN vs LOSS pour identifier ce qui distingue
les bons trades des mauvais. Si une dimension separe nettement, on a un filtre
potentiel a appliquer.

Usage :
    python analyse_diag.py bt_diag_trades.csv
"""
from __future__ import annotations
import sys
import pandas as pd
import numpy as np

CSV = sys.argv[1] if len(sys.argv) > 1 else "bt_diag_trades.csv"
df = pd.read_csv(CSV)
print(f"=== {CSV} ===  {len(df)} setups total\n")

# Quel outcome ?
print("=== OUTCOMES ===")
for o in ["WIN", "LOSS", "NO_FILL", "INVALID_PRICE", "OPEN"]:
    n = (df.outcome == o).sum()
    pct = n / len(df) * 100
    print(f"  {o:14}: {n:4d}  ({pct:.0f}%)")
fermes = df[df.outcome.isin(["WIN", "LOSS"])]
if len(fermes) == 0:
    print("Aucun trade ferme — analyse impossible"); sys.exit(0)

wr = (fermes.outcome == "WIN").mean() * 100
pnl = fermes.pnl_r.sum()
print(f"\n  Trades fermes : {len(fermes)}  WR={wr:.1f}%  PnL={pnl:+.2f}R\n")

# === 1. Par actif ===
print("=== PAR ACTIF ===")
by_inst = fermes.groupby("instrument").agg(
    n=("outcome", "count"),
    win=("outcome", lambda x: (x == "WIN").sum()),
    pnl=("pnl_r", "sum"),
).reset_index()
by_inst["wr"] = by_inst["win"] / by_inst["n"] * 100
by_inst = by_inst.sort_values("pnl", ascending=False)
print(by_inst.to_string(index=False))

# === 2. Par heure UTC ===
print("\n=== PAR HEURE UTC ===")
by_h = fermes.groupby("hour_utc").agg(
    n=("outcome", "count"),
    win=("outcome", lambda x: (x == "WIN").sum()),
    pnl=("pnl_r", "sum"),
).reset_index()
by_h["wr"] = by_h["win"] / by_h["n"] * 100
print(by_h[by_h["n"] >= 1].to_string(index=False))

# === 3. Par killzone ===
print("\n=== PAR KILLZONE ===")
by_kz = fermes.groupby("killzone").agg(
    n=("outcome", "count"),
    win=("outcome", lambda x: (x == "WIN").sum()),
    pnl=("pnl_r", "sum"),
).reset_index()
by_kz["wr"] = by_kz["win"] / by_kz["n"] * 100
print(by_kz.to_string(index=False))

# === 4. Par direction ===
print("\n=== PAR DIRECTION ===")
by_d = fermes.groupby("direction").agg(
    n=("outcome", "count"),
    win=("outcome", lambda x: (x == "WIN").sum()),
    pnl=("pnl_r", "sum"),
).reset_index()
by_d["wr"] = by_d["win"] / by_d["n"] * 100
print(by_d.to_string(index=False))

# === 5. Par contexte H1/H4 trend ===
if "ctx_h1_trend" in fermes.columns:
    print("\n=== PAR TREND H1 ===")
    by_t = fermes.groupby("ctx_h1_trend").agg(
        n=("outcome", "count"),
        win=("outcome", lambda x: (x == "WIN").sum()),
        pnl=("pnl_r", "sum"),
    ).reset_index()
    by_t["wr"] = by_t["win"] / by_t["n"] * 100
    print(by_t.to_string(index=False))

    print("\n=== PAR TREND H4 ===")
    by_t = fermes.groupby("ctx_h4_trend").agg(
        n=("outcome", "count"),
        win=("outcome", lambda x: (x == "WIN").sum()),
        pnl=("pnl_r", "sum"),
    ).reset_index()
    by_t["wr"] = by_t["win"] / by_t["n"] * 100
    print(by_t.to_string(index=False))

    # Trade dans le sens H1 (concordant) vs contre H1
    print("\n=== TRADE VS TREND H1 ===")
    def concordant(row):
        if row.get("ctx_h1_trend") in (None, "range") or pd.isna(row.get("ctx_h1_trend")):
            return "range"
        return "WITH" if row["direction"].startswith(row["ctx_h1_trend"][:4]) else "AGAINST"
    fermes2 = fermes.copy()
    fermes2["vs_h1"] = fermes2.apply(concordant, axis=1)
    by_vs = fermes2.groupby("vs_h1").agg(
        n=("outcome", "count"),
        win=("outcome", lambda x: (x == "WIN").sum()),
        pnl=("pnl_r", "sum"),
    ).reset_index()
    by_vs["wr"] = by_vs["win"] / by_vs["n"] * 100
    print(by_vs.to_string(index=False))

# === 6. MFE / MAE — les LOSS auraient-ils pu etre des WIN si on prend des profits plus tot ? ===
if "mfe_R" in fermes.columns:
    print("\n=== MFE / MAE PAR OUTCOME ===")
    print("(MFE = combien le trade etait en + avant de mourir; MAE = max perte avant SL/TP)")
    for o in ["WIN", "LOSS"]:
        sub = fermes[fermes.outcome == o]
        if len(sub) == 0:
            continue
        print(f"  {o} (n={len(sub)}):")
        print(f"    MFE : median={sub.mfe_R.median():.2f}R  mean={sub.mfe_R.mean():.2f}R  max={sub.mfe_R.max():.2f}R")
        print(f"    MAE : median={sub.mae_R.median():.2f}R  mean={sub.mae_R.mean():.2f}R  max={sub.mae_R.max():.2f}R")

    # LOSS qui ont eu un MFE >= 0.5R = on aurait pu sortir en +
    losses = fermes[fermes.outcome == "LOSS"]
    if len(losses) > 0:
        could_be_saved = losses[losses.mfe_R >= 0.5]
        print(f"\n  LOSS avec MFE >= 0.5R : {len(could_be_saved)}/{len(losses)} ({len(could_be_saved)/len(losses)*100:.0f}%)")
        could_be_saved_1R = losses[losses.mfe_R >= 1.0]
        print(f"  LOSS avec MFE >= 1.0R : {len(could_be_saved_1R)}/{len(losses)} ({len(could_be_saved_1R)/len(losses)*100:.0f}%)")
        if len(could_be_saved_1R) > 0:
            print(f"  -> Sortir a +1R BE aurait sauve {len(could_be_saved_1R)} trades")

# === 7. Hold time (combien de temps les LOSS sont longs ?) ===
if "hold_time_s" in fermes.columns:
    print("\n=== HOLD TIME PAR OUTCOME ===")
    for o in ["WIN", "LOSS"]:
        sub = fermes[fermes.outcome == o]
        if len(sub) == 0: continue
        ht = sub.hold_time_s.dropna()
        print(f"  {o}: median={ht.median():.0f}s  mean={ht.mean():.0f}s  max={ht.max():.0f}s")

# === 8. ML proba — les LOSS ont-elles une proba differente ? ===
print("\n=== ML PROBA PAR OUTCOME ===")
print("(Si LOSS ont une proba significativement < WIN, le ML calibre mal)")
for o in ["WIN", "LOSS", "NO_FILL", "INVALID_PRICE"]:
    sub = df[df.outcome == o]
    if len(sub) == 0: continue
    print(f"  {o:14}: median={sub.ml_proba.median():.3f}  mean={sub.ml_proba.mean():.3f}")

# === 9. Age OB au placement ===
if "age_at_placement_min" in df.columns:
    print("\n=== AGE OB AU PLACEMENT (min) ===")
    for o in ["WIN", "LOSS", "NO_FILL", "INVALID_PRICE"]:
        sub = df[df.outcome == o]
        if len(sub) == 0: continue
        print(f"  {o:14}: median={sub.age_at_placement_min.median():.1f}  mean={sub.age_at_placement_min.mean():.1f}")

# === 10. Taille SL en pips/% ===
if "risk_abs" in fermes.columns:
    print("\n=== TAILLE SL (en % du prix entry) ===")
    fermes2 = fermes.copy()
    fermes2["sl_pct"] = fermes2.risk_abs / fermes2.entry * 100
    for o in ["WIN", "LOSS"]:
        sub = fermes2[fermes2.outcome == o]
        if len(sub) == 0: continue
        print(f"  {o}: SL median = {sub.sl_pct.median():.4f}%  mean={sub.sl_pct.mean():.4f}%")

# === 11. Difference prix au placement vs entry (en %) ===
if "entry_vs_ref_price_pct" in df.columns:
    inv = df[df.outcome == "INVALID_PRICE"]
    if len(inv) > 0:
        print("\n=== ECART entry vs prix marche pour les INVALID_PRICE ===")
        print(f"  median = {inv.entry_vs_ref_price_pct.median():.4f}%")
        print(f"  mean   = {inv.entry_vs_ref_price_pct.mean():.4f}%")
        print(f"  max    = {inv.entry_vs_ref_price_pct.max():.4f}%")
        print(f"  -> En moyenne le prix a depasse l'entry de {inv.entry_vs_ref_price_pct.mean():.3f}%")

print("\n=== END ===")
