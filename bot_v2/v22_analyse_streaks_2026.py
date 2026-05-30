"""v22_analyse_streaks_2026.py - Analyse les series de SL/wins de la simu 2026."""
from __future__ import annotations
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
JSON = ROOT / "v22_vast_export" / "v22_simu_2026_results.json"

data = json.loads(JSON.read_text())
eq = data["equity_curve"]

print(f"Total trades joues : {len(eq)}")
print(f"Periode : {eq[0]['ts'][:10]} -> {eq[-1]['ts'][:10]}")
print()

# Classification de chaque trade
# pnl_R = -1.0 -> LOSS sec (SL touche avant TP1)
# pnl_R = +0.5 -> WIN_BE (TP1 touche puis SL@BE, ou TP1 touche + timeout)
# pnl_R = +1.5 -> WIN_FULL
# pnl_R = 0.0 -> FLAT timeout
def classify(p):
    if p <= -0.99: return "SL"
    if p >= 1.49:  return "WIN_FULL"
    if p >= 0.49:  return "WIN_BE"
    if abs(p) < 0.01: return "FLAT"
    return "OTHER"

# Streaks de SL consecutifs
sl_streak_lengths = []
current = 0
for e in eq:
    if classify(e["pnl_R"]) == "SL":
        current += 1
    else:
        if current > 0: sl_streak_lengths.append(current)
        current = 0
if current > 0: sl_streak_lengths.append(current)

# Streaks de wins (FULL ou BE, pas FLAT)
win_streak_lengths = []
current = 0
for e in eq:
    c = classify(e["pnl_R"])
    if c in ("WIN_FULL", "WIN_BE"):
        current += 1
    else:
        if current > 0: win_streak_lengths.append(current)
        current = 0
if current > 0: win_streak_lengths.append(current)

# Distribution SL streaks
print("=" * 70)
print("SERIES DE SL CONSECUTIFS (sur 2466 trades de la simu 2026)")
print("=" * 70)
cnt = Counter(sl_streak_lengths)
total_series = sum(cnt.values())
print(f"Nombre total de series de SL : {total_series}")
print(f"SL max d'affilee             : {max(sl_streak_lengths) if sl_streak_lengths else 0}")
print(f"\n{'Pertes affilee':<18} {'Frequence':<12} {'%':<8} {'Cumul %':<10}")
print("-" * 50)
cum = 0
for k in sorted(cnt.keys()):
    v = cnt[k]
    pct = v / total_series * 100
    cum += pct
    print(f"  {k:<2} pertes        {v:<12} {pct:<8.1f} {cum:<10.1f}")

# Distribution Win streaks
print()
print("=" * 70)
print("SERIES DE WINS CONSECUTIFS")
print("=" * 70)
cnt_w = Counter(win_streak_lengths)
total_w = sum(cnt_w.values())
print(f"Nombre total de series de wins : {total_w}")
print(f"Wins max d'affilee             : {max(win_streak_lengths) if win_streak_lengths else 0}")
print(f"\n{'Wins affilee':<18} {'Frequence':<12} {'%':<8}")
print("-" * 40)
for k in sorted(cnt_w.keys()):
    v = cnt_w[k]
    pct = v / total_w * 100
    print(f"  {k:<2} wins          {v:<12} {pct:<8.1f}")

# Recap par classification
print()
print("=" * 70)
print("REPARTITION DES OUTCOMES")
print("=" * 70)
outcomes = Counter(classify(e["pnl_R"]) for e in eq)
total = sum(outcomes.values())
for k in ["WIN_FULL", "WIN_BE", "FLAT", "SL", "OTHER"]:
    n = outcomes.get(k, 0)
    if n == 0: continue
    print(f"  {k:<10} : {n:>5}  ({n/total*100:.1f}%)")

# Top 10 PIRES streaks (SL d'affilee >= 4) avec date + actifs
print()
print("=" * 70)
print("PIRES SERIES DE SL (>=4 d'affilee) - dates + actifs + impact capital")
print("=" * 70)
streaks_details = []
current_streak = []
for e in eq:
    if classify(e["pnl_R"]) == "SL":
        current_streak.append(e)
    else:
        if len(current_streak) >= 4:
            streaks_details.append(list(current_streak))
        current_streak = []
if len(current_streak) >= 4:
    streaks_details.append(list(current_streak))

streaks_details.sort(key=lambda s: -len(s))
for i, s in enumerate(streaks_details[:15], 1):
    start = s[0]["ts"][:16]; end = s[-1]["ts"][:16]
    cap_before = s[0]["capital"] - s[0]["pnl_euros"]
    cap_after = s[-1]["capital"]
    pct = (cap_after - cap_before) / cap_before * 100 if cap_before else 0
    assets = ",".join(set(e["asset"] for e in s))
    print(f"  #{i:<3} {len(s)} SL  {start} -> {end}  cap {cap_before:>12.0f}E -> {cap_after:>12.0f}E ({pct:+.1f}%) | {assets}")

# Calcul de risk-of-ruin theorique avec WR mesure
import math
wins = outcomes.get("WIN_FULL", 0) + outcomes.get("WIN_BE", 0)
losses = outcomes.get("SL", 0)
wr = wins / (wins + losses) * 100 if (wins+losses) else 0
print()
print(f"=== WR effectif (WIN vs SL strict, hors FLAT) : {wr:.1f}% ===")
print(f"=== Probabilite theorique d'X SL consecutifs avec ce WR ===")
p_loss = (1 - wr/100)
for k in range(3, 11):
    p = p_loss ** k
    expected_on_2466 = total_series * p
    print(f"  {k} SL : proba {p*100:.3f}%   -> attendu sur 2466 trades : {expected_on_2466:.1f}")
