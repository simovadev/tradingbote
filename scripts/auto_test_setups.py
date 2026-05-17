"""Auto-test : recupere 20 setups + analyse leur qualite avant de livrer."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import requests

from data.fetch import load_cached

BASE = "http://localhost:8000"


def analyze_setup(setup: dict) -> dict:
    """Verifie qualite d'un setup."""
    inst = setup["instrument"]
    df = load_cached(inst, "M1")

    entry_time = pd.Timestamp(setup["entry_time"])
    ob_to_time = pd.Timestamp(setup["ob_zone"]["to_time"])
    direction = setup["direction"]
    entry = setup["entry_price"]
    sl = setup["stop_loss"]
    tp = setup["take_profit"]
    ob_high = setup["ob_zone"]["high"]
    ob_low = setup["ob_zone"]["low"]

    issues = []

    # 1. entry doit etre APRES ob_to_time (le push)
    if entry_time <= ob_to_time:
        issues.append(f"BUG: entry={entry_time} <= ob_to={ob_to_time}")
    else:
        delay_min = (entry_time - ob_to_time).total_seconds() / 60
        if delay_min < 1:
            issues.append(f"Entry trop rapide ({delay_min:.0f}min apres OB)")

    # 2. entry doit etre DANS la zone OB
    ob_mid = (ob_high + ob_low) / 2
    if not (ob_low <= entry <= ob_high):
        issues.append(f"BUG: entry {entry} HORS zone OB [{ob_low}-{ob_high}]")

    # 3. Pour SHORT : SL > entry > TP. Pour LONG : SL < entry < TP
    if direction == "bullish":
        if not (sl < entry < tp):
            issues.append(f"BUG LONG: sl={sl} < entry={entry} < tp={tp} viole")
    else:
        if not (sl > entry > tp):
            issues.append(f"BUG SHORT: sl={sl} > entry={entry} > tp={tp} viole")

    # 4. RR raisonnable
    rr = setup["risk_reward"]
    if rr > 10:
        issues.append(f"RR trop eleve ({rr})")
    if rr < 1:
        issues.append(f"RR trop faible ({rr})")

    # 5. Taille zone OB raisonnable
    zone_size = ob_high - ob_low
    avg_price = df.loc[df.index <= entry_time].tail(50)["close"].mean()
    zone_pct = zone_size / avg_price * 100
    if zone_pct > 1.0:
        issues.append(f"Zone OB trop large ({zone_pct:.2f}% du prix)")

    return {
        "instrument": inst,
        "day": setup["day"],
        "direction": direction,
        "entry_time": entry_time.strftime("%H:%M"),
        "ob_time": ob_to_time.strftime("%H:%M"),
        "delay_min": (entry_time - ob_to_time).total_seconds() / 60,
        "rr": rr,
        "zone_size_pct": zone_pct,
        "outcome": setup["outcome"],
        "issues": issues,
        "ok": len(issues) == 0,
    }


def main():
    print(f"\n=== AUTO-TEST : 20 setups ===\n")
    results = []
    for i in range(20):
        try:
            r = requests.get(f"{BASE}/api/validate/next", timeout=30)
            if r.status_code != 200:
                print(f"  [{i+1}] HTTP {r.status_code}")
                continue
            setup = r.json()
            analysis = analyze_setup(setup)
            results.append(analysis)
            status = "OK" if analysis["ok"] else "BUG"
            issues_str = " | ".join(analysis["issues"]) if analysis["issues"] else ""
            print(f"  [{i+1:02d}] [{status}] {analysis['instrument']:<6} {analysis['day']} "
                  f"{analysis['direction']:<8} entry={analysis['entry_time']} (ob={analysis['ob_time']}, "
                  f"+{analysis['delay_min']:.0f}min) RR={analysis['rr']:.1f} "
                  f"outcome={analysis['outcome']}  {issues_str}")
        except Exception as e:
            print(f"  [{i+1}] ERROR: {e}")

    # Resume
    n_ok = sum(1 for r in results if r["ok"])
    n_total = len(results)
    print(f"\n=== Resume : {n_ok}/{n_total} setups OK ({n_ok/n_total*100:.0f}%) ===")

    # Outcomes
    wins = sum(1 for r in results if r["outcome"] == "win")
    losses = sum(1 for r in results if r["outcome"] == "loss")
    print(f"Outcomes : W={wins} L={losses} winrate={wins/(wins+losses)*100 if wins+losses else 0:.0f}%")

    # Issues frequentes
    all_issues = [issue for r in results for issue in r["issues"]]
    from collections import Counter
    if all_issues:
        print(f"\nIssues frequentes :")
        for issue, count in Counter(all_issues).most_common(5):
            print(f"  {count}x : {issue}")


if __name__ == "__main__":
    main()
