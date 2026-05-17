"""Test 10 setups avec analyse complete : contexte HTF + bougies du retournement."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import requests

from data.fetch import load_cached

BASE = "http://localhost:8000"


def analyze(setup: dict) -> dict:
    """Analyse complete d'un setup."""
    inst = setup["instrument"]
    direction = setup["direction"]
    entry_time = pd.Timestamp(setup["entry_time"])
    entry = setup["entry_price"]
    sl = setup["stop_loss"]
    tp = setup["take_profit"]
    rr = setup["risk_reward"]
    ob_high = setup["ob_zone"]["high"]
    ob_low = setup["ob_zone"]["low"]
    ob_t1 = pd.Timestamp(setup["ob_zone"]["from_time"])
    ob_t2 = pd.Timestamp(setup["ob_zone"]["to_time"])

    df_m1 = load_cached(inst, "M1")
    df_m15 = load_cached(inst, "M15")

    # Bougies du push OB (ce que le bot a marque comme OB)
    ob_candles = setup.get("ob_candles", [])

    # Trend M15 calcule avec la MEME methode que le bot
    from bot.detectors.trend import analyze_trend
    m15_before = df_m15.loc[df_m15.index < entry_time].tail(80)
    if len(m15_before) >= 20:
        trend = analyze_trend(m15_before, left=2, right=2)
        m15_direction = trend.direction.upper()[:4]
        m15_net = trend.confidence   # On stocke la confidence pour info
    else:
        m15_direction = "?"
        m15_net = 0

    # Range Asia (du jour cible)
    day_start = entry_time.normalize()
    asia_data = df_m1.loc[
        (df_m1.index >= day_start)
        & (df_m1.index < day_start + pd.Timedelta(hours=4))
    ]
    asia_h = float(asia_data["high"].max()) if not asia_data.empty else None
    asia_l = float(asia_data["low"].min()) if not asia_data.empty else None

    issues = []

    # Verification 1 : OB bougies coherentes
    if not ob_candles:
        issues.append("Pas de ob_candles")
    else:
        expected_color = "vert" if direction == "bearish" else "rouge"
        colors_ok = True
        for c in ob_candles:
            is_bull = c["close"] > c["open"]
            if direction == "bearish" and not is_bull:
                colors_ok = False
            elif direction == "bullish" and is_bull:
                colors_ok = False
        if not colors_ok:
            issues.append(f"OB devrait etre {expected_color} mais contient bougies opposees")

    # Verification 2 : trend M15 vs direction trade (meme analyzer que le bot)
    if m15_direction == "BULL" and direction == "bearish" and m15_net >= 0.7:
        issues.append(f"Trade SHORT contre tendance M15 BULL (conf={m15_net:.1f})")
    elif m15_direction == "BEAR" and direction == "bullish" and m15_net >= 0.7:
        issues.append(f"Trade LONG contre tendance M15 BEAR (conf={m15_net:.1f})")

    # Verification 3 : TP plus loin que entry vers la direction
    if direction == "bullish" and tp <= entry:
        issues.append(f"BUG: TP<=entry sur LONG")
    if direction == "bearish" and tp >= entry:
        issues.append(f"BUG: TP>=entry sur SHORT")

    # Verification 4 : SL coherent
    if direction == "bullish" and sl >= entry:
        issues.append(f"BUG: SL>=entry sur LONG")
    if direction == "bearish" and sl <= entry:
        issues.append(f"BUG: SL<=entry sur SHORT")

    return {
        "instrument": inst,
        "day": setup["day"],
        "direction": direction,
        "entry_time": entry_time.strftime("%H:%M"),
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "rr": rr,
        "tp_source": setup.get("tp_source", "?"),
        "smt": setup["smt"]["confirmed"],
        "fvg_m15": setup["fvg_m15"],
        "outcome": setup["outcome"],
        "ob_n_candles": len(ob_candles),
        "m15_direction": m15_direction,
        "m15_net": m15_net,
        "asia_high": asia_h,
        "asia_low": asia_l,
        "issues": issues,
        "ok": len(issues) == 0,
    }


def main():
    print(f"\n{'='*120}")
    print(f"TEST 10 SETUPS ICT")
    print(f"{'='*120}\n")

    results = []
    for i in range(10):
        try:
            r = requests.get(f"{BASE}/api/validate/next", timeout=30)
            if r.status_code != 200:
                print(f"  [{i+1:02d}] HTTP {r.status_code}: {r.text[:200]}")
                continue
            setup = r.json()
            a = analyze(setup)
            results.append(a)

            status = "OK " if a["ok"] else "BUG"
            smt_str = "SMT" if a["smt"] else "---"
            fvg_str = "FVG" if a["fvg_m15"] else "---"

            print(f"  [{i+1:02d}] [{status}] {a['instrument']:<6} {a['day']} {a['direction']:<8} @{a['entry_time']}  "
                  f"E={a['entry']:<10} TP={a['tp_source']:<25} RR={a['rr']:<5.2f}  "
                  f"OB={a['ob_n_candles']}c  M15={a['m15_direction']}  {smt_str} {fvg_str}  -> {a['outcome']}")

            for iss in a["issues"]:
                print(f"      ! {iss}")
        except Exception as e:
            print(f"  [{i+1:02d}] ERROR: {e}")

    # Stats
    print(f"\n{'='*120}")
    n = len(results)
    ok = sum(1 for r in results if r["ok"])
    wins = sum(1 for r in results if r["outcome"] == "win")
    losses = sum(1 for r in results if r["outcome"] == "loss")
    smt_count = sum(1 for r in results if r["smt"])
    fvg_count = sum(1 for r in results if r["fvg_m15"])
    avg_rr = sum(r["rr"] for r in results) / n if n else 0

    print(f"Total : {n} setups")
    print(f"  Structurellement OK : {ok}/{n}")
    print(f"  Outcomes : W={wins} L={losses} unknown={n-wins-losses}")
    if wins + losses > 0:
        print(f"  Winrate : {wins/(wins+losses)*100:.0f}%")
    print(f"  RR moyen : {avg_rr:.2f}")
    print(f"  Confluence SMT : {smt_count}/{n}")
    print(f"  Confluence FVG M15 : {fvg_count}/{n}")

    # Distribution TP sources
    from collections import Counter
    tp_sources = Counter(r["tp_source"].split(" ")[0] + " " + r["tp_source"].split(" ")[1] if len(r["tp_source"].split()) > 1 else r["tp_source"] for r in results)
    print(f"\nSources TP utilisees :")
    for src, count in tp_sources.most_common():
        print(f"  {src} : {count}")

    # Distribution actifs
    inst_count = Counter(r["instrument"] for r in results)
    print(f"\nActifs :")
    for inst, count in inst_count.most_common():
        wins_inst = sum(1 for r in results if r["instrument"] == inst and r["outcome"] == "win")
        losses_inst = sum(1 for r in results if r["instrument"] == inst and r["outcome"] == "loss")
        print(f"  {inst} : {count} setups (W={wins_inst} L={losses_inst})")


if __name__ == "__main__":
    main()
