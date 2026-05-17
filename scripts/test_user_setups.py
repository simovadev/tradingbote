"""Verifie si le nouveau detecteur Sweep Wick OB trouve les setups labellises par l'humain."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from sqlalchemy import select

from bot.detectors.swings import detect_swings
from bot.detectors.sweep_wick_ob import detect_sweep_wick_obs
from data.fetch import load_cached
from db.models import HumanLabeledTrade, SessionLocal, init_db


def main():
    init_db()
    with SessionLocal() as s:
        user_trades = s.execute(select(HumanLabeledTrade)).scalars().all()

    print(f"\n=== Verification : detecteur trouve-t-il les {len(user_trades)} setups user ? ===\n")

    # Groupe par (instrument, jour-de-l-OB-reel) - on utilise le t1 de l'OB zone
    by_day: dict = {}
    for t in user_trades:
        t1 = t.ob_zone.get("t1") if t.ob_zone else None
        if t1:
            real_day = pd.Timestamp(t1).date()
        else:
            real_day = t.day.date()
        key = (t.instrument, real_day)
        by_day.setdefault(key, []).append(t)

    total_user = 0
    total_matched = 0

    for (inst, day), trades in by_day.items():
        print(f"\n--- {inst} {day} ({len(trades)} trades user) ---")
        df = load_cached(inst, "M1")
        # Slice : 1 jour
        start = pd.Timestamp(day).tz_localize("UTC")
        end = start + pd.Timedelta(days=1)
        sub = df.loc[start:end]
        if sub.empty:
            print("  pas de M1")
            continue

        # Detecte
        swings = detect_swings(sub, left=3, right=3)
        detected = detect_sweep_wick_obs(sub, swings)
        print(f"  Detecteur a trouve : {len(detected)} sweep-wick OBs")

        for ut in trades:
            total_user += 1
            ut_dir = ut.direction
            ut_high = ut.ob_zone.get("high") if ut.ob_zone else None
            ut_low = ut.ob_zone.get("low") if ut.ob_zone else None
            ut_t1 = ut.ob_zone.get("t1") if ut.ob_zone else None
            ut_t2 = ut.ob_zone.get("t2") if ut.ob_zone else None

            # Match : meme direction + chevauchement de zone + meme periode (+/- 30 min)
            match = None
            for d in detected:
                if d.direction != ut_dir:
                    continue
                # Chevauchement zone (au moins 30% de chevauchement)
                overlap_high = min(d.zone_high, ut_high or d.zone_high)
                overlap_low = max(d.zone_low, ut_low or d.zone_low)
                if overlap_high <= overlap_low:
                    continue
                # Time proche
                if ut_t1:
                    ut_t1_ts = pd.Timestamp(ut_t1)
                    if ut_t1_ts.tz is None:
                        ut_t1_ts = ut_t1_ts.tz_localize("UTC")
                    delta = abs((d.push_end_time - ut_t1_ts).total_seconds())
                    if delta > 30 * 60:  # 30 min
                        continue
                match = d
                break

            status = "MATCH" if match else "RATE"
            ob_str = f"[{ut_low:.2f}-{ut_high:.2f}]" if ut_high else "?"
            t1_str = ut_t1.split("T")[1][:5] if ut_t1 else "?"
            print(f"  [{status}] #{ut.id} {ut_dir:<8} {t1_str}-{ut_t2.split('T')[1][:5] if ut_t2 else '?'} OB {ob_str}", end="")
            if match:
                total_matched += 1
                print(f"  -> push {match.push_start_time.strftime('%H:%M')}-{match.push_end_time.strftime('%H:%M')} "
                      f"[{match.zone_low:.2f}-{match.zone_high:.2f}]")
            else:
                print()

    print(f"\n=== Resultat : {total_matched}/{total_user} setups user detectes par le bot ===")


if __name__ == "__main__":
    main()
