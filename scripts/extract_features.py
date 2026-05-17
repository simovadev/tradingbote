"""Extrait les FEATURES qui distinguent les OB user des OB bruit.

Pour chaque OB user trouvee, mesure :
- push_strength (taille du push en pts)
- push_strength_atr (push / ATR)
- push_n_candles (nombre de bougies du push)
- killzone (Asia/London/NY/HORS)
- distance liquidite sweepee
- volume ratio
- day of week

Compare la distribution TES OB vs LES AUTRES OB du meme jour.
"""
from __future__ import annotations

import statistics as stats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from sqlalchemy import select

from bot.detectors.swings import detect_swings
from bot.detectors.sweep_wick_ob import detect_sweep_wick_obs
from bot.killzones import current_killzone
from data.fetch import load_cached
from db.models import HumanLabeledTrade, SessionLocal, init_db


def featurize(ob, df):
    """Extrait les features mesurables d'un OB."""
    kz = current_killzone(ob.push_end_time)
    avg_price = df["close"].iloc[ob.push_end_index - 50:ob.push_end_index].mean() if ob.push_end_index >= 50 else df["close"].mean()
    # ATR a ce moment
    if ob.push_end_index >= 14:
        recent = df.iloc[ob.push_end_index - 14:ob.push_end_index]
        atr = (recent["high"] - recent["low"]).mean()
    else:
        atr = 1.0
    push_atr = ob.push_strength / atr if atr > 0 else 0

    # Volume relatif du push (somme volumes / moyenne 20 bougies precedentes)
    push_vol = 0.0
    avg_vol = 0.0
    if "volume" in df.columns:
        push_vol = float(df.iloc[ob.push_start_index:ob.push_end_index + 1]["volume"].sum())
        prev_start = max(0, ob.push_start_index - 20)
        avg_vol = float(df.iloc[prev_start:ob.push_start_index]["volume"].mean() or 0)
    vol_ratio = (push_vol / (ob.push_end_index - ob.push_start_index + 1)) / avg_vol if avg_vol > 0 else 0

    # Distance du sweep
    sweep_dist = 0.0
    if ob.swept_swing is not None:
        sweep_dist = abs(ob.swept_level - (ob.zone_high if ob.direction == "bearish" else ob.zone_low))

    return {
        "push_strength": ob.push_strength,
        "push_atr": push_atr,
        "push_n_candles": ob.push_end_index - ob.push_start_index + 1,
        "killzone": kz.name if kz else "none",
        "killzone_label": kz.label if kz else "HORS",
        "sweep_dist": sweep_dist,
        "vol_ratio": vol_ratio,
        "day_of_week": ob.push_end_time.weekday(),
        "hour_utc": ob.push_end_time.hour,
        "zone_size": ob.zone_size,
        "zone_size_pct": (ob.zone_size / avg_price * 100) if avg_price > 0 else 0,
    }


def is_user_match(d, ut) -> bool:
    if d.direction != ut.direction:
        return False
    ut_high = ut.ob_zone.get("high") if ut.ob_zone else None
    ut_low = ut.ob_zone.get("low") if ut.ob_zone else None
    if ut_high is None:
        return False
    overlap_high = min(d.zone_high, ut_high)
    overlap_low = max(d.zone_low, ut_low)
    if overlap_high <= overlap_low:
        return False
    ut_t1 = ut.ob_zone.get("t1")
    if ut_t1:
        ut_t1_ts = pd.Timestamp(ut_t1)
        if ut_t1_ts.tz is None:
            ut_t1_ts = ut_t1_ts.tz_localize("UTC")
        delta = abs((d.push_end_time - ut_t1_ts).total_seconds())
        if delta > 60 * 60:
            return False
    return True


def main():
    init_db()
    with SessionLocal() as s:
        user_trades = s.execute(select(HumanLabeledTrade)).scalars().all()

    by_day: dict = {}
    for t in user_trades:
        t1 = t.ob_zone.get("t1") if t.ob_zone else None
        real_day = pd.Timestamp(t1).date() if t1 else t.day.date()
        by_day.setdefault((t.instrument, real_day), []).append(t)

    user_features: list[dict] = []
    noise_features: list[dict] = []

    for (inst, day), uts in by_day.items():
        df = load_cached(inst, "M1")
        start = pd.Timestamp(day).tz_localize("UTC")
        end = start + pd.Timedelta(days=1)
        sub = df.loc[start:end]
        if sub.empty:
            continue
        swings = detect_swings(sub, left=3, right=3)
        detected = detect_sweep_wick_obs(sub, swings)

        for d in detected:
            matched = any(is_user_match(d, ut) for ut in uts)
            feat = featurize(d, sub)
            feat["instrument"] = inst
            feat["day"] = str(day)
            (user_features if matched else noise_features).append(feat)

    print(f"\n{'='*80}")
    print(f"USER OBs : {len(user_features)}")
    print(f"NOISE OBs : {len(noise_features)}")
    print(f"{'='*80}\n")

    keys = ["push_strength", "push_atr", "push_n_candles", "sweep_dist",
            "vol_ratio", "zone_size_pct", "hour_utc"]
    print(f"{'Feature':<20} | {'USER median':<12} | {'NOISE median':<12} | {'USER min':<10} | {'USER max':<10}")
    print("-" * 80)
    for k in keys:
        u_vals = [f[k] for f in user_features if f[k] is not None]
        n_vals = [f[k] for f in noise_features if f[k] is not None]
        if not u_vals or not n_vals:
            continue
        u_med = stats.median(u_vals)
        n_med = stats.median(n_vals)
        u_min = min(u_vals)
        u_max = max(u_vals)
        print(f"{k:<20} | {u_med:>10.2f}   | {n_med:>10.2f}   | {u_min:>8.2f}   | {u_max:>8.2f}")

    # Killzone distribution
    print(f"\n{'Killzone':<20} | USER  | NOISE")
    print("-" * 50)
    from collections import Counter
    u_kz = Counter(f["killzone_label"] for f in user_features)
    n_kz = Counter(f["killzone_label"] for f in noise_features)
    for kz in set(list(u_kz) + list(n_kz)):
        u = u_kz.get(kz, 0)
        n = n_kz.get(kz, 0)
        u_pct = u / len(user_features) * 100 if user_features else 0
        n_pct = n / len(noise_features) * 100 if noise_features else 0
        print(f"{kz:<20} | {u} ({u_pct:.0f}%)  | {n} ({n_pct:.0f}%)")


if __name__ == "__main__":
    main()
