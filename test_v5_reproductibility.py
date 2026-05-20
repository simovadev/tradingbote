"""Test V5 reproductibilite : meme OB doit donner meme proba ML.

Prend les setups de la simulation 24h (BTCUSD 19:42 ML=0.76, etc.) et les rejoue
avec EXACTEMENT le meme code que le scan live, pour confirmer que :
- meme code = meme proba = pas de bug
- OU difference = bug a identifier

Usage : python test_v5_reproductibility.py
"""
import os
import sys
import pandas as pd

# Auto-detect root (le dossier du script)
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from bot_v2.mt5_executor import MT5Executor
from bot_v2.config import get_param, SMT_PAIRS
from bot_v2.concepts.order_block import detect_order_blocks
from bot_v2.concepts.liquidity import find_swings
from bot_v2.concepts.structure import detect_structure_breaks, detect_trend
from bot_v2.concepts.fvg import detect_fvg
from bot_v2.concepts.breaker import detect_breakers
from bot_v2.concepts.mss_setup import detect_mss_setups
from bot_v2.concepts.htf_swings import collect_htf_swings
from bot_v2.concepts.daily_bias import build_d1_from_h1
from bot_v2.pipeline import evaluate_ob
from bot_v2 import ml_filter
import json
import pickle


# Setups identifies par la simulation 24h (live_v5_local.log lignes 111-120)
# Format : (asset, ts_utc, direction, proba_simu_attendue)
SIMU_SETUPS = [
    ("BTCUSD", "2026-05-19 19:42:00", "bearish",  0.760),
    ("DJ30",   "2026-05-19 20:08:00", "bullish",  0.775),
    ("BTCUSD", "2026-05-20 01:49:00", "bullish",  0.858),
    ("DJ30",   "2026-05-20 03:17:00", "bullish",  0.819),
    ("SP500",  "2026-05-20 03:18:00", "bullish",  0.776),
    ("SP500",  "2026-05-20 06:51:00", "bullish",  0.755),
    ("GBPUSD", "2026-05-20 07:04:00", "bullish",  0.771),
    ("USDCHF", "2026-05-20 10:05:00", "bearish",  0.842),
    ("GBPUSD", "2026-05-20 11:11:00", "bullish",  0.774),
    ("EURUSD", "2026-05-20 17:33:00", "bearish",  0.763),
]


def load_model_v5(asset):
    """Charge le modele V5 d'un actif."""
    pkl = os.path.join(ROOT, "bot_v2", f"ml_model_{asset}_admiral_v5.pkl")
    feat = os.path.join(ROOT, "bot_v2", f"ml_features_{asset}_admiral_v5.json")
    with open(pkl, "rb") as f:
        model = pickle.load(f)
    features = json.loads(open(feat).read())["features"]
    return model, features


def test_setup(mt5, asset, ts_target, direction_target, proba_attendue):
    """Rejoue le scan pour cet asset, cherche l'OB validant a ts_target."""
    target_ts = pd.Timestamp(ts_target, tz="UTC")
    print(f"\n{'='*70}")
    print(f"TEST {asset} ts={ts_target} direction={direction_target} (simu={proba_attendue})")
    print(f"{'='*70}")

    # Fetch bougies (meme code que scan_asset)
    df_m1 = mt5.get_bars(asset, "M1", 3000, force_sync=True)
    if df_m1 is None or len(df_m1) < 200:
        print(f"  KO: M1 fetch fail")
        return
    df_m1 = df_m1.iloc[:-1]

    df_m15 = mt5.get_bars(asset, "M15", 500)
    df_h1 = mt5.get_bars(asset, "H1", 500)
    if df_m15 is None or df_h1 is None:
        print(f"  KO: M15/H1 fetch fail")
        return
    df_m15 = df_m15.iloc[:-1]
    df_h1 = df_h1.iloc[:-1]

    try:
        df_d1 = mt5.get_bars(asset, "D1", 100)
        if df_d1 is None or len(df_d1) < 10:
            df_d1 = build_d1_from_h1(df_h1)
    except Exception:
        df_d1 = build_d1_from_h1(df_h1)

    # HTF
    htf_dfs = {"H1": df_h1, "D1": df_d1}
    try:
        df_h4 = mt5.get_bars(asset, "H4", 500)
        if df_h4 is not None and len(df_h4) > 0:
            htf_dfs["H4"] = df_h4
    except Exception:
        pass
    htf_swings = collect_htf_swings(htf_dfs, swing_strength=3)

    # SMT
    correlated_dfs = {}
    for corr_name, corr_type in SMT_PAIRS.get(asset, []):
        try:
            df_c = mt5.get_bars(corr_name, "M1", 3000)
            if df_c is not None and len(df_c) > 0:
                correlated_dfs[corr_name] = (df_c, corr_type)
        except Exception:
            continue

    # Detection OBs + MSS (meme params que scan_asset)
    sws = get_param(asset, "swing_strength_m1", 2)
    obs = detect_order_blocks(df_m1, swing_strength=sws, max_group_size=2)

    cache = {
        "swings_ltf": find_swings(df_m1, strength=sws),
        "fvgs_ltf": detect_fvg(df_m1),
        "breakers_ltf": detect_breakers(df_m1),
        "obs_htf": detect_order_blocks(df_m15),
    }
    cache["structure_breaks"] = detect_structure_breaks(df_m1, swings=cache["swings_ltf"])
    cache["htf_trend"] = detect_trend(cache["swings_ltf"], lookback=6)
    cache["obs_htf2"] = detect_order_blocks(df_h1)

    mss_setups = detect_mss_setups(df_m1, structure_breaks=cache["structure_breaks"], swings=cache["swings_ltf"])

    print(f"  Bougies: M1={len(df_m1)} M15={len(df_m15)} H1={len(df_h1)}")
    print(f"  Detection: {len(obs)} OBs, {len(mss_setups)} MSS")

    # Cherche l'OB qui valide au ts_target (+/- 1 min)
    matches = []
    for ob in obs:
        ob_ts = df_m1.index[ob.validation_index]
        delta = abs((ob_ts - target_ts).total_seconds())
        if delta <= 60 and ob.direction == direction_target:
            matches.append((ob, ob_ts, delta))

    if not matches:
        print(f"  KO: aucun OB {direction_target} trouve a {target_ts} (+/- 1min)")
        print(f"  OBs proches :")
        for ob in obs:
            ob_ts = df_m1.index[ob.validation_index]
            delta = abs((ob_ts - target_ts).total_seconds()) / 60
            if delta < 30:
                print(f"    {ob_ts} | {ob.direction} | delta={delta:.1f}min")
        return

    print(f"  OB(s) trouve(s) : {len(matches)}")
    for ob, ob_ts, delta_s in matches:
        print(f"\n  --- OB {asset} {ob.direction} validation={ob_ts} (delta={delta_s:.0f}s) ---")
        try:
            r = evaluate_ob(
                ob, df_m1, df_m15, df_d1, asset,
                ltf_name="M1", htf_name="M15",
                df_htf2=df_h1, htf2_name="H1",
                correlated_dfs=correlated_dfs,
                htf_swings=htf_swings,
                df_h1=df_h1, min_score=0, min_quality=0,
                cache=cache,
            )
        except Exception as e:
            print(f"    EXCEPTION evaluate_ob: {e}")
            continue

        if r.verdict != "TRADE":
            print(f"    REJECTED: {r.rejection_reason}")
            continue

        # Predict proba (meme code que scan_asset)
        model, features = load_model_v5(asset)
        feats = ml_filter._features_from_result(r, ob, asset, df_ltf=df_m1, df_d1=df_d1, mss_setups=mss_setups)
        import pandas as _pd
        X = _pd.DataFrame([[feats.get(f, 0) for f in features]], columns=features)
        proba = float(model.predict_proba(X)[0, 1])

        ecart = abs(proba - proba_attendue)
        verdict = "OK" if ecart < 0.05 else f"DIFF {ecart:.3f}"
        print(f"    score={r.score} quality={r.quality.total_quality_score if r.quality else 0}")
        print(f"    has_mss_nearby={feats['has_mss_nearby']} vol_ratio_setup={feats['vol_ratio_setup']:.3f}")
        print(f"    phase_reversal={feats['phase_reversal']} phase_manipulation={feats['phase_manipulation']}")
        print(f"    PROBA={proba:.3f}  (attendu simu={proba_attendue:.3f})  >>> {verdict}")


def main():
    mt5 = MT5Executor()
    mt5.initialize()
    print(f"\nMT5 connecte: balance={mt5.get_balance()}")
    print(f"Broker offset: {mt5.broker_utc_offset_sec}s")

    for asset, ts, direction, proba in SIMU_SETUPS:
        test_setup(mt5, asset, ts, direction, proba)


if __name__ == "__main__":
    main()
