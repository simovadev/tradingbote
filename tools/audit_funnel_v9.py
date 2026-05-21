"""Audit de l'entonnoir de filtres V9.

Sur 1 mois de donnees recentes (M1 Vantage), pour 4 actifs representatifs :
1. Compte les OB bruts detectes.
2. Passe chacun dans evaluate_ob et enregistre le verdict + la cause de rejet.
3. Pour les TRADE, calcule la proba et compte ceux qui passent le seuil.

Sortie : un tableau "entonnoir" qui montre OU on perd du volume.

But : identifier les filtres trop stricts pour la V10.

Usage : python tools/audit_funnel_v9.py
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

os.environ["BUILD_DATA_DIR"] = "data_vantage"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from bot_v2.config import SMT_PAIRS, get_param
from bot_v2.data_loader import load
from bot_v2.concepts.htf_swings import collect_htf_swings
from bot_v2.concepts.liquidity import find_swings
from bot_v2.concepts.fvg import detect_fvg
from bot_v2.concepts.breaker import detect_breakers
from bot_v2.concepts.order_block import detect_order_blocks
from bot_v2.concepts.structure import detect_structure_breaks, detect_trend
from bot_v2.concepts.mss_setup import detect_mss_setups
from bot_v2.concepts.daily_bias import build_d1_from_h1
from bot_v2.pipeline import evaluate_ob
from bot_v2 import ml_filter
import joblib, json

# Actifs representatifs : 1 metal, 1 indice US, 1 forex major, 1 forex
ASSETS = ["XAUUSD", "NAS100", "EURUSD", "USDCAD"]

# Fenetre : 1 mois recent
WINDOW_DAYS = 30


def classify_rejection(reason: str) -> str:
    """Regroupe les raisons de rejet en categories."""
    if not reason:
        return "?"
    r = reason.lower()
    if "overnight" in r:
        return "forex_overnight_US"
    if "accumulation" in r:
        return "phase_accumulation"
    if "fibo" in r:
        return "range_fibo_KO"
    if "quality" in r:
        return "quality_min"
    if "rr" in r or "setup invalide" in r:
        return "RR_setup_invalide"
    if "score" in r:
        return "score_min"
    return "autre"


def audit_asset(asset: str):
    df_m1_full = load(asset, "M1")
    end = df_m1_full.index[-1]
    start = end - pd.Timedelta(days=WINDOW_DAYS)
    # Buffer 30j avant pour les swings HTF
    load_start = start - pd.Timedelta(days=30)

    df_m1 = load(asset, "M1", start=load_start, end=end)
    df_m15 = load(asset, "M15", start=load_start, end=end)
    df_h1 = load(asset, "H1", start=load_start, end=end)
    df_h4 = load(asset, "H4", start=load_start, end=end)
    try:
        df_d1 = load(asset, "D1", start=load_start, end=end)
        if len(df_d1) < 10:
            raise FileNotFoundError
    except Exception:
        df_d1 = build_d1_from_h1(df_h1)

    htf_swings = collect_htf_swings({"H1": df_h1, "H4": df_h4, "D1": df_d1}, swing_strength=3)

    correlated_dfs = {}
    for cn, ct in SMT_PAIRS.get(asset, []):
        try:
            dc = load(cn, "M1", start=load_start, end=end)
            correlated_dfs[cn] = (dc, ct)
        except Exception:
            pass

    sws = get_param(asset, "swing_strength_m1", 2)
    obs = detect_order_blocks(df_m1, swing_strength=sws)
    # OB dans la fenetre d'analyse (apres le buffer)
    obs_window = [ob for ob in obs if df_m1.index[ob.validation_index] >= start]

    cache = {
        "swings_ltf": find_swings(df_m1, strength=sws),
        "fvgs_ltf": detect_fvg(df_m1),
        "breakers_ltf": detect_breakers(df_m1),
        "obs_htf": detect_order_blocks(df_m15),
    }
    cache["structure_breaks"] = detect_structure_breaks(
        df_m1, swings=cache["swings_ltf"], fvgs=cache["fvgs_ltf"])
    cache["htf_trend"] = detect_trend(cache["swings_ltf"], lookback=6)
    cache["obs_htf2"] = detect_order_blocks(df_h1)
    mss_setups = detect_mss_setups(
        df_m1, structure_breaks=cache["structure_breaks"],
        swings=cache["swings_ltf"], fvgs=cache["fvgs_ltf"])

    model = joblib.load(ROOT / f"bot_v2/ml_model_{asset}_vantage_v9.pkl")
    features = json.load(open(ROOT / f"bot_v2/ml_features_{asset}_vantage_v9.json"))["features"]
    thr = ml_filter.ML_THRESHOLDS.get(asset, 0.65)

    rej_counter = Counter()
    n_trade = 0
    n_passed_thr = 0
    proba_below = 0
    for ob in obs_window:
        try:
            r = evaluate_ob(
                ob, df_m1, df_m15, df_d1, asset,
                ltf_name="M1", htf_name="M15", df_htf2=df_h1, htf2_name="H1",
                correlated_dfs=correlated_dfs, htf_swings=htf_swings,
                df_h1=df_h1, min_score=0, min_quality=0, cache=cache,
            )
        except Exception:
            rej_counter["exception"] += 1
            continue
        if r.verdict != "TRADE":
            rej_counter[classify_rejection(r.rejection_reason or "")] += 1
            continue
        n_trade += 1
        # Proba ML
        feat = ml_filter._features_from_result(r, ob, asset, df_ltf=df_m1, df_d1=df_d1, mss_setups=mss_setups)
        X = pd.DataFrame([[feat.get(f, 0) for f in features]], columns=features)
        proba = float(model.predict_proba(X)[0, 1])
        if proba >= thr:
            n_passed_thr += 1
        else:
            proba_below += 1

    return {
        "asset": asset,
        "n_ob": len(obs_window),
        "rejections": dict(rej_counter),
        "n_trade_verdict": n_trade,
        "n_proba_below": proba_below,
        "n_passed": n_passed_thr,
        "threshold": thr,
        "window_days": WINDOW_DAYS,
    }


def main():
    print("=" * 80)
    print(f"AUDIT ENTONNOIR V9 - {WINDOW_DAYS} derniers jours")
    print("=" * 80)
    print()

    totals = Counter()
    n_ob_tot = n_trade_tot = n_below_tot = n_pass_tot = 0

    for asset in ASSETS:
        try:
            res = audit_asset(asset)
        except Exception as e:
            print(f"  {asset} : ERREUR {e}")
            import traceback; traceback.print_exc()
            continue

        n_ob = res["n_ob"]
        print(f"### {asset}  (seuil {res['threshold']:.2f})")
        print(f"  OB bruts detectes (fenetre)   : {n_ob}")
        print(f"  --- REJETS pipeline ---")
        rej_total = 0
        for cause, n in sorted(res["rejections"].items(), key=lambda x: -x[1]):
            pct = 100 * n / n_ob if n_ob else 0
            print(f"    {cause:24s} : {n:>5} ({pct:.0f}%)")
            totals[cause] += n
            rej_total += n
        print(f"  --- SURVIVENT (verdict TRADE) : {res['n_trade_verdict']} ({100*res['n_trade_verdict']/n_ob:.0f}%)")
        print(f"      proba < seuil  : {res['n_proba_below']}")
        print(f"      proba >= seuil : {res['n_passed']}  <<< TRADES PRIS")
        tpd = res["n_passed"] / res["window_days"]
        print(f"      -> {tpd:.2f} trades/jour sur {asset}")
        print()

        n_ob_tot += n_ob
        n_trade_tot += res["n_trade_verdict"]
        n_below_tot += res["n_proba_below"]
        n_pass_tot += res["n_passed"]

    print("=" * 80)
    print("ENTONNOIR GLOBAL (4 actifs)")
    print("=" * 80)
    print(f"OB bruts                     : {n_ob_tot}")
    print(f"Rejets pipeline (par cause)  :")
    for cause, n in sorted(totals.items(), key=lambda x: -x[1]):
        print(f"  {cause:24s} : {n:>5} ({100*n/n_ob_tot:.0f}% des OB)")
    print(f"Verdict TRADE                : {n_trade_tot} ({100*n_trade_tot/n_ob_tot:.0f}%)")
    print(f"  rejetes par ML (proba<seuil): {n_below_tot}")
    print(f"  PASSENT (trades pris)       : {n_pass_tot} ({100*n_pass_tot/n_ob_tot:.1f}% des OB bruts)")
    print()
    print(f">>> Sur {WINDOW_DAYS}j, 4 actifs : {n_pass_tot} trades = {n_pass_tot/WINDOW_DAYS:.1f}/jour")
    print(f">>> Extrapole 14 actifs : ~{n_pass_tot/WINDOW_DAYS*14/4:.1f} trades/jour")


if __name__ == "__main__":
    main()
