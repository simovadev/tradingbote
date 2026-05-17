"""Analyse les OB dessines par le user pour comprendre son style.

Pour chaque zone dessinee :
1. Identifie l'OB Vizion le plus proche dans la zone (si existe)
2. Lance evaluate_ob complet -> capture rejection_reason
3. Categorise dans un "bucket" de rejet

Sortie : rapport avec
- N zones tracees
- Combien matchent un OB Vizion (deja detecte)
- Pour les OB Vizion correspondants, quels filtres bloquent
- Patterns communs : session, direction, taille, etc.
- Config proposee pour capter la majorite des setups user
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, asdict
from typing import Literal

import pandas as pd

from bot_v2.config import SMT_PAIRS, get_param
from bot_v2.concepts.breaker import detect_breakers
from bot_v2.concepts.daily_bias import build_d1_from_h1
from bot_v2.concepts.fvg import detect_fvg
from bot_v2.concepts.htf_swings import collect_htf_swings
from bot_v2.concepts.killzones import killzone_at
from bot_v2.concepts.liquidity import find_swings
from bot_v2.concepts.order_block import detect_order_blocks
from bot_v2.concepts.structure import detect_structure_breaks, detect_trend
from bot_v2.data_loader import load
from bot_v2.manual_labels import load_drawn_obs
from bot_v2.pipeline import evaluate_ob


@dataclass
class DrawnAnalysisRow:
    """Resultat d'analyse pour 1 zone dessinee."""
    start_ts: str
    end_ts: str
    direction: str
    ob_high: float
    ob_low: float
    # Match avec Vizion
    matched_vizion_ob: bool          # True si un OB Vizion existe dans cette zone
    matched_ob_ts: str | None        # validation_ts de l'OB Vizion matche
    matched_ob_score: int | None
    # Pourquoi rejete ?
    pipeline_verdict: str | None     # "TRADE" | "REJECTED"
    rejection_bucket: str | None     # categorie de rejet simplifiee
    rejection_reason: str | None     # reason brut
    # Contexte
    killzone: str | None
    has_displacement: bool | None
    daily_bias: str | None           # "bullish" | "bearish" | "neutral"


def _categorize_rejection(reason: str | None) -> str:
    if not reason:
        return "unknown"
    r = reason.lower()
    if "daily bias" in r and "contraire" in r:
        return "daily_bias_contraire"
    if "session sans" in r:
        return "session_play"
    if "phase accumulation" in r or "phase manipulation" in r:
        return "phase_acc_manip"
    if "displacement" in r:
        return "displacement_faible"
    if "parent" in r or "contexte ob m15" in r:
        return "pas_parent_ob_M15"
    if "grand-parent" in r or "h1 grand" in r:
        return "pas_grandparent_H1"
    if "zone equilibrium" in r or "zone premium" in r or "zone discount" in r:
        return "discount_premium"
    if "fvg sync" in r:
        return "pas_fvg_sync"
    if "rr insuffisant" in r or "setup invalide" in r:
        return "rr_insuffisant"
    if "smt" in r and "nas100" in r:
        return "nas100_sans_smt"
    if "score" in r:
        return "score_min"
    if "quality" in r:
        return "quality_min"
    if "range fibo" in r:
        return "range_fibo_nul"
    if "killzone" in r:
        return "killzone"
    return "autre"


def _find_matching_ob(
    obs: list,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    direction: str,
    ob_high: float,
    ob_low: float,
):
    """Trouve l'OB Vizion le plus proche d'une zone dessinee.

    Critere :
    - direction identique
    - validation_ts dans [start_ts - 30min, end_ts + 30min]
    - zone qui chevauche (au moins partiellement) la zone dessinee
    """
    window_start = start_ts - pd.Timedelta(minutes=30)
    window_end = end_ts + pd.Timedelta(minutes=30)

    candidates = [
        ob for ob in obs
        if ob.direction == direction
        and window_start <= ob.validation_ts <= window_end
        and not (ob.ob_high < ob_low or ob.ob_low > ob_high)  # overlap zones
    ]
    if not candidates:
        return None
    # Le plus proche en temps du milieu de la zone dessinee
    mid = start_ts + (end_ts - start_ts) / 2
    return min(candidates, key=lambda ob: abs((ob.validation_ts - mid).total_seconds()))


def analyze_drawn_obs(instrument: str = "XAUUSD") -> dict:
    """Analyse complete des OB dessines par le user."""
    drawn = load_drawn_obs()
    if not drawn:
        return {"error": "Aucun OB dessine. Va sur /label_month, mode 'Dessiner mon OB'."}

    # Determine la plage temporelle couverte
    all_ts = []
    for d in drawn:
        all_ts.append(pd.Timestamp(d.start_ts))
        all_ts.append(pd.Timestamp(d.end_ts))
    period_start = min(all_ts) - pd.Timedelta(hours=2)
    period_end = max(all_ts) + pd.Timedelta(hours=2)

    # Pre-charge le contexte LTF + HTF + caches pour evaluate_ob
    df_ltf_full = load(instrument, "M1")
    df_htf = load(instrument, "M15")
    try:
        df_htf2 = load(instrument, "H1")
    except Exception:
        df_htf2 = None
    try:
        df_d1 = load(instrument, "D1")
        if len(df_d1) < 10:
            raise FileNotFoundError
    except Exception:
        df_h1 = load(instrument, "H1")
        df_d1 = build_d1_from_h1(df_h1)

    htf_dfs_swings = {}
    for tf in ["H1", "H4", "D1"]:
        try:
            htf_dfs_swings[tf] = load(instrument, tf)
        except Exception:
            pass
    htf_swings = collect_htf_swings(htf_dfs_swings, swing_strength=3)

    mask = (df_ltf_full.index >= period_start) & (df_ltf_full.index <= period_end)
    df_ltf = df_ltf_full[mask]

    correlated_dfs = {}
    for corr_name, corr_type in SMT_PAIRS.get(instrument, []):
        try:
            df_c = load(corr_name, "M1")
            mask_c = (df_c.index >= period_start) & (df_c.index <= period_end)
            correlated_dfs[corr_name] = (df_c[mask_c], corr_type)
        except Exception:
            continue

    sws = get_param(instrument, "swing_strength_m1", 2)
    obs = detect_order_blocks(df_ltf, swing_strength=sws, max_group_size=2)
    swings_ltf = find_swings(df_ltf, strength=sws)
    cache = {
        "swings_ltf": swings_ltf,
        "fvgs_ltf": detect_fvg(df_ltf),
        "breakers_ltf": detect_breakers(df_ltf),
        "obs_htf": detect_order_blocks(df_htf),
    }
    cache["structure_breaks"] = detect_structure_breaks(df_ltf, swings=swings_ltf)
    cache["htf_trend"] = detect_trend(swings_ltf, lookback=6)
    if df_htf2 is not None:
        cache["obs_htf2"] = detect_order_blocks(df_htf2)

    rows: list[DrawnAnalysisRow] = []
    bucket_counter: Counter = Counter()
    session_counter: Counter = Counter()
    direction_counter: Counter = Counter()
    matched_count = 0
    accepted_count = 0

    for d in drawn:
        start = pd.Timestamp(d.start_ts)
        end = pd.Timestamp(d.end_ts)
        match = _find_matching_ob(obs, start, end, d.direction, d.ob_high, d.ob_low)

        # Killzone basee sur le centre de la zone
        kz_check_ts = start + (end - start) / 2
        kz = killzone_at(kz_check_ts)
        session_counter[kz or "hors_KZ"] += 1
        direction_counter[d.direction] += 1

        if match is None:
            rows.append(DrawnAnalysisRow(
                start_ts=d.start_ts, end_ts=d.end_ts, direction=d.direction,
                ob_high=d.ob_high, ob_low=d.ob_low,
                matched_vizion_ob=False, matched_ob_ts=None, matched_ob_score=None,
                pipeline_verdict=None,
                rejection_bucket="vizion_n_a_pas_detecte_OB",
                rejection_reason="Aucun OB Vizion ne matche la zone dessinee",
                killzone=kz, has_displacement=None, daily_bias=None,
            ))
            bucket_counter["vizion_n_a_pas_detecte_OB"] += 1
            continue

        matched_count += 1
        # Evalue l'OB matche
        try:
            r = evaluate_ob(
                match, df_ltf, df_htf, df_d1, instrument,
                ltf_name="M1", htf_name="M15",
                df_htf2=df_htf2, htf2_name="H1",
                correlated_dfs=correlated_dfs,
                htf_swings=htf_swings,
                df_h1=df_htf2,
                min_score=0, min_quality=0,
                cache=cache,
            )
            bucket = "TRADE_OK" if r.verdict == "TRADE" else _categorize_rejection(r.rejection_reason)
            if r.verdict == "TRADE":
                accepted_count += 1
            bucket_counter[bucket] += 1
            rows.append(DrawnAnalysisRow(
                start_ts=d.start_ts, end_ts=d.end_ts, direction=d.direction,
                ob_high=d.ob_high, ob_low=d.ob_low,
                matched_vizion_ob=True,
                matched_ob_ts=match.validation_ts.isoformat(),
                matched_ob_score=int(r.score),
                pipeline_verdict=r.verdict,
                rejection_bucket=bucket,
                rejection_reason=r.rejection_reason,
                killzone=r.killzone_name,
                has_displacement=None,
                daily_bias=r.daily_bias.bias if r.daily_bias else None,
            ))
        except Exception as e:
            bucket_counter["pipeline_error"] += 1
            rows.append(DrawnAnalysisRow(
                start_ts=d.start_ts, end_ts=d.end_ts, direction=d.direction,
                ob_high=d.ob_high, ob_low=d.ob_low,
                matched_vizion_ob=True, matched_ob_ts=match.validation_ts.isoformat(),
                matched_ob_score=None,
                pipeline_verdict=None, rejection_bucket="pipeline_error",
                rejection_reason=str(e)[:200],
                killzone=kz, has_displacement=None, daily_bias=None,
            ))

    # Genere config proposee selon les rejets les plus frequents
    suggestions = _suggest_config(bucket_counter, len(drawn))

    return {
        "n_drawn": len(drawn),
        "n_matched_vizion": matched_count,
        "n_currently_accepted": accepted_count,
        "n_not_detected_by_vizion": len(drawn) - matched_count,
        "rejection_buckets": dict(bucket_counter.most_common()),
        "session_distribution": dict(session_counter.most_common()),
        "direction_distribution": dict(direction_counter.most_common()),
        "rows": [asdict(r) for r in rows],
        "suggestions": suggestions,
    }


def _suggest_config(bucket_counter: Counter, total: int) -> list[dict]:
    """Genere des suggestions de modification config selon les rejets."""
    suggestions = []

    def pct(n):
        return round(n / total * 100, 1) if total > 0 else 0

    if bucket_counter.get("vizion_n_a_pas_detecte_OB", 0) > 0:
        n = bucket_counter["vizion_n_a_pas_detecte_OB"]
        suggestions.append({
            "priority": "high" if n / total > 0.2 else "medium",
            "issue": f"{n} zones dessinees ({pct(n)}%) ne sont pas detectees par Vizion du tout",
            "action": "Assouplir swing_strength_m1 (2->1) et/ou max_group_size (2->5) dans config.py",
            "param": "swing_strength_m1",
            "current": 2,
            "suggested": 1,
        })

    if bucket_counter.get("daily_bias_contraire", 0) > 0:
        n = bucket_counter["daily_bias_contraire"]
        suggestions.append({
            "priority": "high" if n / total > 0.2 else "medium",
            "issue": f"{n} setups ({pct(n)}%) rejetes par daily_bias contraire",
            "action": "Transformer le filtre en penalite de score (-15 pts) au lieu de rejet eliminatoire",
            "param": "daily_bias_filter",
            "current": "REJECT",
            "suggested": "PENALTY",
        })

    if bucket_counter.get("displacement_faible", 0) > 0:
        n = bucket_counter["displacement_faible"]
        suggestions.append({
            "priority": "medium" if n / total > 0.15 else "low",
            "issue": f"{n} setups ({pct(n)}%) rejetes par displacement faible",
            "action": "Baisser min_displacement_atr (0.8 -> 0.5)",
            "param": "min_displacement_atr",
            "current": 0.8,
            "suggested": 0.5,
        })

    if bucket_counter.get("pas_parent_ob_M15", 0) > 0:
        n = bucket_counter["pas_parent_ob_M15"]
        suggestions.append({
            "priority": "medium" if n / total > 0.15 else "low",
            "issue": f"{n} setups ({pct(n)}%) rejetes par absence d'OB parent M15",
            "action": "Elargir parent_ob_tolerance_pct (0.003 -> 0.006) ou desactiver filtre HTF",
            "param": "parent_ob_tolerance_pct",
            "current": 0.003,
            "suggested": 0.006,
        })

    if bucket_counter.get("discount_premium", 0) > 0:
        n = bucket_counter["discount_premium"]
        suggestions.append({
            "priority": "low",
            "issue": f"{n} setups ({pct(n)}%) en zone equilibrium",
            "action": "Elargir la zone D/P (0.35/0.65 -> 0.45/0.55)",
            "param": "discount_premium_threshold",
            "current": "0.35/0.65",
            "suggested": "0.45/0.55",
        })

    if bucket_counter.get("pas_fvg_sync", 0) > 0:
        n = bucket_counter["pas_fvg_sync"]
        suggestions.append({
            "priority": "low",
            "issue": f"{n} setups ({pct(n)}%) sans FVG sync",
            "action": "Rendre FVG sync optionnel (deja bonus, devrait pas rejeter)",
            "param": "fvg_sync_required",
            "current": True,
            "suggested": False,
        })

    return suggestions


if __name__ == "__main__":
    import json
    result = analyze_drawn_obs()
    print(json.dumps(result, indent=2, default=str))
