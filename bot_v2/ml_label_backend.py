"""Backend pour les pages de labeling manuel.

- scan_period() : retourne tous les OB Vizion bruts sur [start, end] + niveaux
- scan_with_ml_prediction() : idem + proba v4 (highlight bleu)
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd

from bot_v2.config import SMT_PAIRS, get_param
from bot_v2.concepts.breaker import detect_breakers
from bot_v2.concepts.daily_bias import build_d1_from_h1
from bot_v2.concepts.fvg import detect_fvg
from bot_v2.concepts.htf_swings import collect_htf_swings
from bot_v2.concepts.liquidity import find_swings
from bot_v2.concepts.order_block import detect_order_blocks
from bot_v2.concepts.structure import detect_structure_breaks, detect_trend
from bot_v2.data_loader import load
from bot_v2.pipeline import evaluate_ob


V4_MODEL_PATH = Path("c:/Users/Shadow/TradingBot/bot_v2/ml_model_v4_user_style.pkl")


@dataclass
class LabelTrade:
    """Un OB Vizion brut a labeller (peut etre TRADE ou REJECTED)."""
    ts: str
    direction: str
    score: int
    ml_proba: float
    rr: float
    entry: float
    stop_loss: float
    take_profit: float
    ob_high: float
    ob_low: float
    ob_start_ts: str          # 1ere bougie du groupe
    ob_end_ts: str            # bougie de validation
    killzone: str | None
    outcome: str | None       # "WIN"|"LOSS"|"NO_FILL"|None si pas simule
    pnl_usd: float | None
    v4_proba: float | None    # proba v4 (style user) si modele dispo, sinon None
    # Nouveau : verdict + raison de rejet (pour afficher OB rejetes en gris)
    verdict: str = "TRADE"    # "TRADE" | "REJECTED"
    rejection_bucket: str | None = None
    rejection_reason: str | None = None


def _has_v4_model() -> bool:
    return V4_MODEL_PATH.exists()


def _load_v4_model():
    if not _has_v4_model():
        return None, None
    try:
        with open(V4_MODEL_PATH, "rb") as f:
            payload = pickle.load(f)
        return payload["model"], payload["features"]
    except Exception:
        return None, None


def _v4_predict(model, features: list[str], r, ob) -> float:
    from bot_v2 import ml_filter
    feats = ml_filter._features_from_result(r, ob, "XAUUSD")
    X = pd.DataFrame([[feats.get(f, 0) for f in features]], columns=features)
    return float(model.predict_proba(X)[0, 1])


def _categorize_rejection_short(reason: str | None) -> str:
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
    if "grand-parent" in r:
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


def scan_period(
    instrument: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    include_outcome: bool = True,
    include_rejected: bool = False,
) -> list[LabelTrade]:
    """Scan tous les OB Vizion bruts dans [start, end] et retourne metadata.

    Args:
        include_rejected: si True, inclut aussi les OB rejetes par les filtres,
            avec verdict='REJECTED' + rejection_bucket + rejection_reason.
            Permet a l'UI d'afficher tous les OB du jour, pas juste ceux qui passent.
    """
    df_ltf = load(instrument, "M1")
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

    mask = (df_ltf.index >= start) & (df_ltf.index <= end)
    df_ltf_w = df_ltf[mask]
    if len(df_ltf_w) < 50:
        return []

    correlated_dfs = {}
    for corr_name, corr_type in SMT_PAIRS.get(instrument, []):
        try:
            df_c = load(corr_name, "M1")
            mask_c = (df_c.index >= start) & (df_c.index <= end)
            correlated_dfs[corr_name] = (df_c[mask_c], corr_type)
        except Exception:
            continue

    sws = get_param(instrument, "swing_strength_m1", 2)
    obs = detect_order_blocks(df_ltf_w, swing_strength=sws, max_group_size=2)
    swings_ltf = find_swings(df_ltf_w, strength=sws)

    cache = {
        "swings_ltf": swings_ltf,
        "fvgs_ltf": detect_fvg(df_ltf_w),
        "breakers_ltf": detect_breakers(df_ltf_w),
        "obs_htf": detect_order_blocks(df_htf),
    }
    cache["structure_breaks"] = detect_structure_breaks(df_ltf_w, swings=swings_ltf)
    cache["htf_trend"] = detect_trend(swings_ltf, lookback=6)
    if df_htf2 is not None:
        cache["obs_htf2"] = detect_order_blocks(df_htf2)

    from bot_v2 import ml_filter
    v4_model, v4_features = _load_v4_model()

    out: list[LabelTrade] = []
    for ob in obs:
        try:
            r = evaluate_ob(
                ob, df_ltf_w, df_htf, df_d1, instrument,
                ltf_name="M1", htf_name="M15",
                df_htf2=df_htf2, htf2_name="H1",
                correlated_dfs=correlated_dfs,
                htf_swings=htf_swings,
                df_h1=df_htf2 if df_htf2 is not None else None,
                min_score=0, min_quality=0,
                cache=cache,
            )
        except Exception:
            continue

        # Si rejete : on l'ajoute quand meme si include_rejected, sinon skip
        if r.verdict != "TRADE" or r.trade_setup is None:
            if not include_rejected:
                continue
            # OB rejete : on n'a pas de trade_setup, donc entry/SL/TP par defaut
            from bot_v2.concepts.order_block import ob_entry, ob_stop_loss
            entry_est = ob_entry(ob)
            sl_est = ob_stop_loss(ob, df_ltf_w)
            tp_est = entry_est + (entry_est - sl_est) * 2 if ob.direction == "bullish" \
                     else entry_est - (sl_est - entry_est) * 2
            rr_est = 2.0
            kz_est = r.killzone_name if r.killzone_name else None
            ml_p = 0.0  # pas pertinent pour un OB rejete
            v4_p = None
            bucket = _categorize_rejection_short(r.rejection_reason)
            out.append(LabelTrade(
                ts=ob.validation_ts.isoformat(),
                direction=ob.direction,
                score=int(r.score),
                ml_proba=ml_p, rr=rr_est,
                entry=float(entry_est),
                stop_loss=float(sl_est),
                take_profit=float(tp_est),
                ob_high=float(ob.ob_high), ob_low=float(ob.ob_low),
                ob_start_ts=ob.group_start_ts.isoformat(),
                ob_end_ts=ob.validation_ts.isoformat(),
                killzone=kz_est,
                outcome=None, pnl_usd=None, v4_proba=v4_p,
                verdict="REJECTED",
                rejection_bucket=bucket,
                rejection_reason=r.rejection_reason,
            ))
            continue

        ml_p = ml_filter.predict_proba(r, ob, instrument)
        v4_p = _v4_predict(v4_model, v4_features, r, ob) if v4_model is not None else None

        outcome, pnl = None, None
        if include_outcome:
            try:
                from bot_v2.backtest import simulate_trade
                from bot_v2.trade_setup import compute_position_size, TradeSetup
                lots, risk_usd = compute_position_size(
                    r.trade_setup.entry_price, r.trade_setup.stop_loss,
                    instrument, balance=60.0, risk_pct=0.10,
                )
                if lots > 0:
                    sim_setup = TradeSetup(
                        instrument=instrument, direction=r.trade_setup.direction,
                        entry_price=r.trade_setup.entry_price,
                        stop_loss=r.trade_setup.stop_loss,
                        take_profit=r.trade_setup.take_profit,
                        rr=r.trade_setup.rr,
                        risk_points=r.trade_setup.risk_points,
                        reward_points=r.trade_setup.reward_points,
                        risk_usd=risk_usd, reward_usd=risk_usd * r.trade_setup.rr,
                        position_size_lots=lots,
                        ob_validation_ts=r.trade_setup.ob_validation_ts,
                        tp_source=r.trade_setup.tp_source,
                    )
                    tr = simulate_trade(sim_setup, df_ltf_w, ob.validation_index + 1)
                    outcome = tr.outcome
                    pnl = round(float(tr.pnl_usd), 2)
            except Exception:
                pass

        out.append(LabelTrade(
            ts=ob.validation_ts.isoformat(),
            direction=ob.direction,
            score=int(r.score),
            ml_proba=round(ml_p, 3),
            rr=float(r.trade_setup.rr),
            entry=float(r.trade_setup.entry_price),
            stop_loss=float(r.trade_setup.stop_loss),
            take_profit=float(r.trade_setup.take_profit),
            ob_high=float(ob.ob_high),
            ob_low=float(ob.ob_low),
            ob_start_ts=ob.group_start_ts.isoformat(),
            ob_end_ts=ob.validation_ts.isoformat(),
            killzone=r.killzone_name,
            outcome=outcome,
            pnl_usd=pnl,
            v4_proba=round(v4_p, 3) if v4_p is not None else None,
        ))

    return out


def get_candles(instrument: str, start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
    """Retourne les bougies M1 dans la fenetre pour affichage TradingView."""
    df = load(instrument, "M1")
    mask = (df.index >= start) & (df.index <= end)
    df = df[mask]
    return [
        {"time": int(t.timestamp()), "open": float(r["open"]), "high": float(r["high"]),
         "low": float(r["low"]), "close": float(r["close"])}
        for t, r in df.iterrows()
    ]


def diagnose_period(
    instrument: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict:
    """Analyse les rejets sur la periode : combien d'OB tombent a chaque filtre.

    Permet de voir QUEL filtre est le plus restrictif et faut alleger.
    """
    df_ltf = load(instrument, "M1")
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

    mask = (df_ltf.index >= start) & (df_ltf.index <= end)
    df_ltf_w = df_ltf[mask]
    if len(df_ltf_w) < 50:
        return {"error": "Pas assez de bougies"}

    correlated_dfs = {}
    for corr_name, corr_type in SMT_PAIRS.get(instrument, []):
        try:
            df_c = load(corr_name, "M1")
            mask_c = (df_c.index >= start) & (df_c.index <= end)
            correlated_dfs[corr_name] = (df_c[mask_c], corr_type)
        except Exception:
            continue

    sws = get_param(instrument, "swing_strength_m1", 2)
    obs = detect_order_blocks(df_ltf_w, swing_strength=sws, max_group_size=2)
    swings_ltf = find_swings(df_ltf_w, strength=sws)

    cache = {
        "swings_ltf": swings_ltf,
        "fvgs_ltf": detect_fvg(df_ltf_w),
        "breakers_ltf": detect_breakers(df_ltf_w),
        "obs_htf": detect_order_blocks(df_htf),
    }
    cache["structure_breaks"] = detect_structure_breaks(df_ltf_w, swings=swings_ltf)
    cache["htf_trend"] = detect_trend(swings_ltf, lookback=6)
    if df_htf2 is not None:
        cache["obs_htf2"] = detect_order_blocks(df_htf2)

    rejection_counts: dict[str, int] = {}
    rejection_examples: dict[str, list[str]] = {}
    trade_ok = 0

    for ob in obs:
        try:
            r = evaluate_ob(
                ob, df_ltf_w, df_htf, df_d1, instrument,
                ltf_name="M1", htf_name="M15",
                df_htf2=df_htf2, htf2_name="H1",
                correlated_dfs=correlated_dfs,
                htf_swings=htf_swings,
                df_h1=df_htf2,
                min_score=0, min_quality=0,
                cache=cache,
            )
        except Exception:
            continue
        if r.verdict == "TRADE" and r.trade_setup is not None:
            trade_ok += 1
            continue
        # Classifie le rejet en bucket simple
        reason = r.rejection_reason or "unknown"
        bucket = reason.split("(")[0].split("—")[0].strip()
        # Simplifie en categories
        if "Daily bias" in bucket:
            bucket = "1_daily_bias_contraire"
        elif "Session sans" in bucket:
            bucket = "2_session_play"
        elif "Phase" in bucket:
            bucket = "3_phase_acc_manip"
        elif "displacement" in bucket.lower():
            bucket = "4_displacement_faible"
        elif "Pas de contexte OB M15" in bucket or "parent" in bucket.lower():
            bucket = "5_pas_parent_ob_M15"
        elif "grand-parent" in bucket or "H1 grand" in bucket:
            bucket = "5b_pas_grandparent_H1"
        elif "zone" in bucket.lower() and ("premium" in bucket.lower() or "discount" in bucket.lower() or "neutre" in bucket.lower()):
            bucket = "6_discount_premium"
        elif "Range Fibo" in bucket:
            bucket = "6b_range_fibo_nul"
        elif "RR insuffisant" in bucket or "setup invalide" in bucket:
            bucket = "7_rr_insuffisant"
        elif "Score" in bucket:
            bucket = "8_score_min"
        elif "Quality" in bucket:
            bucket = "9_quality_min"
        elif "NAS100 sans SMT" in bucket:
            bucket = "10_nas100_sans_smt"
        rejection_counts[bucket] = rejection_counts.get(bucket, 0) + 1
        rejection_examples.setdefault(bucket, []).append(
            f"{ob.validation_ts.isoformat()[:16]} {ob.direction[:4]}"
        )

    return {
        "n_obs_total": len(obs),
        "n_trade_ok": trade_ok,
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "rejection_examples": {k: v[:3] for k, v in rejection_examples.items()},
    }


def get_random_day(instrument: str = "XAUUSD") -> tuple[pd.Timestamp, pd.Timestamp]:
    """Pioche 1 jour aleatoire dans la fenetre OOS (post 2025-06)."""
    import random
    df = load(instrument, "M1")
    OOS_START = pd.Timestamp("2025-06-11", tz="UTC")
    end_data = df.index[-1]
    days_avail = (end_data - OOS_START).days - 1
    offset = random.randint(0, days_avail)
    day_start = (OOS_START + pd.Timedelta(days=offset)).normalize()
    day_end = day_start + pd.Timedelta(days=1)
    return day_start, day_end
