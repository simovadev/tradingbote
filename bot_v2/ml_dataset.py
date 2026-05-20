"""Construction du dataset ML : extrait features + outcome pour chaque OB candidat.

Pour CHAQUE OB detecte (acceptes ET rejetes par le pipeline Vizion), on extrait :
- Features Vizion (score, confluences, RR, displacement, killzone, etc.)
- Outcome de la simulation (WIN/LOSS/NO_FILL)

Output : dataset.parquet pour entrainer un LightGBM.

PRINCIPE : on garde TOUTES les regles Vizion eliminatoires (daily bias contraire, etc.)
mais on retire les filtres "qualite" (min_score, min_quality) pour avoir un dataset large.
Le ML apprendra a filtrer la qualite.
"""
from __future__ import annotations

import sys
# Auto-detect Windows vs Linux pour le sys.path
_ROOT = "c:/Users/Shadow/TradingBot" if sys.platform == "win32" else "/workspace/TradingBot"
sys.path.insert(0, _ROOT)

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from bot_v2.backtest import simulate_trade
from bot_v2.config import SMT_PAIRS, get_param, primary_instruments
from bot_v2.concepts.breaker import detect_breakers
from bot_v2.concepts.daily_bias import build_d1_from_h1
from bot_v2.concepts.fvg import detect_fvg
from bot_v2.concepts.htf_swings import collect_htf_swings
from bot_v2.concepts.killzones import killzone_at
from bot_v2.concepts.liquidity import find_swings
from bot_v2.concepts.order_block import detect_order_blocks
from bot_v2.concepts.structure import detect_structure_breaks, detect_trend
from bot_v2.data_loader import load
from bot_v2.pipeline import evaluate_ob
from bot_v2.trade_setup import compute_position_size, TradeSetup


def _compute_atr(df, idx, period=14):
    """ATR True Range moyen sur `period` bougies precedant idx."""
    if idx < period:
        return 0.0
    sl = df.iloc[max(0, idx - period):idx]
    high_low = sl["high"] - sl["low"]
    high_close = (sl["high"] - sl["close"].shift(1)).abs()
    low_close = (sl["low"] - sl["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return float(tr.mean()) if len(tr) > 0 else 0.0


def _get_daily_levels(df_d1, validation_ts):
    """Retourne (pdh, pdl, daily_open, prev_day_open) au moment de validation_ts."""
    if df_d1 is None or len(df_d1) == 0:
        return None, None, None, None
    past = df_d1[df_d1.index < validation_ts]
    if len(past) < 2:
        return None, None, None, None
    yesterday = past.iloc[-1]
    today_open_row = df_d1[df_d1.index >= validation_ts]
    if len(today_open_row) > 0:
        # Si validation est dans la journee D1 courante
        today_session = past.iloc[-1] if past.iloc[-1].name.date() == validation_ts.date() else None
        today_open = today_session["open"] if today_session is not None else yesterday["close"]
    else:
        today_open = yesterday["close"]
    prev_day_open = past.iloc[-2]["open"] if len(past) >= 2 else yesterday["open"]
    return float(yesterday["high"]), float(yesterday["low"]), float(today_open), float(prev_day_open)


def _extract_features(r, ob, instrument, df_ltf=None, df_d1=None, df_htf=None, mss_setups=None):
    """Extrait features ML V3 - cleaned & enrichi.

    V3 (2026-05-19) :
    - Vire 7 features constantes (multi_liq_sweep, kz_none, has_sync_fvg,
      has_grandparent, has_phase_reversal, is_mss_setup, has_mss_confirmation)
    - Ajoute features temps : hour_of_day, day_of_week, minutes_into_killzone
    - Ajoute features volatilite : atr_at_setup, atr_ratio_100
    - Ajoute features distance : dist_to_pdh_pct, dist_to_pdl_pct, dist_to_d1_open_pct
    - Ajoute features structure : bars_since_last_swing

    V5 (2026-05-20) : 4 nouvelles features
    - phase_expansion, phase_reversal, phase_manipulation
    - vol_ratio_setup (ATR14/ATR100)
    - has_mss_nearby (MSS a ±10 bougies de la validation OB)
    """
    setup = r.trade_setup
    conf = r.confluences or []
    conf_text = " ".join(conf)

    f = {
        # Meta
        "instrument": instrument,
        "ts": ob.validation_ts,
        "direction": ob.direction,
        # Scores Vizion
        "score": r.score,
        "quality": r.quality.total_quality_score if r.quality else 0,
        "ob_strength": r.quality.ob_strength if r.quality else 0,
        "sweep_strength": r.quality.sweep_strength if r.quality else 0,
        "retest_count": r.quality.retest_count if r.quality else 0,
        "is_unicorn": int(r.quality.is_unicorn) if r.quality else 0,
        # Trade setup
        "rr": setup.rr if setup else 0,
        "risk_points": setup.risk_points if setup else 0,
        "tp_source_htf": int("htf" in (setup.tp_source if setup else "")),
        "tp_source_capped": int("capped" in (setup.tp_source if setup else "")),
        # Daily bias
        "daily_bias_aligned": int(r.daily_bias_ok is True),
        "daily_bias_neutral": int(r.daily_bias is not None and r.daily_bias.bias == "neutral"),
        # Killzone (one-hot)
        "kz_london": int(r.killzone_name == "London"),
        "kz_ny_am": int(r.killzone_name == "NY_AM"),
        "kz_ny_pm": int(r.killzone_name == "NY_PM"),
        "kz_asia": int(r.killzone_name == "Asia"),
        "kz_ny_lunch": int(r.killzone_name == "NY_Lunch"),
        # Confluences (les non-constantes seulement)
        "has_smt": int("smt_" in conf_text),
        "has_feu_vert": int("feu_vert" in conf_text),
        "has_breaker_kz": int("breaker_in_KZ" in conf_text),
        "has_mss_fvg": int("MSS_with_FVG" in conf_text),
        "has_po3_dist": int("po3_distribution" in conf_text),
        "has_phase_expansion": int("phase_expansion" in conf_text),
        "has_open_midnight_respect": int("respecte_OpenMidnightNY" in conf_text),
        # V3.5 (2026-05-19) : indicateurs des filtres relaches
        # Permet au ML d'apprendre si l'absence de ces conditions est penalisante
        "has_FVG_sync": int("OB_FVG_sync" in conf_text),
        "has_parent_ob": int("parent_ob_" in conf_text and "no_parent_ob_" not in conf_text),
        "has_grandparent_ob": int("grandparent_ob_" in conf_text and "no_grandparent" not in conf_text),
        "has_good_zone": int("mauvaise_zone" not in conf_text and "zone=" in conf_text),
        "has_session_direction": int("session_sans_direction" not in conf_text),
        # OB structure
        "ob_group_size": ob.group_size,
        "bars_sweep_to_validation": ob.validation_index - getattr(ob.sweep, "sweep_index", ob.group_start_index),
        "bars_group_to_validation": ob.validation_index - ob.group_start_index,
        "is_bullish": int(ob.direction == "bullish"),
    }

    # ==================== NEW V3 FEATURES ====================
    ts = ob.validation_ts

    # 1. Features temps
    f["hour_of_day"] = int(ts.hour)
    f["day_of_week"] = int(ts.dayofweek)  # 0=lundi, 4=vendredi
    # minutes_into_killzone (depuis debut de la KZ courante)
    # On utilise hour:minute simple - les KZ commencent toutes a heure ronde
    f["minutes_into_killzone"] = int(ts.hour * 60 + ts.minute) % 60 if r.killzone_name else -1

    # 2. Features volatilite (calculees sur df_ltf au moment de validation)
    if df_ltf is not None and ob.validation_index is not None:
        atr_setup = _compute_atr(df_ltf, ob.validation_index, period=14)
        atr_100 = _compute_atr(df_ltf, ob.validation_index, period=100)
        f["atr_at_setup"] = atr_setup
        f["atr_ratio_100"] = (atr_setup / atr_100) if atr_100 > 0 else 1.0
    else:
        f["atr_at_setup"] = 0.0
        f["atr_ratio_100"] = 1.0

    # 3. Features distance (PDH/PDL/D1 open)
    pdh, pdl, d1_open, _ = _get_daily_levels(df_d1, ts)
    entry_price = setup.entry_price if setup else (ob.ob_low + ob.ob_high) / 2
    if pdh and entry_price:
        f["dist_to_pdh_pct"] = abs(entry_price - pdh) / pdh * 100
    else:
        f["dist_to_pdh_pct"] = 0.0
    if pdl and entry_price:
        f["dist_to_pdl_pct"] = abs(entry_price - pdl) / pdl * 100
    else:
        f["dist_to_pdl_pct"] = 0.0
    if d1_open and entry_price:
        f["dist_to_d1_open_pct"] = (entry_price - d1_open) / d1_open * 100
    else:
        f["dist_to_d1_open_pct"] = 0.0

    # ==================== NEW V5 FEATURES (2026-05-20) ====================
    # Phase de marche (feature ML pour que le ML apprenne)
    # Note : phase_expansion existe deja en V3 (line 118), on garde mais aussi
    # phase_reversal et phase_manipulation (transformes en bonus/malus en V5).
    f["phase_reversal"] = int("phase_reversal" in conf_text)
    f["phase_manipulation"] = int("phase_manipulation" in conf_text)

    # Volatilite ratio courte/longue (a la validation OB)
    if df_ltf is not None and ob.validation_index >= 100:
        h14 = df_ltf.iloc[ob.validation_index-14:ob.validation_index]["high"]
        l14 = df_ltf.iloc[ob.validation_index-14:ob.validation_index]["low"]
        h100 = df_ltf.iloc[ob.validation_index-100:ob.validation_index]["high"]
        l100 = df_ltf.iloc[ob.validation_index-100:ob.validation_index]["low"]
        a14 = float((h14 - l14).mean())
        a100 = float((h100 - l100).mean())
        f["vol_ratio_setup"] = (a14 / a100) if a100 > 0 else 1.0
    else:
        f["vol_ratio_setup"] = 1.0

    # Presence MSS proche (remplace le filtre dur confirm_ob_with_mss)
    _mss_list = mss_setups or []
    f["has_mss_nearby"] = int(any(
        abs(getattr(mss, "mss", mss).break_index - ob.validation_index) <= 10
        for mss in _mss_list
    ))

    return f


# Chaines de TF Vizion par LTF (user 2026-05-16 : multi-TF)
# Note : M30 pas dispo dans data_loader, on utilise H1 a la place pour M5.
TF_CHAINS = {
    "M1":  {"ltf": "M1",  "htf": "M15", "htf2": "H1"},
    "M5":  {"ltf": "M5",  "htf": "H1",  "htf2": "H4"},
    "M15": {"ltf": "M15", "htf": "H1",  "htf2": "H4"},
}


# ============ CACHE WORKER-LEVEL (refacto 2026-05-19) ============
# Chaque worker process garde en cache les data globales (M1/HTF/SMT/swings/OBs HTF)
# pour eviter de les recharger/recalculer a chaque chunk.
# Gain estime : x3-x5 sur la vitesse de build (selon nb chunks).
_WORKER_CACHE = {}


def _get_global_data(inst, ltf_name, htf_name, htf2_name):
    """Charge et cache les data globales (toutes periodes) + indicateurs HTF.

    Appele 1 fois par worker process, reutilise pour tous les chunks du worker.
    """
    cache_key = (inst, ltf_name, htf_name, htf2_name)
    if cache_key in _WORKER_CACHE:
        return _WORKER_CACHE[cache_key]

    # === Load data globales ===
    df_ltf = load(inst, ltf_name)
    df_htf = load(inst, htf_name)
    try:
        df_htf2 = load(inst, htf2_name)
    except Exception:
        df_htf2 = None
    try:
        df_d1 = load(inst, "D1")
        if len(df_d1) < 10:
            raise FileNotFoundError
    except Exception:
        df_h1 = load(inst, "H1")
        df_d1 = build_d1_from_h1(df_h1)

    # === Precompute HTF indicateurs (1 fois pour toutes !) ===
    htf_dfs_swings = {}
    for tf in ["H1", "H4", "D1"]:
        try:
            htf_dfs_swings[tf] = load(inst, tf)
        except Exception:
            pass
    htf_swings = collect_htf_swings(htf_dfs_swings, swing_strength=3)

    # === Precompute OBs HTF (1 fois pour toutes !) ===
    obs_htf = detect_order_blocks(df_htf)
    obs_htf2 = detect_order_blocks(df_htf2) if df_htf2 is not None else None

    # === Load SMT correlated data (XAGUSD, DXY pour XAUUSD) ===
    correlated_dfs_full = {}
    for corr_name, corr_type in SMT_PAIRS.get(inst, []):
        try:
            df_c = load(corr_name, ltf_name)
            if len(df_c) > 0:
                correlated_dfs_full[corr_name] = (df_c, corr_type)
        except Exception:
            continue

    data = {
        "df_ltf": df_ltf,
        "df_htf": df_htf,
        "df_htf2": df_htf2,
        "df_d1": df_d1,
        "htf_swings": htf_swings,
        "obs_htf": obs_htf,
        "obs_htf2": obs_htf2,
        "correlated_dfs_full": correlated_dfs_full,
    }
    _WORKER_CACHE[cache_key] = data
    return data


def _process_instrument(args):
    """Worker process : extrait dataset complet pour un instrument.

    Args tuple : (inst, start_ts, end_ts) ou (inst, start_ts, end_ts, ltf).
    LTF par defaut = M1. Si M5/M15/M30 fournis, on adapte les chaines HTF.

    REFACTO 2026-05-19 : utilise _WORKER_CACHE pour charger data 1 fois par worker.
    """
    if len(args) == 4:
        inst, start_ts, end_ts, ltf = args
    else:
        inst, start_ts, end_ts = args
        ltf = "M1"

    chain = TF_CHAINS.get(ltf, TF_CHAINS["M1"])
    ltf_name = chain["ltf"]
    htf_name = chain["htf"]
    htf2_name = chain["htf2"]

    try:
        # REVERT optim cache worker (deadlock pandas+multiprocessing sur Linux,
        # cf commit 9e50ed2). On recharge les data par chunk (stable).
        df_ltf = load(inst, ltf_name)
        df_htf = load(inst, htf_name)
        try:
            df_htf2 = load(inst, htf2_name)
        except Exception:
            df_htf2 = None
        try:
            df_d1 = load(inst, "D1")
            if len(df_d1) < 10:
                raise FileNotFoundError
        except Exception:
            df_h1 = load(inst, "H1")
            df_d1 = build_d1_from_h1(df_h1)

        htf_dfs_swings = {}
        for tf in ["H1", "H4", "D1"]:
            try:
                htf_dfs_swings[tf] = load(inst, tf)
            except Exception:
                pass
        htf_swings = collect_htf_swings(htf_dfs_swings, swing_strength=3)

        mask = (df_ltf.index >= start_ts) & (df_ltf.index <= end_ts)
        df_ltf_w = df_ltf[mask]
        if len(df_ltf_w) < 100:
            return [], inst, "Pas assez de donnees"

        correlated_dfs = {}
        for corr_name, corr_type in SMT_PAIRS.get(inst, []):
            try:
                df_c = load(corr_name, ltf_name)
                if len(df_c) > 0:
                    mask_c = (df_c.index >= start_ts) & (df_c.index <= end_ts)
                    correlated_dfs[corr_name] = (df_c[mask_c], corr_type)
            except Exception:
                continue

        swing_strength_ltf = get_param(inst, "swing_strength_m1", 2)
        obs = detect_order_blocks(df_ltf_w, swing_strength=swing_strength_ltf)

        # Cache pour acceleration (chunk-local : swings/fvg/breakers sur la fenetre)
        cache = {
            "swings_ltf": find_swings(df_ltf_w, strength=swing_strength_ltf),
            "fvgs_ltf": detect_fvg(df_ltf_w),
            "breakers_ltf": detect_breakers(df_ltf_w),
            "obs_htf": detect_order_blocks(df_htf),
        }
        cache["structure_breaks"] = detect_structure_breaks(df_ltf_w, swings=cache["swings_ltf"])
        cache["htf_trend"] = detect_trend(cache["swings_ltf"], lookback=6)
        if df_htf2 is not None:
            cache["obs_htf2"] = detect_order_blocks(df_htf2)

        df_h1_for_feu_vert = df_htf2 if df_htf2 is not None else None

        # V5 (user 2026-05-20) : virer confirm_ob_with_mss (filtre trop strict).
        # Le ML decide via feature has_mss_nearby.
        from bot_v2.concepts.mss_setup import detect_mss_setups
        mss_setups = detect_mss_setups(
            df_ltf_w,
            structure_breaks=cache["structure_breaks"],
            swings=cache["swings_ltf"],
        )
        cache["mss_setups"] = mss_setups
        print(f"    OB total: {len(obs)}, MSS: {len(mss_setups)}", flush=True)
        prefiltered_obs = obs
        prefiltered_mss = []

        rows = []
        for ob in prefiltered_obs:
            r = evaluate_ob(
                ob, df_ltf_w, df_htf, df_d1, inst,
                ltf_name=ltf_name, htf_name=htf_name,
                df_htf2=df_htf2, htf2_name=htf2_name,
                correlated_dfs=correlated_dfs,
                htf_swings=htf_swings,
                df_h1=df_h1_for_feu_vert,
                min_score=0, min_quality=0,
                cache=cache,
            )
            # On garde uniquement les OB qui passent les regles Vizion ELIMINATOIRES
            # (bias daily contraire, hors KZ, phase A/M, displacement faible, etc.)
            # Ces "vrais candidats" sont ce que le ML va apprendre a filtrer
            if r.verdict != "TRADE":
                continue
            if r.trade_setup is None:
                continue

            f = _extract_features(r, ob, inst, df_ltf=df_ltf_w, df_d1=df_d1, df_htf=df_htf, mss_setups=mss_setups)

            # Simulate trade pour avoir l'outcome
            try:
                setup = r.trade_setup
                lots, risk_usd = compute_position_size(
                    setup.entry_price, setup.stop_loss, inst,
                    balance=60.0, risk_pct=0.10,
                )
                if lots <= 0:
                    continue
                sim_setup = TradeSetup(
                    instrument=inst, direction=setup.direction,
                    entry_price=setup.entry_price, stop_loss=setup.stop_loss,
                    take_profit=setup.take_profit, rr=setup.rr,
                    risk_points=setup.risk_points, reward_points=setup.reward_points,
                    risk_usd=risk_usd, reward_usd=risk_usd * setup.rr,
                    position_size_lots=lots,
                    ob_validation_ts=setup.ob_validation_ts,
                    tp_source=setup.tp_source,
                )
                tr = simulate_trade(sim_setup, df_ltf_w, ob.validation_index + 1)
                f["outcome"] = tr.outcome  # "WIN" | "LOSS" | "NO_FILL" | "PENDING"
                f["pnl_usd"] = tr.pnl_usd
                f["bars_to_exit"] = (tr.exit_index - tr.fill_index) if (tr.exit_index and tr.fill_index) else None
            except Exception as e:
                f["outcome"] = "ERROR"
                f["pnl_usd"] = 0.0
                f["bars_to_exit"] = None

            rows.append(f)

        # =============== BOUCLE MSS SETUPS ===============
        # Les MSS sont evalues SANS filtre daily_bias (user 2026-05-16).
        # Raison : un MSS = changement de structure CONTRE la tendance par definition.
        # Filtrer par daily_bias annulerait l'essence du setup. Le ML decidera.
        # On garde le daily_bias comme FEATURE pour que le ML l'apprenne.
        from bot_v2.concepts.daily_bias import compute_daily_bias
        from bot_v2.concepts.killzones import killzone_at
        for mss in prefiltered_mss:
            if mss.retest_index is None:
                continue

            # On calcule le bias pour la feature, mais on ne filtre PAS
            target_date = mss.retest_ts.normalize()
            bias = compute_daily_bias(df_d1, target_date)
            bias_label = bias.bias if bias is not None else "neutral"

            # Setup : entry/SL deja calcules dans MSSSetup, TP = RR=2 fixed
            risk = abs(mss.entry_price - mss.stop_loss)
            if risk == 0:
                continue
            if mss.direction == "bullish":
                tp = mss.entry_price + risk * 2.0
            else:
                tp = mss.entry_price - risk * 2.0

            # Features MSS - on remplit autant de features que possible
            kz = killzone_at(mss.retest_ts)
            mss_feat = {
                "instrument": inst,
                "ts": mss.retest_ts,
                "direction": mss.direction,
                # Scores generiques (a defaut, valeurs neutres)
                "score": 50,
                "quality": 50,
                "ob_strength": 5,
                "sweep_strength": 0,
                "retest_count": 1,
                "is_unicorn": 0,
                "multi_liq_sweep": 0,
                # Trade
                "rr": 2.0,
                "risk_points": float(risk),
                "tp_source_htf": 0,
                "tp_source_capped": 1,
                # Daily bias (feature, plus filtre)
                "daily_bias_aligned": int(bias_label == mss.direction),
                "daily_bias_neutral": int(bias_label == "neutral"),
                # KZ
                "kz_london": int(kz == "London"),
                "kz_ny_am": int(kz == "NY_AM"),
                "kz_ny_pm": int(kz == "NY_PM"),
                "kz_asia": int(kz == "Asia"),
                "kz_ny_lunch": int(kz == "NY_Lunch"),
                "kz_none": int(kz is None),
                # Confluences
                "has_sync_fvg": 1,  # MSS+FVG par definition
                "has_smt": 0,
                "has_feu_vert": 0,
                "has_breaker_kz": 0,
                "has_mss_fvg": 1,  # MSS_with_FVG explicite
                "has_grandparent": 0,
                "has_po3_dist": 0,
                "has_phase_expansion": 0,
                "has_phase_reversal": 0,
                "has_open_midnight_respect": 0,
                # Structure
                "ob_group_size": 2,  # convention pour MSS
                "bars_sweep_to_validation": mss.retest_index - mss.mss.swing.index,
                "bars_group_to_validation": mss.retest_index - mss.mss.break_index,
                "is_bullish": int(mss.direction == "bullish"),
                "is_mss_setup": 1,  # flag MSS
            }

            # Simulate trade pour outcome
            try:
                lots, risk_usd = compute_position_size(
                    mss.entry_price, mss.stop_loss, inst, balance=60.0, risk_pct=0.10,
                )
                if lots <= 0:
                    continue
                sim_setup = TradeSetup(
                    instrument=inst, direction=mss.direction,
                    entry_price=mss.entry_price, stop_loss=mss.stop_loss,
                    take_profit=tp, rr=2.0,
                    risk_points=float(risk), reward_points=float(abs(tp - mss.entry_price)),
                    risk_usd=risk_usd, reward_usd=risk_usd * 2.0,
                    position_size_lots=lots,
                    ob_validation_ts=mss.retest_ts,
                    tp_source="mss_rr2",
                )
                tr = simulate_trade(sim_setup, df_ltf_w, mss.retest_index + 1)
                mss_feat["outcome"] = tr.outcome
                mss_feat["pnl_usd"] = tr.pnl_usd
                mss_feat["bars_to_exit"] = (tr.exit_index - tr.fill_index) if (tr.exit_index and tr.fill_index) else None
            except Exception:
                mss_feat["outcome"] = "ERROR"
                mss_feat["pnl_usd"] = 0.0
                mss_feat["bars_to_exit"] = None

            rows.append(mss_feat)

        return rows, inst, None
    except Exception as e:
        import traceback
        return [], inst, f"{e}\n{traceback.format_exc()}"


def _chunk_worker(args):
    """Worker au niveau module (picklable par ProcessPoolExecutor sous Windows).

    Args : (inst, cs, ce, ppath, ltf) - ltf par defaut M1.
    """
    if len(args) == 5:
        inst, cs, ce, ppath, ltf = args
    else:
        inst, cs, ce, ppath = args
        ltf = "M1"
    rows, _, error = _process_instrument((inst, cs, ce, ltf))
    if error:
        return inst, cs.date(), None, error
    if rows:
        import pandas as _pd
        _pd.DataFrame(rows).to_parquet(ppath)
    return inst, cs.date(), len(rows), None


def build_dataset(start_ts, end_ts, instruments=None, output_path=None, chunk_months=6, ltf="M1", version_suffix=""):
    """Construit le dataset ML pour la fenetre [start_ts, end_ts].

    Decoupage en CHUNKS de chunk_months mois pour eviter OOM
    (5 ans M1 = 1.7M bougies, le pipeline n'arrive pas a le traiter d'un coup).

    Args:
        version_suffix: ajoute au nom du dossier de chunks (ex: "_V5" -> ml_partial_M1_V5/).
                        Evite de melanger les chunks V4/V5 (features differentes).
    """
    if instruments is None:
        instruments = primary_instruments()

    # Decoupe en chunks de chunk_months mois
    chunks = []
    cur = start_ts
    while cur < end_ts:
        nxt = min(cur + pd.Timedelta(days=30 * chunk_months), end_ts)
        chunks.append((cur, nxt))
        cur = nxt

    print(f"Construction dataset : {start_ts.date()} -> {end_ts.date()}", flush=True)
    print(f"Actifs : {instruments}", flush=True)
    print(f"Decoupage : {len(chunks)} chunks de ~{chunk_months} mois", flush=True)
    print(f"Total taches : {len(instruments)} actifs x {len(chunks)} chunks = {len(instruments)*len(chunks)}", flush=True)

    all_rows = []
    total_tasks = len(instruments) * len(chunks)
    # Dossier separe par TF pour eviter melanges
    # FIX 2026-05-19 : auto-detect Windows vs Linux path
    import sys as _sys
    _root = "c:/Users/Shadow/TradingBot" if _sys.platform == "win32" else "/workspace/TradingBot"
    partial_dir = Path(f"{_root}/data/ml_partial_{ltf}{version_suffix}")
    partial_dir.mkdir(parents=True, exist_ok=True)

    # Construit la liste des taches a faire (skip celles deja sauvegardees)
    tasks_to_do = []
    skipped = 0
    for inst in instruments:
        for chunk_start, chunk_end in chunks:
            chunk_id = f"{inst}_{ltf}_{chunk_start.strftime('%Y%m%d')}"
            partial_path = partial_dir / f"{chunk_id}.parquet"
            if partial_path.exists():
                df_existing = pd.read_parquet(partial_path)
                all_rows.extend(df_existing.to_dict("records"))
                skipped += 1
            else:
                tasks_to_do.append((inst, chunk_start, chunk_end, partial_path, ltf))

    print(f"Chunks deja sauvegardes (skip) : {skipped}/{total_tasks}", flush=True)
    print(f"Chunks a traiter : {len(tasks_to_do)}", flush=True)

    if not tasks_to_do:
        print("Tout est deja fait.", flush=True)
    else:
        # Workers auto-detect.
        # V5 (2026-05-20) : cap monte a 64 (RAM serveur 251GB suffit pour 64 workers).
        # Avant : 32 max (limite arbitraire deadlock pandas+multiprocessing).
        # Override via env var N_WORKERS si besoin.
        import os
        env_workers = os.environ.get("N_WORKERS")
        if env_workers:
            n_workers = min(int(env_workers), len(tasks_to_do))
        else:
            cpu_count = os.cpu_count() or 4
            if cpu_count <= 8:
                n_workers = min(6, len(tasks_to_do))
            else:
                # Serveur : 64 max (gain 2x sur EPYC 96+ cores)
                n_workers = min(64, cpu_count, len(tasks_to_do))
        print(f"CPU cores detected : {os.cpu_count()}, workers utilises : {n_workers}", flush=True)

        done_count = 0
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_chunk_worker, t): t for t in tasks_to_do}
            for fut in as_completed(futures):
                inst, date, n_rows, error = fut.result()
                done_count += 1
                if error:
                    print(f"  [{done_count}/{len(tasks_to_do)}] {inst} {date}: ERREUR {error[:150]}", flush=True)
                    continue
                print(f"  [{done_count}/{len(tasks_to_do)}] {inst} {date}: {n_rows} candidats", flush=True)
                # Charge le partial sauve par le worker dans all_rows.
                # Cohenrent avec write (ligne 403) : nom = {inst}_{ltf}_{date}.
                chunk_id = f"{inst}_{ltf}_{pd.Timestamp(date).strftime('%Y%m%d')}"
                ppath = partial_dir / f"{chunk_id}.parquet"
                if ppath.exists():
                    df_partial = pd.read_parquet(ppath)
                    all_rows.extend(df_partial.to_dict("records"))

    df = pd.DataFrame(all_rows)
    print(f"\nDataset total : {len(df)} lignes")
    if len(df) == 0:
        print("AUCUN candidat trouve, rien a sauvegarder")
        return df

    # Stats outcomes
    print("\nDistribution outcomes :")
    print(df["outcome"].value_counts().to_string())

    closed = df[df["outcome"].isin(["WIN", "LOSS"])]
    if len(closed) > 0:
        wr = (closed["outcome"] == "WIN").mean() * 100
        print(f"\nWR brut (sans ML) : {wr:.1f}% sur {len(closed)} trades fermes")

    # Save
    if output_path is None:
        # Nom inclut TF si different de M1 (M1 garde compat avec main historique)
        if ltf == "M1":
            output_path = Path(f"{_root}/data/ml_dataset.parquet")
        else:
            output_path = Path(f"{_root}/data/ml_dataset_{ltf}.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path)
    print(f"\nDataset sauve : {output_path}")
    return df


def main():
    # PHASE 1 v2 : XAUUSD ONLY, 4 ans (2022-05 -> 2026-05).
    # v1 sur 2 ans (583 trades train) avait overfit : train WR 91% vs test 51%.
    # 4 ans = ~1750 trades train -> meilleur generalisation attendue.
    # Phase 2 (user 2026-05-16) : NAS100, meme methodologie que XAUUSD v7
    instruments = ["NAS100"]
    df = load("NAS100", "M1")
    earliest_end = df.index[-1]
    latest_start = earliest_end - pd.Timedelta(days=365 * 2)  # 2 ans
    print(f"Plage utilisee : {latest_start.date()} -> {earliest_end.date()} (2 ans)")
    print(f"Actif : NAS100 (Phase 2)")
    build_dataset(latest_start, earliest_end, instruments, chunk_months=3)


if __name__ == "__main__":
    main()
