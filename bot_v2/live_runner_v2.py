"""Bot live V2 multi-asset Vizion ICT/SMC - ORDRES LIMIT (= backtest).

Difference vs live_runner.py (V1) :
- V1 : ordre MARKET au moment du scan -> entry = prix actuel (peut deriver de l'OB)
- V2 : ordre LIMIT a setup.entry_price (= prix OB) -> fill exact comme backtest

Pourquoi V2 :
Dans le backtest, simulate_trade() attend que le prix REVIENNE toucher setup.entry_price
(pullback dans l'OB). Donc SL/TP figes sont coherents avec l'entry.

En V1 live, on placait market order au prix actuel mais avec SL/TP de l'OB d'origine.
Si le prix derivait entre detection et execution, le RR effectif etait casse
(ex: trade JP225 #2 du 19/05 : entry 60509 mais SL/TP figes a 60667/60478 = RR 0.19).

V2 corrige ce probleme : place un BUY/SELL LIMIT au prix OB exact, avec expiration 60min.
Si le prix touche entry_price -> trade execute avec RR=2 garanti.
Si pas touche en 60min -> ordre annule automatiquement.

Workflow toutes les SCAN_INTERVAL_SEC :
1. Pour chaque actif : fetch les bougies fraiches via MT5
2. Pipeline Vizion (OB+MSS, evaluate_ob, ML filter)
3. Si setup valide ET pas en cooldown ET <3 trades+pending concurrents :
    -> Place ordre PENDING LIMIT au prix OB
    -> Log dans SQLite
4. Reconcilie les trades fermes par broker (SL/TP touche) -> update journal
5. Nettoie les pending expires

Decision user 2026-05-19 : V2 = LIMIT (vrai fix du bug RR du 19/05 matin)
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any

# === V10 (2026-05-22) : aligne le live sur le training V10 ===
# Le dataset/modeles V10 ont ete construits avec swing_strength=1 (+88% d'OB)
# et RR=1.5. Le live DOIT utiliser les memes valeurs sinon il voit des OB
# differents et des SL/TP differents -> divergence training/live.
# On force les env vars AVANT l'import de bot_v2.config.
os.environ.setdefault("SWS_OVERRIDE", "1")
os.environ.setdefault("RR_OVERRIDE", "1.5")

import pandas as pd

from bot_v2 import ml_filter
from bot_v2.backtest import simulate_trade  # juste pour calcul lot
from bot_v2.concepts.breaker import detect_breakers
from bot_v2.concepts.daily_bias import build_d1_from_h1
from bot_v2.concepts.fvg import detect_fvg
from bot_v2.concepts.htf_swings import collect_htf_swings
from bot_v2.concepts.liquidity import find_swings
from bot_v2.concepts.mss_setup import detect_mss_setups, confirm_ob_with_mss
from bot_v2.concepts.order_block import detect_order_blocks
from bot_v2.concepts.structure import detect_structure_breaks, detect_trend
from bot_v2.config import SMT_PAIRS, INSTRUMENTS, get_param
from bot_v2.data_buffer import DataBuffer
from bot_v2.live_state import LiveState
from bot_v2.mt5_executor import MT5Executor
from bot_v2.pipeline import evaluate_ob
from bot_v2.push_dashboard import DashboardPusher
from bot_v2.trade_setup import compute_position_size


log = logging.getLogger("live")


# ========== CONFIG LIVE ==========

# Actifs traders en live (alignement avec backtest valide)
# Phase 4 (user 2026-05-18) : +7 nouveaux actifs (SP500, DJ30, UK100, FRA40, JP225, USDCAD, USDCHF)
# NZDUSD ecarte (WR Test 53.5% trop faible)
# JP225 desactive (user 2026-05-19) : ecart Dukascopy/Vantage -410 pts (0.67%) sur session
# off-hours Asia. Setup OB Duka non reproductible sur Vantage. A re-evaluer si on
# trouve data plus proche de Vantage.
LIVE_ASSETS = [
    # Phase 1-3 (8 actifs valides)
    "XAUUSD", "NAS100", "GER40", "BTCUSD", "EURUSD", "GBPUSD", "AUDUSD", "USDJPY",
    # Phase 4 (4 indices + 2 forex, JP225 ecarte)
    "SP500", "DJ30", "UK100", "FRA40", "USDCAD", "USDCHF",
]

# Reduire la liste en mode test (5€ : pas assez pour BTC/NAS/GER lot min)
TEST_MODE_ASSETS = ["XAUUSD", "EURUSD", "GBPUSD", "AUDUSD", "USDJPY", "USDCAD", "USDCHF"]
TEST_MODE_THRESHOLD = 50.0  # Balance < 50€ = mode test

# Scan interval (user 2026-05-18 : 30s -> 15s pour plus de reactivite)
SCAN_INTERVAL_SEC = 5  # FIX 2026-05-20 : 15 -> 5 sec pour temps reel

# Cooldown par actif
COOLDOWN_SEC = 15 * 60

# V4 (user 2026-05-20) : MAX_CONCURRENT 8 (au lieu de 3) - le bot V3.5 a 14 actifs
# avec WR 73-80%, on peut avoir plus de slots en parallele (decorrelation).
MAX_CONCURRENT = 8

# MM V4 (user 2026-05-20) : risque reduit pour multi-trades V3.5
# Avec MAX_CONCURRENT=8 * 5% = 40% exposition max = safe
# Au-dessus 5000€ : 3% pour limiter DD a 24% max
THRESHOLD_SAFE_MODE = 5000.0
RISK_PCT_AGGRESSIVE = 0.05    # 5% par trade jusqu'a 5000€ (vs 30% V2)
RISK_PCT_SAFE = 0.03           # 3% au-dessus de 5000€
RISK_PCT_TEST = 0.02

# Magic number (identifie nos trades dans MT5)
BOT_MAGIC = 20260517

# History candles to fetch
# V5.4 (2026-05-21) : utilise DataBuffer (data_vantage/) au lieu de MT5 direct.
# Buffer charge 200k+ M1 mais on lit que ce qu'il faut pour matcher le TRAINING.
#
# TRAINING (ml_dataset.py) calcule sur :
#   - df_ltf_w (M1) : chunk 3 mois = ~88k M1
#   - df_htf (M15) : chunk + 30j buffer = ~4 mois = ~11k M15
#   - df_htf2 (H1) : chunk + 30j buffer = ~2800 H1
#   - df_d1 : ~120 D1 (4 mois)
# Le LIVE doit utiliser les MEMES tailles pour matcher les features.
N_BARS_M1 = 88000    # = chunk 3 mois training (~88k)
N_BARS_M15 = 11000   # = chunk + buffer 4 mois training
N_BARS_H1 = 2800     # = chunk + buffer 4 mois training
N_BARS_D1 = 120      # = chunk + buffer 4 mois training

# Buffers data par actif (init dans main(), un par asset)
DATA_BUFFERS: dict[str, "DataBuffer"] = {}

# Pusher dashboard (module-global, init dans run_live, lu par execute_setup)
_PUSHER: "DashboardPusher | None" = None


def get_risk_pct(balance: float) -> float:
    if balance < TEST_MODE_THRESHOLD:
        return RISK_PCT_TEST
    if balance < THRESHOLD_SAFE_MODE:
        return RISK_PCT_AGGRESSIVE
    return RISK_PCT_SAFE


def get_active_assets(balance: float) -> list[str]:
    """Mode test (balance < 50€) : actifs avec lot min petit uniquement."""
    if balance < TEST_MODE_THRESHOLD:
        return TEST_MODE_ASSETS
    return LIVE_ASSETS


# ========== MODELS ML ==========

_models_cache: dict[str, tuple[Any, list[str]]] = {}

# Repertoire racine du repo (auto-detecte selon l'OS / user)
ROOT_DIR = Path(__file__).resolve().parent.parent
BOT_V2_DIR = ROOT_DIR / "bot_v2"


def load_model(instrument: str) -> tuple[Any, list[str]] | None:
    """Charge le meilleur modele disponible : V10 > V9 > V8 > V7 > V5 > V4 > V3.5 > V2.

    V10 (2026-05-22) : 15 nouvelles features (PO3 enrichi, momentum, displacement,
    atr_regime), swing_strength=1 (+88% d'OB), RR=1.5, hyperparametres 'deeper'.
    AUC OOS moy 0.741. WR@0.70 81%. Volume x2.5 vs V9. IMPORTANT : le live DOIT
    tourner avec swing_strength=1 (cf SWS_OVERRIDE) pour voir les memes OB que
    le dataset de training.

    V9 (2026-05-22) : fix data leakages multiples (retests, MSS, FVG, breaker...).
    V8 (2026-05-21) : fix data leakage PDH/PDL/D1_open uniquement.
    """
    if instrument in _models_cache:
        return _models_cache[instrument]

    # Cascade : V10 > V9 > V8 > V7 > V5 Admiral > V4 > V3_5 > V2 legacy
    candidates = [
        (BOT_V2_DIR / f"ml_model_{instrument}_vantage_v10.pkl",
         BOT_V2_DIR / f"ml_features_{instrument}_vantage_v10.json"),
        (BOT_V2_DIR / f"ml_model_{instrument}_vantage_v9.pkl",
         BOT_V2_DIR / f"ml_features_{instrument}_vantage_v9.json"),
        (BOT_V2_DIR / f"ml_model_{instrument}_vantage_v8.pkl",
         BOT_V2_DIR / f"ml_features_{instrument}_vantage_v8.json"),
        (BOT_V2_DIR / f"ml_model_{instrument}_vantage_v7.pkl",
         BOT_V2_DIR / f"ml_features_{instrument}_vantage_v7.json"),
        (BOT_V2_DIR / f"ml_model_{instrument}_admiral_v5.pkl",
         BOT_V2_DIR / f"ml_features_{instrument}_admiral_v5.json"),
        (BOT_V2_DIR / f"ml_model_{instrument}_admiral_v4.pkl",
         BOT_V2_DIR / f"ml_features_{instrument}_admiral_v4.json"),
        (BOT_V2_DIR / f"ml_model_{instrument}_admiral_v3_5.pkl",
         BOT_V2_DIR / f"ml_features_{instrument}_admiral_v3_5.json"),
        (BOT_V2_DIR / f"ml_model_{instrument}.pkl",
         BOT_V2_DIR / f"ml_features_{instrument}.json"),
    ]
    model_path = feat_path = None
    version = None
    for mp, fp in candidates:
        if mp.exists() and fp.exists():
            model_path = mp
            feat_path = fp
            if "_vantage_v10" in mp.name:
                version = "V10-Vantage-MoreVolume"
            elif "_vantage_v9" in mp.name:
                version = "V9-Vantage-NoLeakage-FULL"
            elif "_vantage_v8" in mp.name:
                version = "V8-Vantage-PartialFix"
            elif "_vantage_v7" in mp.name:
                version = "V7-Vantage-8ans"
            elif "_v5" in mp.name:
                version = "V5-Admiral"
            elif "_v4" in mp.name:
                version = "V4"
            elif "_v3_5" in mp.name:
                version = "V3.5"
            else:
                version = "V2-legacy"
            break

    if model_path is None:
        log.warning(f"Modele manquant pour {instrument} (V4/V3.5/V2 tous absents), skip")
        return None

    log.info(f"Modele {instrument} : {version} ({model_path.name})")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    features = json.load(open(feat_path))["features"]
    _models_cache[instrument] = (model, features)
    return model, features


def predict_proba(model, features, r, ob, instrument, df_ltf=None, df_d1=None, mss_setups=None) -> float:
    # FIX V5.1 (2026-05-20) : passer mss_setups pour la feature has_mss_nearby.
    # Sans ca, has_mss_nearby=0 toujours -> ML proba chute ~0.20 -> bot ne trade jamais.
    feats = ml_filter._features_from_result(r, ob, instrument, df_ltf=df_ltf, df_d1=df_d1, mss_setups=mss_setups)
    X = pd.DataFrame([[feats.get(f, 0) for f in features]], columns=features)
    return float(model.predict_proba(X)[0, 1])


# ========== PIPELINE PAR ACTIF (split fetch / compute pour ProcessPool) ==========


def fetch_payload(
    mt5_exec: MT5Executor, instrument: str, balance: float | None, debug_diag: bool,
    evaluated_keys: set | None = None,
) -> dict | None:
    """V5.7 (2026-05-21) : phase FETCH (process principal).

    Recupere les bougies M1/M15/H1/H4/D1 + les correles SMT. Construit un
    dict picklable a envoyer aux workers ProcessPool. AUCUN calcul ICT ici.

    evaluated_keys : set des timestamps (str) d'OB deja evalues lors d'un
    cycle precedent -> compute_asset les exclut (1 OB = 1 evaluation, OOS).

    Retourne None si les donnees sont insuffisantes (skip cet actif).
    """
    buffer = DATA_BUFFERS.get(instrument)
    if buffer is None:
        # Fallback : fetch direct MT5 (max 80k) - utile si parquet manquant.
        log.warning(f"DIAG {instrument}: pas de buffer, fallback fetch MT5")
        df_m1 = mt5_exec.get_bars(instrument, "M1", N_BARS_M1, force_sync=True)
        if df_m1 is None or len(df_m1) < 200:
            return None
        df_m1 = df_m1.iloc[:-1]
        df_m15 = mt5_exec.get_bars(instrument, "M15", N_BARS_M15)
        df_h1 = mt5_exec.get_bars(instrument, "H1", N_BARS_H1)
        if df_m15 is None or df_h1 is None:
            return None
        df_m15 = df_m15.iloc[:-1]
        df_h1 = df_h1.iloc[:-1]
        try:
            df_d1 = mt5_exec.get_bars(instrument, "D1", N_BARS_D1)
            if df_d1 is None or len(df_d1) < 10:
                raise ValueError
        except Exception:
            df_d1 = build_d1_from_h1(df_h1)
        try:
            df_h4 = mt5_exec.get_bars(instrument, "H4", 500)
        except Exception:
            df_h4 = None
    else:
        n_new = buffer.update()
        if debug_diag and n_new > 0:
            log.info(f"BUFFER {instrument}: +{n_new} nouvelles M1")
        df_m1 = buffer.get_m1(n=N_BARS_M1)
        if df_m1 is None or len(df_m1) < 200:
            return None
        df_m1 = df_m1.iloc[:-1]
        df_m15 = buffer.get_m15(n=N_BARS_M15)
        df_h1 = buffer.get_h1(n=N_BARS_H1)
        df_h4 = buffer.get_h4(n=500)
        df_d1 = buffer.get_d1(n=N_BARS_D1)
        if df_d1 is None or len(df_d1) < 10:
            df_d1 = build_d1_from_h1(df_h1)

    # SMT correles (depuis buffer si dispo)
    correlated_dfs: dict[str, tuple[pd.DataFrame, str]] = {}
    for corr_name, corr_type in SMT_PAIRS.get(instrument, []):
        corr_buffer = DATA_BUFFERS.get(corr_name)
        try:
            if corr_buffer is not None:
                df_c = corr_buffer.get_m1(n=N_BARS_M1)
            else:
                df_c = mt5_exec.get_bars(corr_name, "M1", N_BARS_M1)
            if df_c is not None and len(df_c) > 0:
                correlated_dfs[corr_name] = (df_c, corr_type)
        except Exception:
            continue

    return {
        "instrument": instrument,
        "balance": balance,
        "debug_diag": debug_diag,
        "df_m1": df_m1,
        "df_m15": df_m15,
        "df_h1": df_h1,
        "df_h4": df_h4,
        "df_d1": df_d1,
        "correlated_dfs": correlated_dfs,
        "evaluated_keys": evaluated_keys or set(),
    }


def compute_asset(payload: dict) -> dict:
    """V5.7 (2026-05-21) : phase COMPUTE (workers ProcessPool).

    Reproduit fidelement la logique de scan_asset() a partir de la detection
    OB+MSS. Aucun acces MT5 / DATA_BUFFERS / state — tout vient du payload.

    Returns:
        Dict picklable : {instrument, setups, rejected_log, diag_log, latency_ms,
        error}. setups = liste de dicts (instrument, ts, ob, r, proba, df_m1).
    """
    import time as _time
    _scan_start = _time.time()

    instrument = payload["instrument"]
    df_m1 = payload["df_m1"]
    df_m15 = payload["df_m15"]
    df_h1 = payload["df_h1"]
    df_h4 = payload["df_h4"]
    df_d1 = payload["df_d1"]
    correlated_dfs = payload["correlated_dfs"]
    balance = payload["balance"]
    debug_diag = payload["debug_diag"]

    diag_log: list[str] = []
    rejected_log: list[tuple] = []

    try:
        # HTF swings (D1, H4, H1)
        htf_dfs: dict[str, pd.DataFrame] = {"H1": df_h1, "D1": df_d1}
        if df_h4 is not None and len(df_h4) > 0:
            htf_dfs["H4"] = df_h4
        htf_swings = collect_htf_swings(htf_dfs, swing_strength=3)

        # Detection OB+MSS
        _t = _time.time()
        sws = get_param(instrument, "swing_strength_m1", 2)
        obs = detect_order_blocks(df_m1, swing_strength=sws)  # max_group_size defaut=5, ALIGNE OOS
        cache = {
            "swings_ltf": find_swings(df_m1, strength=sws),
            "fvgs_ltf": detect_fvg(df_m1),
            "breakers_ltf": detect_breakers(df_m1),
            "obs_htf": detect_order_blocks(df_m15),
        }
        cache["structure_breaks"] = detect_structure_breaks(
            df_m1, swings=cache["swings_ltf"], fvgs=cache["fvgs_ltf"]
        )
        cache["htf_trend"] = detect_trend(cache["swings_ltf"], lookback=6)
        cache["obs_htf2"] = detect_order_blocks(df_h1)
        cache_build_s = _time.time() - _t

        mss_setups = detect_mss_setups(
            df_m1, structure_breaks=cache["structure_breaks"], swings=cache["swings_ltf"],
            fvgs=cache["fvgs_ltf"],
        )
        # V5.1 : virer filtre dur confirm_ob_with_mss (le ML decide via has_mss_nearby).
        obs_confirmed = obs

        # Filtre : OB des 60 dernieres minutes.
        # FIX V8 (2026-05-21) : recent_cutoff remis a 60min (etait 3min -> BUG).
        # Avec 3min le bot ne voyait qu'un OB ultra-frais par cycle et ratait
        # tous les OB qui atteignent ML>=0.75 quelques minutes apres leur
        # validation (quand le contexte se developpe). Resultat : 0 trade en
        # live alors que le backtest 24h trouve ~23 trades/jour.
        # Avec 60min : le bot re-evalue tous les OB recents a chaque cycle,
        # place le trade des qu'un atteint le seuil. _seen_setups dedoublonne.
        now = df_m1.index[-1]
        recent_cutoff = now - pd.Timedelta(minutes=60)
        obs_recent = [
            ob for ob in obs_confirmed
            if df_m1.index[ob.validation_index] >= recent_cutoff
        ]

        # NOTE (2026-05-21 soir) : le filtre "1 OB = 1 evaluation" etait
        # CONTRE-PRODUCTIF. L'OOS evalue 1 fois par OB mais avec tout l'historique
        # parquet disponible (donc des bougies POSTERIEURES a la validation, ce
        # qui simule un OB "mur" avec son retest). Le live au moment de la
        # validation n'a PAS ces bougies futures -> proba differente.
        # Solution : on laisse le live re-evaluer l'OB a chaque cycle pendant
        # recent_cutoff=60min. La proba grimpe au fil des bougies (retest,
        # displacement) -> rejoint la proba OOS au bout de quelques minutes.
        # Le dedoublonnage des TRADES PLACES est gere ailleurs via _seen_setups.
        # _evaluated_obs n'est plus utilise pour filtrer (mais on continue de
        # le maintenir pour des stats / audit eventuel).

        if debug_diag:
            from bot_v2.concepts.killzones import killzone_at
            last_ts = df_m1.index[-1]
            latest_ob_ts = "aucun"
            if obs_confirmed:
                latest_ob_ts = df_m1.index[max(ob.validation_index for ob in obs_confirmed)]
            kz_now = killzone_at(last_ts) or "AUCUNE"
            cutoff_60 = last_ts - pd.Timedelta(minutes=60)
            cutoff_15 = last_ts - pd.Timedelta(minutes=15)
            n_obs_60min = sum(
                1 for ob in obs_confirmed
                if df_m1.index[ob.validation_index] >= cutoff_60
            )
            n_obs_15min = sum(
                1 for ob in obs_confirmed
                if df_m1.index[ob.validation_index] >= cutoff_15
            )
            diag_log.append(
                f"DIAG {instrument}: M1={len(df_m1)} M15={len(df_m15)} H1={len(df_h1)} "
                f"last_bar={last_ts} kz={kz_now} | OB_brut={len(obs)} OB+MSS={len(obs_confirmed)} "
                f"latest_OB_MSS_ts={latest_ob_ts} OB_60min={n_obs_60min} OB_15min={n_obs_15min} "
                f"OB_cutoff={len(obs_recent)}"
            )

            # Diag detaille des OB 15min (pour comprendre pourquoi pas trade)
            obs_15min = [
                ob for ob in obs_confirmed
                if df_m1.index[ob.validation_index] >= cutoff_15
            ]
            if obs_15min:
                loaded_diag = load_model(instrument)
                if loaded_diag is not None:
                    model_d, features_d = loaded_diag
                    thr_d = ml_filter.get_dynamic_threshold(instrument, balance)
                    for ob in obs_15min:
                        obts = df_m1.index[ob.validation_index]
                        age_min = (last_ts - obts).total_seconds() / 60
                        try:
                            r_d = evaluate_ob(
                                ob, df_m1, df_m15, df_d1, instrument,
                                ltf_name="M1", htf_name="M15",
                                df_htf2=df_h1, htf2_name="H1",
                                correlated_dfs=correlated_dfs,
                                htf_swings=htf_swings,
                                df_h1=df_h1, min_score=0, min_quality=0,
                                cache=cache,
                            )
                            if r_d.verdict == "TRADE" and r_d.trade_setup is not None:
                                proba_d = predict_proba(
                                    model_d, features_d, r_d, ob, instrument,
                                    df_ltf=df_m1, df_d1=df_d1, mss_setups=mss_setups,
                                )
                                verdict_d = (
                                    f"TRADE ml={proba_d:.3f} "
                                    f"{'OK' if proba_d >= thr_d else f'<{thr_d}'}"
                                )
                            else:
                                verdict_d = f"REJET: {(r_d.rejection_reason or 'no_trade')[:40]}"
                        except Exception as e:
                            verdict_d = f"EXCEPTION: {str(e)[:30]}"
                        diag_log.append(
                            f"  DIAG {instrument} 15min OB | ts={obts} age={age_min:.1f}min | {verdict_d}"
                        )

        if not obs_recent:
            _scan_total = _time.time() - _scan_start
            diag_log.append(
                f"LATENCY {instrument}: total={_scan_total*1000:.0f}ms | cache_build={cache_build_s*1000:.0f}ms"
            )
            return {
                "instrument": instrument, "setups": [], "rejected_log": rejected_log,
                "diag_log": diag_log, "latency_ms": int(_scan_total * 1000), "error": None,
            }

        # Pipeline Vizion + ML
        loaded = load_model(instrument)
        if loaded is None:
            return {
                "instrument": instrument, "setups": [], "rejected_log": rejected_log,
                "diag_log": diag_log, "latency_ms": 0, "error": None,
            }
        model, features = loaded
        threshold = ml_filter.get_dynamic_threshold(instrument, balance)

        valid_setups: list[dict] = []
        diag_reasons: dict[str, int] = {}
        diag_ml_probas: list[float] = []
        for ob in obs_recent:
            try:
                r = evaluate_ob(
                    ob, df_m1, df_m15, df_d1, instrument,
                    ltf_name="M1", htf_name="M15",
                    df_htf2=df_h1, htf2_name="H1",
                    correlated_dfs=correlated_dfs,
                    htf_swings=htf_swings,
                    df_h1=df_h1, min_score=0, min_quality=0,
                    cache=cache,
                )
            except Exception as e:
                diag_reasons["evaluate_ob_exception"] = (
                    diag_reasons.get("evaluate_ob_exception", 0) + 1
                )
                continue

            if r.verdict != "TRADE" or r.trade_setup is None:
                reason = r.rejection_reason or "no_trade"
                # V5.9c : enrichir le rejected_log avec les details OB et pipeline
                # pour les afficher dans le dashboard (entry/SL/TP si dispo).
                ts_ob = df_m1.index[ob.validation_index]
                _entry = _sl = _tp = _rr = None
                if r.trade_setup is not None:
                    _entry = float(r.trade_setup.entry_price)
                    _sl = float(r.trade_setup.stop_loss)
                    _tp = float(r.trade_setup.take_profit)
                    _rr = float(r.trade_setup.rr)
                rejected_log.append((
                    instrument, ts_ob, ob.direction, reason, None,
                    int(r.score) if r.score is not None else None,
                    # Champs additionnels (compat : les anciens listeners
                    # n'utilisent que les 6 premiers via *entry)
                    {
                        "threshold": float(threshold),
                        "entry": _entry, "sl": _sl, "tp": _tp, "rr": _rr,
                        "ob_high": float(ob.ob_high), "ob_low": float(ob.ob_low),
                    },
                ))
                diag_reasons[reason] = diag_reasons.get(reason, 0) + 1
                continue

            proba = predict_proba(
                model, features, r, ob, instrument,
                df_ltf=df_m1, df_d1=df_d1, mss_setups=mss_setups,
            )
            diag_ml_probas.append(proba)
            if proba < threshold:
                ts_ob = df_m1.index[ob.validation_index]
                _entry = float(r.trade_setup.entry_price)
                _sl = float(r.trade_setup.stop_loss)
                _tp = float(r.trade_setup.take_profit)
                _rr = float(r.trade_setup.rr)
                rejected_log.append((
                    instrument, ts_ob, ob.direction,
                    f"ml_below_thr_{proba:.3f}", float(proba), int(r.score),
                    {
                        "threshold": float(threshold),
                        "entry": _entry, "sl": _sl, "tp": _tp, "rr": _rr,
                        "ob_high": float(ob.ob_high), "ob_low": float(ob.ob_low),
                    },
                ))
                diag_reasons[f"ml_below_{threshold:.2f}"] = (
                    diag_reasons.get(f"ml_below_{threshold:.2f}", 0) + 1
                )
                continue

            valid_setups.append({
                "instrument": instrument,
                "ts": df_m1.index[ob.validation_index],
                "ob": ob,
                "r": r,
                "proba": proba,
                "df_m1": df_m1,
            })

        if debug_diag and (diag_reasons or diag_ml_probas):
            proba_summary = ""
            if diag_ml_probas:
                proba_summary = (
                    f" | probas_ML={[f'{p:.3f}' for p in diag_ml_probas]} "
                    f"(seuil={threshold:.2f})"
                )
            diag_log.append(
                f"DIAG {instrument}: rejets = "
                f"{dict(sorted(diag_reasons.items(), key=lambda x: -x[1]))}{proba_summary}"
            )

        _scan_total = _time.time() - _scan_start
        diag_log.append(
            f"LATENCY {instrument}: total={_scan_total*1000:.0f}ms | cache_build={cache_build_s*1000:.0f}ms"
        )
        return {
            "instrument": instrument, "setups": valid_setups, "rejected_log": rejected_log,
            "diag_log": diag_log, "latency_ms": int(_scan_total * 1000), "error": None,
        }

    except Exception as e:
        import traceback
        return {
            "instrument": instrument, "setups": [], "rejected_log": rejected_log,
            "diag_log": diag_log, "latency_ms": 0,
            "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        }


def scan_asset(mt5_exec: MT5Executor, instrument: str, state: LiveState, balance: float | None = None, debug_diag: bool = False) -> list[dict]:
    """Scanne un actif : fetch bougies + pipeline Vizion + ML filter.

    Args:
        balance: solde courant (passe a get_dynamic_threshold, seuil V5 = 0.75 partout)
        debug_diag: log diagnostic complet (bougies, OB, rejections) - 1 fois/5min.

    Returns:
        Liste de setups valides PRETS a etre executes (deja filtres ML, hors cooldown).
    """
    import time as _time
    _scan_start = _time.time()
    _timings = {}

    # 1. Buffer M1 (V5.4 : 7+ mois d'historique en memoire, vs limite MT5 80k)
    _t = _time.time()
    buffer = DATA_BUFFERS.get(instrument)
    if buffer is None:
        # Fallback : fetch direct MT5 (max 80k)
        log.warning(f"DIAG {instrument}: pas de buffer, fallback fetch MT5")
        df_m1 = mt5_exec.get_bars(instrument, "M1", N_BARS_M1, force_sync=True)
        if df_m1 is None or len(df_m1) < 200:
            if debug_diag:
                log.info(f"DIAG {instrument}: M1 KO (fallback)")
            return []
        df_m1 = df_m1.iloc[:-1]
        df_m15 = mt5_exec.get_bars(instrument, "M15", N_BARS_M15)
        df_h1 = mt5_exec.get_bars(instrument, "H1", N_BARS_H1)
        if df_m15 is None or df_h1 is None:
            return []
        df_m15 = df_m15.iloc[:-1]
        df_h1 = df_h1.iloc[:-1]
        try:
            df_d1 = mt5_exec.get_bars(instrument, "D1", N_BARS_D1)
            if df_d1 is None or len(df_d1) < 10:
                raise ValueError
        except Exception:
            df_d1 = build_d1_from_h1(df_h1)
        try:
            df_h4 = mt5_exec.get_bars(instrument, "H4", 500)
        except Exception:
            df_h4 = None
    else:
        # Buffer disponible : update (ajoute nouvelles M1) puis lit tout en memoire
        n_new = buffer.update()
        if debug_diag and n_new > 0:
            log.info(f"BUFFER {instrument}: +{n_new} nouvelles M1")
        # IMPORTANT : on limite a N_BARS_M1 (80k = ~55j) pour ne PAS calculer
        # swings/fvg/breakers sur 200k bougies (= ~30s/scan x 14 actifs = trop lent)
        # Le buffer stocke 200k mais on lit seulement les dernieres 80k
        df_m1 = buffer.get_m1(n=N_BARS_M1)
        if df_m1 is None or len(df_m1) < 200:
            if debug_diag:
                log.info(f"DIAG {instrument}: buffer M1 vide")
            return []
        df_m1 = df_m1.iloc[:-1]  # vire bougie en cours
        df_m15 = buffer.get_m15(n=N_BARS_M15)
        df_h1 = buffer.get_h1(n=N_BARS_H1)
        df_h4 = buffer.get_h4(n=500)
        df_d1 = buffer.get_d1(n=N_BARS_D1)
        if df_d1 is None or len(df_d1) < 10:
            df_d1 = build_d1_from_h1(df_h1)
    _timings["fetch_m1"] = _time.time() - _t
    _timings["fetch_htf"] = 0  # inclus dans fetch_m1 si buffer

    # HTF swings (D1, H4, H1)
    _t = _time.time()
    htf_dfs: dict[str, pd.DataFrame] = {"H1": df_h1, "D1": df_d1}
    if df_h4 is not None and len(df_h4) > 0:
        htf_dfs["H4"] = df_h4
    htf_swings = collect_htf_swings(htf_dfs, swing_strength=3)
    _timings["htf_swings"] = _time.time() - _t

    # SMT correles (depuis buffer si dispo)
    _t = _time.time()
    correlated_dfs = {}
    for corr_name, corr_type in SMT_PAIRS.get(instrument, []):
        corr_buffer = DATA_BUFFERS.get(corr_name)
        try:
            if corr_buffer is not None:
                df_c = corr_buffer.get_m1(n=N_BARS_M1)
            else:
                df_c = mt5_exec.get_bars(corr_name, "M1", N_BARS_M1)
            if df_c is not None and len(df_c) > 0:
                correlated_dfs[corr_name] = (df_c, corr_type)
        except Exception:
            continue
    _timings["smt_fetch"] = _time.time() - _t

    # 2. Detection OB+MSS
    _t = _time.time()
    sws = get_param(instrument, "swing_strength_m1", 2)
    obs = detect_order_blocks(df_m1, swing_strength=sws)  # max_group_size defaut=5, ALIGNE OOS
    cache = {
        "swings_ltf": find_swings(df_m1, strength=sws),
        "fvgs_ltf": detect_fvg(df_m1),
        "breakers_ltf": detect_breakers(df_m1),
        "obs_htf": detect_order_blocks(df_m15),
    }
    # V5.5 (2026-05-21) : on passe fvgs_ltf pre-calcule a structure_breaks ET
    # mss_setups -> evite 2 detect_fvg redondants sur 88k bougies (~3.5s gagne).
    cache["structure_breaks"] = detect_structure_breaks(
        df_m1, swings=cache["swings_ltf"], fvgs=cache["fvgs_ltf"]
    )
    cache["htf_trend"] = detect_trend(cache["swings_ltf"], lookback=6)
    cache["obs_htf2"] = detect_order_blocks(df_h1)
    _timings["cache_build"] = _time.time() - _t

    mss_setups = detect_mss_setups(
        df_m1, structure_breaks=cache["structure_breaks"], swings=cache["swings_ltf"],
        fvgs=cache["fvgs_ltf"],
    )
    _timings["mss_detect"] = _time.time() - _t
    # V5.1 (2026-05-20) : virer filtre dur confirm_ob_with_mss.
    # Le ML decide via feature has_mss_nearby (aligne avec V5 training).
    obs_confirmed = obs

    # 3. Filtre : OB des 60 dernieres minutes.
    # FIX V8 (2026-05-21) : recent_cutoff remis a 60min (etait 3min -> BUG).
    # 3min = le bot ratait tous les OB atteignant ML>=0.75 quelques minutes
    # apres validation -> 0 trade en live vs ~23/jour en backtest.
    now = df_m1.index[-1]
    recent_cutoff = now - pd.Timedelta(minutes=60)
    obs_recent = [ob for ob in obs_confirmed if df_m1.index[ob.validation_index] >= recent_cutoff]

    if debug_diag:
        from bot_v2.concepts.killzones import killzone_at
        last_ts = df_m1.index[-1]
        latest_ob_ts = "aucun"
        if obs_confirmed:
            latest_ob_ts = df_m1.index[max(ob.validation_index for ob in obs_confirmed)]
        kz_now = killzone_at(last_ts) or "AUCUNE"
        cutoff_60 = last_ts - pd.Timedelta(minutes=60)
        cutoff_15 = last_ts - pd.Timedelta(minutes=15)
        n_obs_60min = sum(1 for ob in obs_confirmed if df_m1.index[ob.validation_index] >= cutoff_60)
        n_obs_15min = sum(1 for ob in obs_confirmed if df_m1.index[ob.validation_index] >= cutoff_15)
        log.info(
            f"DIAG {instrument}: M1={len(df_m1)} M15={len(df_m15)} H1={len(df_h1)} "
            f"last_bar={last_ts} kz={kz_now} | OB_brut={len(obs)} OB+MSS={len(obs_confirmed)} "
            f"latest_OB_MSS_ts={latest_ob_ts} OB_60min={n_obs_60min} OB_15min={n_obs_15min} OB_cutoff={len(obs_recent)}"
        )

        # NOUVEAU : pour chaque OB des 15 dernieres minutes (meme ceux hors cutoff),
        # affiche son verdict pipeline+ML pour comprendre pourquoi pas tradé.
        obs_15min = [ob for ob in obs_confirmed if df_m1.index[ob.validation_index] >= cutoff_15]
        # Active pour TOUS les actifs qui ont au moins 1 OB dans les 15min
        if obs_15min:
            loaded_diag = load_model(instrument)
            if loaded_diag is not None:
                model_d, features_d = loaded_diag
                thr_d = ml_filter.get_dynamic_threshold(instrument, balance)
                for ob in obs_15min:
                    obts = df_m1.index[ob.validation_index]
                    age_min = (last_ts - obts).total_seconds() / 60
                    try:
                        r_d = evaluate_ob(
                            ob, df_m1, df_m15, df_d1, instrument,
                            ltf_name="M1", htf_name="M15",
                            df_htf2=df_h1, htf2_name="H1",
                            correlated_dfs=correlated_dfs,
                            htf_swings=htf_swings,
                            df_h1=df_h1, min_score=0, min_quality=0,
                            cache=cache,
                        )
                        if r_d.verdict == "TRADE" and r_d.trade_setup is not None:
                            proba_d = predict_proba(model_d, features_d, r_d, ob, instrument, df_ltf=df_m1, df_d1=df_d1, mss_setups=mss_setups)
                            verdict_d = f"TRADE ml={proba_d:.3f} {'OK' if proba_d >= thr_d else f'<{thr_d}'}"
                            # DUMP FEATURES si XAUUSD et ml < 0.55 -> comprendre quelles features tirent vers le bas
                            if instrument == "XAUUSD" and proba_d < 0.55:
                                feats_dump = ml_filter._features_from_result(r_d, ob, instrument, df_ltf=df_m1, df_d1=df_d1, mss_setups=mss_setups)
                                # Affiche les features importantes
                                key_feats = ["score", "quality", "ob_strength", "sweep_strength", "rr",
                                             "atr_at_setup", "atr_ratio_100",
                                             "dist_to_pdh_pct", "dist_to_pdl_pct", "dist_to_d1_open_pct",
                                             "hour_of_day", "minutes_into_killzone",
                                             "has_FVG_sync", "has_parent_ob", "has_grandparent_ob",
                                             "has_good_zone", "has_session_direction",
                                             "daily_bias_aligned", "kz_ny_am", "kz_london"]
                                log.info(f"  >>> FEATURES dump (ml={proba_d:.3f}) :")
                                for k in key_feats:
                                    log.info(f"      {k:<25} = {feats_dump.get(k, '?')}")
                        else:
                            verdict_d = f"REJET: {(r_d.rejection_reason or 'no_trade')[:40]}"
                    except Exception as e:
                        verdict_d = f"EXCEPTION: {str(e)[:30]}"
                    log.info(f"  DIAG {instrument} 15min OB | ts={obts} age={age_min:.1f}min | {verdict_d}")

    if not obs_recent:
        return []

    # 4. Pipeline Vizion + ML
    loaded = load_model(instrument)
    if loaded is None:
        return []
    model, features = loaded
    # V5 (user 2026-05-20) : seuil ML = 0.75 partout via ML_THRESHOLDS (WR 76-87%)
    threshold = ml_filter.get_dynamic_threshold(instrument, balance)

    valid_setups = []
    diag_reasons: dict[str, int] = {}
    diag_ml_probas: list[float] = []  # toutes les probas ML calculees (rejet inclus)
    for ob in obs_recent:
        try:
            r = evaluate_ob(
                ob, df_m1, df_m15, df_d1, instrument,
                ltf_name="M1", htf_name="M15",
                df_htf2=df_h1, htf2_name="H1",
                correlated_dfs=correlated_dfs,
                htf_swings=htf_swings,
                df_h1=df_h1, min_score=0, min_quality=0,
                cache=cache,
            )
        except Exception as e:
            log.debug(f"evaluate_ob fail {instrument}: {e}")
            diag_reasons["evaluate_ob_exception"] = diag_reasons.get("evaluate_ob_exception", 0) + 1
            continue

        if r.verdict != "TRADE" or r.trade_setup is None:
            reason = r.rejection_reason or "no_trade"
            state.log_rejected(instrument, df_m1.index[ob.validation_index],
                              ob.direction, reason)
            diag_reasons[reason] = diag_reasons.get(reason, 0) + 1
            continue

        proba = predict_proba(model, features, r, ob, instrument, df_ltf=df_m1, df_d1=df_d1, mss_setups=mss_setups)
        diag_ml_probas.append(proba)
        if proba < threshold:
            state.log_rejected(instrument, df_m1.index[ob.validation_index],
                              ob.direction, f"ml_below_thr_{proba:.3f}",
                              ml_proba=proba, score=r.score)
            diag_reasons[f"ml_below_{threshold:.2f}"] = diag_reasons.get(f"ml_below_{threshold:.2f}", 0) + 1
            continue

        valid_setups.append({
            "instrument": instrument,
            "ts": df_m1.index[ob.validation_index],
            "ob": ob,
            "r": r,
            "proba": proba,
            "df_m1": df_m1,
        })

    if debug_diag and (diag_reasons or diag_ml_probas):
        proba_summary = ""
        if diag_ml_probas:
            proba_summary = (
                f" | probas_ML={[f'{p:.3f}' for p in diag_ml_probas]} "
                f"(seuil={threshold:.2f})"
            )
        log.info(f"DIAG {instrument}: rejets = {dict(sorted(diag_reasons.items(), key=lambda x: -x[1]))}{proba_summary}")

    # V5.2 : log latence totale scan + breakdown
    _scan_total = _time.time() - _scan_start
    _timings["pipeline_eval"] = _scan_total - sum(_timings.values())
    _bd = " ".join([f"{k}={v*1000:.0f}ms" for k, v in _timings.items()])
    log.info(f"LATENCY {instrument}: total={_scan_total*1000:.0f}ms | {_bd}")

    return valid_setups


# ========== EXECUTION ==========

def execute_setup(mt5_exec: MT5Executor, state: LiveState, setup_dict: dict,
                  balance: float) -> bool:
    """Place l'ordre MT5 pour un setup."""
    instrument = setup_dict["instrument"]
    r = setup_dict["r"]
    ob = setup_dict["ob"]
    setup = r.trade_setup

    # === Age du setup ===
    # FIX V8 (2026-05-21) : seuil 3min -> 60min (aligne sur recent_cutoff).
    # 3min rejetait structurellement tous les bons setups : un OB ICT met
    # 20-40 min a murir (retest -> proba ML monte au seuil). Le commentaire
    # ci-dessous confirme qu'il n'y a aucun risque a placer le LIMIT meme si
    # l'OB a quelques dizaines de minutes (LIMIT au prix OB, fill seulement
    # si le prix revient toucher entry, sinon expire). 60min = coherent.
    age_setup_min = (pd.Timestamp.now(tz="UTC") - ob.validation_ts).total_seconds() / 60
    if age_setup_min > 60:
        log.warning(f"SETUP TROP VIEUX {instrument} : validation il y a {age_setup_min:.1f} min, SKIP")
        return False

    # User 2026-05-20 : aligne sur backtest = pas de check derive.
    # Le LIMIT au prix OB se fill SI ET SEULEMENT SI le prix revient toucher entry.
    # Sinon il expire dans 30 min. Pas de risque a placer le LIMIT meme si prix
    # eloigne -> identique au backtest qui place toujours.
    info = mt5_exec.symbol_info(instrument)
    if info is None:
        log.error(f"Symbol info None pour {instrument}")
        return False

    # Calcul lots selon balance (utilise les VRAIES valeurs MT5 du broker)
    risk_pct = get_risk_pct(balance)
    try:

        # Distance SL/TP en unites de prix (depuis l'OB)
        sl_distance = abs(setup.entry_price - setup.stop_loss)
        if sl_distance == 0:
            log.error(f"SL distance = 0 sur {instrument}")
            return False

        # V2 : SL/TP et entry du setup ICT inchanges (LIMIT au prix OB).
        # Pas besoin de check derive : si prix passe l'entry -> trade fill -> RR garanti.
        # Si prix touche pas l'entry en 60min -> ordre annule, on passe.
        sl_price = float(setup.stop_loss)
        tp_price = float(setup.take_profit)
        entry_price = float(setup.entry_price)

        # Risk en EUR (compte EUR)
        risk_eur = balance * risk_pct

        # === NOUVEAU (user 2026-05-18) : utilise les VRAIES valeurs MT5 du broker ===
        # info.trade_tick_value = gain en devise compte pour 1 lot et 1 tick
        # info.trade_tick_size  = taille minimum de mouvement du prix (ex: 0.01 XAU)
        tick_value_real = getattr(info, "trade_tick_value", 0)
        tick_size_real = getattr(info, "trade_tick_size", 0)

        if tick_value_real > 0 and tick_size_real > 0:
            # Calcul precis avec valeurs MT5 reelles (en devise du compte, donc EUR)
            n_ticks_sl = sl_distance / tick_size_real
            risk_per_lot = n_ticks_sl * tick_value_real
            calcul_source = "MT5_real"
        else:
            # Fallback : config hardcodee (en USD, approx EUR=USD)
            tick_value_fallback = INSTRUMENTS[instrument]["tick_value"]
            risk_per_lot = sl_distance * tick_value_fallback
            calcul_source = "config_fallback"

        if risk_per_lot == 0:
            return False
        lots_calc = risk_eur / risk_per_lot
        log.debug(f"{instrument} lots calc : sl_dist={sl_distance}, risk_per_lot={risk_per_lot:.2f}, "
                  f"lots={lots_calc:.4f} [source={calcul_source}]")
        # Arrondi au volume_step
        vol_step = info.volume_step
        # FIX 2026-05-20 : arrondi vers le BAS (pas max(volume_min, ...)) pour respecter risk vise
        # Si lots_calc < volume_min, on prend volume_min mais on flag pour check risk
        lots_rounded = round(lots_calc / vol_step) * vol_step
        lots = max(info.volume_min, lots_rounded)
        lots = round(lots, 2)

        if lots > info.volume_max:
            lots = info.volume_max

        # === SECURITE 1 : Verifie que la perte max sur SL <= 1.5 x risk vise ===
        # Recalcul avec les MEMES valeurs que le calcul des lots (coherence)
        if tick_value_real > 0 and tick_size_real > 0:
            perte_si_sl = (sl_distance / tick_size_real) * tick_value_real * lots
        else:
            perte_si_sl = sl_distance * INSTRUMENTS[instrument]["tick_value"] * lots

        # FIX 2026-05-20 (user: "0.10% et annule au lieu d'ajuster une seconde fois le lot") :
        # Si la perte est > 1.5x risk visé, tente de réduire le lot d'un cran de volume_step
        # AVANT de SKIP. Beaucoup d'actifs ont volume_min=0.1 mais step=0.01 -> on peut souvent
        # descendre a 0.05 ou 0.02 et rentrer dans le risk.
        if perte_si_sl > risk_eur * 1.5 and lots > info.volume_min:
            # Combien de lots faut-il pour respecter risk_eur exactement ?
            target_lots = risk_eur / risk_per_lot
            # Arrondi au volume_step inferieur
            target_lots = (target_lots // vol_step) * vol_step
            target_lots = max(info.volume_min, round(target_lots, 2))
            log.info(
                f"{instrument} : lot {lots} -> {target_lots} (ajuste pour respecter risk {risk_eur:.2f}€)"
            )
            lots = target_lots
            if tick_value_real > 0 and tick_size_real > 0:
                perte_si_sl = (sl_distance / tick_size_real) * tick_value_real * lots
            else:
                perte_si_sl = sl_distance * INSTRUMENTS[instrument]["tick_value"] * lots

        if perte_si_sl > risk_eur * 1.5:
            log.error(
                f"REJET {instrument} : perte SL={perte_si_sl:.2f}€ > 1.5x risk ({risk_eur*1.5:.2f}€) "
                f"meme avec lot min {info.volume_min}. Balance trop faible pour cet actif. SKIP."
            )
            state.log_event("WARN", f"Balance insuffisante {instrument} : skip")
            return False

        log.info(
            f"{instrument} : lots={lots}, perte_max_SL={perte_si_sl:.2f}€ "
            f"(risk vise={risk_eur:.2f}€, source={calcul_source})"
        )

        # === SECURITE 2 : Verifie margin disponible (max 80% balance utilisable) ===
        # FIX V8 (2026-05-21) : utilise mt5.order_calc_margin() (vraie marge avec
        # levier) au lieu de info.margin_initial * lots. Pour les Forex, MT5
        # renvoie margin_initial = taille du contrat (100000) -> le calcul donnait
        # 15000€ de marge pour 0.15 lot USDCAD -> bons trades rejetes a tort.
        # order_calc_margin tient compte du levier 1:500 -> marge reelle ~30€.
        try:
            import MetaTrader5 as _mt5
            from bot_v2.mt5_executor import to_broker_symbol
            broker_sym = to_broker_symbol(instrument)
            # NB : order_calc_margin renvoie 0 pour les types *_LIMIT chez Vantage.
            # On utilise ORDER_TYPE_BUY/SELL (marche) -> meme marge que le LIMIT.
            _otype = (_mt5.ORDER_TYPE_BUY if setup.direction == "bullish"
                      else _mt5.ORDER_TYPE_SELL)
            margin_required = _mt5.order_calc_margin(_otype, broker_sym, lots, entry_price)
            if margin_required is not None and margin_required > 0:
                if margin_required > balance * 0.8:
                    log.error(
                        f"REJET {instrument} : margin requis {margin_required:.2f}€ > 80% balance ({balance*0.8:.2f}€). SKIP."
                    )
                    return False
        except Exception as _me:
            log.debug(f"order_calc_margin {instrument} KO ({_me}) - check margin saute")

    except Exception as e:
        log.error(f"Calc lots error {instrument}: {e}")
        return False

    # DEBUG V8 (2026-05-21) : trace avant l'envoi de l'ordre
    log.info(
        f"DEBUG {instrument} : avant place_limit_order | lots={lots} "
        f"entry={entry_price} sl={sl_price} tp={tp_price}"
    )

    # V2 : Place un ordre LIMIT au prix de l'OB (= comportement backtest).
    # Si prix touche entry_price -> fill au prix exact. Si non -> expire en 60min.
    result = mt5_exec.place_limit_order(
        symbol=instrument,
        direction=setup.direction,
        volume=lots,
        entry_price=entry_price,
        sl=sl_price,
        tp=tp_price,
        expiration_minutes=30,  # = max_bars_to_fill du backtest (30 bougies M1)
        comment=f"V2-{ob.direction[0].upper()} ml={setup_dict['proba']:.2f}",
        magic=BOT_MAGIC,
    )

    if result is None:
        return False

    # Journal SQLite
    state.log_trade_opened(
        ticket=result["ticket"],
        instrument=instrument,
        direction=setup.direction,
        opened_ts=pd.Timestamp.now(tz="UTC"),
        entry=result["price"],
        sl=sl_price,
        tp=tp_price,
        rr=float(setup.rr),
        volume=result["volume"],
        score=int(r.score),
        ml_proba=float(setup_dict["proba"]),
        killzone=r.killzone_name,
        comment=f"risk_pct={risk_pct:.0%}",
    )

    # Update cooldown
    state.set_last_trade_ts(instrument, pd.Timestamp.now(tz="UTC"))

    # V5.8 : push TRADE_EXECUTED vers le dashboard (Telegram notif)
    if _PUSHER is not None:
        try:
            _PUSHER.push_trade_executed(
                instrument=instrument,
                ticket=int(result["ticket"]),
                direction=setup.direction,
                entry=float(result["price"]),
                sl=float(sl_price),
                tp=float(tp_price),
                volume=float(result["volume"]),
                rr=float(setup.rr),
                ml_proba=float(setup_dict["proba"]),
                score=int(r.score),
                killzone=r.killzone_name,
            )
        except Exception as _pe:
            log.debug(f"push_trade_executed fail: {_pe}")

    log.info(
        f"PENDING ORDER {instrument} {setup.direction} vol={lots} "
        f"entry={entry_price:.5f} SL={sl_price:.5f} TP={tp_price:.5f} "
        f"RR={setup.rr:.2f} ML={setup_dict['proba']:.3f} risk={risk_pct:.0%} "
        f"balance={balance:.2f}€ (expire 60min)"
    )
    return True


# ========== RECONCILIATION ==========

def reconcile_closed_trades(mt5_exec: MT5Executor, state: LiveState):
    """Met a jour le journal pour les trades fermes par le broker (SL/TP touche)."""
    pending = state.get_pending_tickets()
    if not pending:
        return

    # FIX 2026-05-20 : un LIMIT pending n'est PAS une position ouverte !
    # Sans ce check, le bot marquait tous les LIMIT en ORPHAN 10s apres creation.
    open_positions = {p["ticket"] for p in mt5_exec.get_positions()}
    pending_mt5 = {po["ticket"] for po in mt5_exec.get_pending_orders(magic=BOT_MAGIC)}
    # Un ticket est ferme s'il n'est NI dans positions NI dans pending orders MT5
    closed_tickets = [t for t in pending if t not in open_positions and t not in pending_mt5]

    if not closed_tickets:
        return

    # Recupere l'historique des deals des dernieres 48h
    from_ts = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=48)
    to_ts = pd.Timestamp.now(tz="UTC")
    deals = mt5_exec.get_closed_deals(from_ts, to_ts, magic=BOT_MAGIC)

    # Index par position_id ET par order (Vantage utilise parfois "order" pas "position_id")
    deals_by_pos = {}
    deals_by_order = {}
    for d in deals:
        deals_by_pos.setdefault(d["position_id"], []).append(d)
        deals_by_order.setdefault(d.get("order", 0), []).append(d)

    for ticket in closed_tickets:
        # Cherche d'abord par position_id, puis par order ID en fallback
        related = deals_by_pos.get(ticket, []) or deals_by_order.get(ticket, [])
        if not related:
            # Apres 5 min sans trouver, on marque le trade comme ferme par defaut (orphan)
            # pour eviter la boucle de warnings infinie
            log.warning(f"Pas de deals trouves pour ticket {ticket}, mark as orphan")
            state.update_trade_closed(
                ticket=ticket,
                closed_ts=pd.Timestamp.now(tz="UTC"),
                outcome="ORPHAN",
                pnl_real=0.0,
            )
            continue

        # PnL net = somme des profits des deals lies a la position
        pnl = sum(d["profit"] for d in related)
        # Outcome : WIN si pnl > 0, LOSS sinon
        outcome = "WIN" if pnl > 0 else "LOSS"
        last_deal_time = max(d["time"] for d in related)

        state.update_trade_closed(
            ticket=ticket,
            closed_ts=last_deal_time,
            outcome=outcome,
            pnl_real=float(pnl),
        )
        log.info(f"TRADE CLOSED ticket={ticket} {outcome} pnl={pnl:+.2f}€")

        # V5.8 : push TRADE_CLOSED vers le dashboard (Telegram notif)
        if _PUSHER is not None:
            try:
                # Duree approximative : depuis le 1er deal (open) jusqu'au dernier (close)
                _first_deal_time = min(d["time"] for d in related)
                _dur_min = (last_deal_time - _first_deal_time).total_seconds() / 60.0
                _PUSHER.push_trade_closed(
                    ticket=int(ticket),
                    outcome=outcome,
                    pnl_real=float(pnl),
                    duration_min=_dur_min,
                )
            except Exception as _pe:
                log.debug(f"push_trade_closed fail: {_pe}")


# ========== DIAGNOSTICS AU DEMARRAGE ==========

def boot_diagnostics(mt5_exec: MT5Executor, state: LiveState):
    """Audit complet au demarrage : fetch + OB detection + symboles + killzone.

    But : reveler tout probleme structurel AVANT d'attendre des trades.
    User 2026-05-20 : "fait des debug pour trouver tout au lieu d'attendre".
    """
    from bot_v2.concepts.killzones import killzone_at, to_ny_time

    log.info("=" * 70)
    log.info("BOOT DIAGNOSTICS - audit de tous les actifs")
    log.info("=" * 70)

    # === 1. Heure broker vs heure UTC reelle ===
    utc_now = pd.Timestamp.now(tz="UTC")
    # Recupere la derniere bougie M1 EURUSD pour deduire le broker time
    df_test = mt5_exec.get_bars("EURUSD", "M1", 5)
    if df_test is not None and len(df_test) > 0:
        broker_last = df_test.index[-1]
        offset_min = (broker_last - utc_now).total_seconds() / 60
        log.info(
            f"TIME | UTC reel = {utc_now.strftime('%Y-%m-%d %H:%M:%S')} | "
            f"Broker last M1 = {broker_last.strftime('%Y-%m-%d %H:%M:%S')} | "
            f"offset = {offset_min:+.1f} min"
        )
        if abs(offset_min) > 30:
            log.warning(
                f"!! BROKER TIME != UTC ({offset_min:+.1f} min). "
                f"Les bougies sont en time broker traite comme UTC. "
                f"Killzones peuvent etre decalees."
            )
    else:
        log.error("TIME | impossible de fetch EURUSD M1 pour deduire broker time")

    # === 2. Killzone actuelle ===
    kz = killzone_at(utc_now)
    ny_time = to_ny_time(utc_now).strftime('%H:%M')
    log.info(f"KILLZONE | UTC={utc_now.strftime('%H:%M')} NY={ny_time} -> {kz or 'AUCUNE'}")

    # === 3. Audit par actif : symbole + bougies + OB ===
    log.info("-" * 70)
    log.info(f"{'ACTIF':<10}{'SYMBOL_BROKER':<18}{'TICK':<10}{'M1':<8}{'M15':<8}{'H1':<8}{'OB_brut':<10}{'OB+MSS':<10}")
    log.info("-" * 70)

    from bot_v2.mt5_executor import to_broker_symbol
    from bot_v2.concepts.order_block import detect_order_blocks
    from bot_v2.concepts.mss_setup import detect_mss_setups, confirm_ob_with_mss
    from bot_v2.concepts.liquidity import find_swings
    from bot_v2.concepts.structure import detect_structure_breaks

    issues = []
    for asset in LIVE_ASSETS:
        broker_sym = to_broker_symbol(asset)
        # Tick
        tick = mt5_exec.get_tick(asset)
        tick_ok = "OK" if tick and tick.bid > 0 else "KO"
        # Bougies
        df_m1 = mt5_exec.get_bars(asset, "M1", 200)
        df_m15 = mt5_exec.get_bars(asset, "M15", 200)
        df_h1 = mt5_exec.get_bars(asset, "H1", 200)
        n_m1 = len(df_m1) if df_m1 is not None else 0
        n_m15 = len(df_m15) if df_m15 is not None else 0
        n_h1 = len(df_h1) if df_h1 is not None else 0

        n_ob = 0
        n_ob_mss = 0
        if df_m1 is not None and len(df_m1) >= 100:
            try:
                sws = get_param(asset, "swing_strength_m1", 2)
                obs = detect_order_blocks(df_m1, swing_strength=sws)  # max_group_size defaut=5, ALIGNE OOS
                n_ob = len(obs)
                swings = find_swings(df_m1, strength=sws)
                sb = detect_structure_breaks(df_m1, swings=swings)
                mss = detect_mss_setups(df_m1, structure_breaks=sb, swings=swings)
                obs_conf = confirm_ob_with_mss(obs, mss, window_bars=10)
                n_ob_mss = len(obs_conf)
            except Exception as e:
                log.debug(f"DIAG {asset} OB detection failed: {e}")

        log.info(
            f"{asset:<10}{broker_sym:<18}{tick_ok:<10}{n_m1:<8}{n_m15:<8}{n_h1:<8}{n_ob:<10}{n_ob_mss:<10}"
        )

        # Flag les anomalies
        if tick_ok == "KO":
            issues.append(f"{asset}: TICK KO - symbole non disponible chez broker")
        if n_m1 < 100:
            issues.append(f"{asset}: M1 insuffisant ({n_m1} bougies)")
        if n_ob == 0 and tick_ok == "OK":
            issues.append(f"{asset}: AUCUN OB detecte (verifier swing_strength ou data)")

    log.info("-" * 70)

    # === 4. Resume des anomalies ===
    if issues:
        log.warning(f"!! {len(issues)} ANOMALIES DETECTEES :")
        for iss in issues:
            log.warning(f"   - {iss}")
    else:
        log.info("OK : aucune anomalie detectee - tous les actifs sont fetchables")

    # === 5. Etat MT5 pending/positions ===
    positions = mt5_exec.get_positions()
    pendings = mt5_exec.get_pending_orders(magic=BOT_MAGIC)
    log.info(f"MT5 | positions ouvertes = {len(positions)} | pending orders bot = {len(pendings)}")
    for p in positions:
        log.info(f"   POS {p.get('symbol')} ticket={p.get('ticket')} vol={p.get('volume')}")
    for po in pendings:
        age = (utc_now - po["time_setup"]).total_seconds() / 60
        log.info(f"   PEND {po['symbol']} ticket={po['ticket']} age={age:.1f}min")

    # === 6 + 7. DIAGNOSTICS BOOT LOURDS — DESACTIVES (2026-05-21) ===
    # simulate_last_24h() et debug_last_hour() rejouent le pipeline ICT sur
    # 17 actifs x 88k bougies au demarrage. Sur le VPS Contabo (peu de cores,
    # RAM limitee) ca saturait la RAM et crashait le bot avant le 1er cycle.
    # Ces diagnostics sont purement informatifs -> on demarre direct sur le
    # scan live. Pour analyser les 24h passees, lancer backtest_last_24h_local.py
    # sur le PC local (machine plus puissante).
    log.info("=" * 70)
    log.info("BOOT : diagnostics retrospectifs desactives (VPS) -> scan live direct")
    log.info("=" * 70)


def simulate_last_24h(mt5_exec: MT5Executor):
    """Rejoue le pipeline sur les 24 dernieres heures pour chaque actif.
    Compte les setups qui auraient passe ML+pipeline -> trade vrai.
    """
    from bot_v2.concepts.killzones import killzone_at

    utc_now = pd.Timestamp.now(tz="UTC")
    cutoff_24h = utc_now - pd.Timedelta(hours=24)

    total_setups = 0
    total_passed = 0
    all_trades = []

    for asset in LIVE_ASSETS:
        # Fetch 1500 bougies M1 = 25h (couvre les 24h)
        df_m1 = mt5_exec.get_bars(asset, "M1", 1500)
        if df_m1 is None or len(df_m1) < 200:
            continue
        df_m1 = df_m1.iloc[:-1]  # exclure bougie en cours

        df_m15 = mt5_exec.get_bars(asset, "M15", 500)
        df_h1 = mt5_exec.get_bars(asset, "H1", 500)
        if df_m15 is None or df_h1 is None:
            continue
        df_m15 = df_m15.iloc[:-1]
        df_h1 = df_h1.iloc[:-1]

        try:
            df_d1 = mt5_exec.get_bars(asset, "D1", 100)
            if df_d1 is None or len(df_d1) < 10:
                df_d1 = build_d1_from_h1(df_h1)
        except Exception:
            df_d1 = build_d1_from_h1(df_h1)

        # HTF + SMT
        htf_dfs = {"H1": df_h1, "D1": df_d1}
        try:
            df_h4 = mt5_exec.get_bars(asset, "H4", 500)
            if df_h4 is not None:
                htf_dfs["H4"] = df_h4
        except Exception:
            pass
        htf_swings = collect_htf_swings(htf_dfs, swing_strength=3)

        correlated_dfs = {}
        for corr_name, corr_type in SMT_PAIRS.get(asset, []):
            try:
                df_c = mt5_exec.get_bars(corr_name, "M1", 1500)
                if df_c is not None and len(df_c) > 0:
                    correlated_dfs[corr_name] = (df_c, corr_type)
            except Exception:
                continue

        # Detection
        sws = get_param(asset, "swing_strength_m1", 2)
        from bot_v2.concepts.order_block import detect_order_blocks
        from bot_v2.concepts.mss_setup import detect_mss_setups, confirm_ob_with_mss
        from bot_v2.concepts.liquidity import find_swings
        from bot_v2.concepts.structure import detect_structure_breaks, detect_trend
        from bot_v2.concepts.fvg import detect_fvg
        from bot_v2.concepts.breaker import detect_breakers

        obs = detect_order_blocks(df_m1, swing_strength=sws)  # max_group_size defaut=5, ALIGNE OOS
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
        # V5.1 (2026-05-20) : virer confirm_ob_with_mss (ML decide via has_mss_nearby)
        obs_confirmed = obs

        # Filtre 24h
        obs_24h = [ob for ob in obs_confirmed if df_m1.index[ob.validation_index] >= cutoff_24h]

        # Pipeline ML
        loaded = load_model(asset)
        if loaded is None:
            continue
        model, features = loaded
        threshold = ml_filter.get_dynamic_threshold(asset, 150.0)

        n_setups = len(obs_24h)
        n_passed = 0
        rejets_24h = {}
        probas_passed = []

        for ob in obs_24h:
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
            except Exception:
                rejets_24h["evaluate_exception"] = rejets_24h.get("evaluate_exception", 0) + 1
                continue

            if r.verdict != "TRADE" or r.trade_setup is None:
                reason = r.rejection_reason or "no_trade"
                # Tronque les raisons longues pour la lisibilite
                if "displacement" in reason.lower():
                    reason = "displacement_faible"
                elif "killzone" in reason.lower():
                    reason = "killzone_hors"
                rejets_24h[reason] = rejets_24h.get(reason, 0) + 1
                continue

            proba = predict_proba(model, features, r, ob, asset, df_ltf=df_m1, df_d1=df_d1, mss_setups=mss_setups)
            if proba < threshold:
                rejets_24h[f"ml_below_{threshold:.2f}"] = rejets_24h.get(f"ml_below_{threshold:.2f}", 0) + 1
                continue

            n_passed += 1
            probas_passed.append(proba)
            kz = killzone_at(df_m1.index[ob.validation_index]) or "?"
            # Sauvegarde aussi les features pour comparaison
            feats_for_diff = ml_filter._features_from_result(r, ob, asset, df_ltf=df_m1, df_d1=df_d1, mss_setups=mss_setups)
            all_trades.append({
                "asset": asset,
                "ts": df_m1.index[ob.validation_index],
                "direction": ob.direction,
                "proba": proba,
                "killzone": kz,
                "features": feats_for_diff if asset == "XAUUSD" else None,
            })

        total_setups += n_setups
        total_passed += n_passed
        if n_setups > 0:
            log.info(
                f"{asset:<10} setups_24h={n_setups:>3} passes={n_passed:>2} "
                f"rejets={dict(sorted(rejets_24h.items(), key=lambda x: -x[1])[:3])}"
            )

    log.info("-" * 70)
    log.info(f"TOTAL : {total_setups} setups detectes en 24h -> {total_passed} auraient ete trades")

    if all_trades:
        all_trades.sort(key=lambda t: t["ts"])
        log.info("")
        log.info("Liste des trades (chronologique) :")
        for t in all_trades:
            ts = t["ts"].strftime("%Y-%m-%d %H:%M")
            log.info(
                f"   {ts} | {t['asset']:<8} | {t['direction']:<8} | "
                f"kz={t['killzone']:<12} | ML={t['proba']:.3f}"
            )

        # DUMP features pour trades XAUUSD passes - permet comparaison avec OB rejetes live
        xauusd_trades = [t for t in all_trades if t["asset"] == "XAUUSD" and t.get("features")]
        if xauusd_trades:
            log.info("")
            log.info("=== FEATURES dump XAUUSD trades PASSES (pour comparaison live) ===")
            key_feats = ["score", "quality", "ob_strength", "sweep_strength", "rr",
                         "atr_at_setup", "atr_ratio_100",
                         "dist_to_pdh_pct", "dist_to_pdl_pct", "dist_to_d1_open_pct",
                         "hour_of_day", "minutes_into_killzone",
                         "has_FVG_sync", "has_parent_ob", "has_grandparent_ob",
                         "has_good_zone", "has_session_direction",
                         "daily_bias_aligned", "kz_ny_am", "kz_london",
                         "has_smt", "has_feu_vert", "has_breaker_kz", "has_mss_fvg"]
            for t in xauusd_trades:
                log.info(f"  TRADE {t['ts'].strftime('%Y-%m-%d %H:%M')} ml={t['proba']:.3f}:")
                for k in key_feats:
                    log.info(f"    {k:<25} = {t['features'].get(k, '?')}")


def debug_last_hour(mt5_exec: MT5Executor):
    """DEBUG ULTRA DETAILLE de la derniere heure pour chaque actif.

    Pour CHAQUE OB+MSS de la derniere heure, affiche :
    - Timestamp validation, killzone, age en min
    - Verdict pipeline (TRADE / REJECTED + raison)
    - Si TRADE : proba ML + serait-il pris par le live ?
    - Pourquoi le live l'aurait skip (recent_cutoff, BOT_START_TS, etc.)
    - TEST CUTOFF : pour chaque cutoff (1, 2, 3, 5, 10 min), serait-il pris ?
    """
    from bot_v2.concepts.killzones import killzone_at
    from bot_v2.concepts.order_block import detect_order_blocks
    from bot_v2.concepts.mss_setup import detect_mss_setups, confirm_ob_with_mss
    from bot_v2.concepts.liquidity import find_swings
    from bot_v2.concepts.structure import detect_structure_breaks, detect_trend
    from bot_v2.concepts.fvg import detect_fvg
    from bot_v2.concepts.breaker import detect_breakers

    log.info("=" * 90)
    log.info("DEBUG DERNIERE HEURE - chaque OB analyse en detail")
    log.info("=" * 90)

    utc_now = pd.Timestamp.now(tz="UTC")
    # On scanne sur 24h pour avoir un echantillon large (sinon dead zone London-NY = 0 OB)
    cutoff_1h = utc_now - pd.Timedelta(hours=24)

    grand_total_obs = 0
    grand_total_would_trade = 0
    valid_setups_all = []
    cutoff_tests = [1, 2, 3, 5, 10, 15]

    for asset in LIVE_ASSETS:
        df_m1 = mt5_exec.get_bars(asset, "M1", 1500, force_sync=True)
        if df_m1 is None or len(df_m1) < 200:
            continue
        df_m1 = df_m1.iloc[:-1]

        df_m15 = mt5_exec.get_bars(asset, "M15", 500)
        df_h1 = mt5_exec.get_bars(asset, "H1", 500)
        if df_m15 is None or df_h1 is None:
            continue
        df_m15 = df_m15.iloc[:-1]
        df_h1 = df_h1.iloc[:-1]

        try:
            df_d1 = mt5_exec.get_bars(asset, "D1", 100)
            if df_d1 is None or len(df_d1) < 10:
                df_d1 = build_d1_from_h1(df_h1)
        except Exception:
            df_d1 = build_d1_from_h1(df_h1)

        htf_dfs = {"H1": df_h1, "D1": df_d1}
        try:
            df_h4 = mt5_exec.get_bars(asset, "H4", 500)
            if df_h4 is not None:
                htf_dfs["H4"] = df_h4
        except Exception:
            pass
        htf_swings = collect_htf_swings(htf_dfs, swing_strength=3)

        correlated_dfs = {}
        for corr_name, corr_type in SMT_PAIRS.get(asset, []):
            try:
                df_c = mt5_exec.get_bars(corr_name, "M1", 1500)
                if df_c is not None and len(df_c) > 0:
                    correlated_dfs[corr_name] = (df_c, corr_type)
            except Exception:
                continue

        sws = get_param(asset, "swing_strength_m1", 2)
        obs = detect_order_blocks(df_m1, swing_strength=sws)  # max_group_size defaut=5, ALIGNE OOS
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
        # V5.1 (2026-05-20) : virer confirm_ob_with_mss (ML decide via has_mss_nearby)
        obs_confirmed = obs

        # Filtre 24h
        obs_24h_list = [ob for ob in obs_confirmed if df_m1.index[ob.validation_index] >= cutoff_1h]
        if not obs_24h_list:
            continue

        loaded = load_model(asset)
        if loaded is None:
            continue
        model, features = loaded
        threshold = ml_filter.get_dynamic_threshold(asset, 150.0)

        # SIMULATION : pour chaque OB+MSS qui passe ML, simule different cutoffs.
        # Dans la realite live, l'OB est detecte X min apres sa validation_ts
        # a cause de la latence broker + MT5. On suppose que le bot scanne dans la
        # minute qui suit la fermeture de la bougie de validation_ts.
        # Donc : si recent_cutoff >= 1 min, l'OB est TOUJOURS pris (puisque le scan
        # se passe juste apres la fermeture). Le cutoff sert a tolerer une plus grande
        # latence MT5 ou un bot qui a "rate" un cycle de scan.
        for ob in obs_24h_list:
            ts = df_m1.index[ob.validation_index]
            grand_total_obs += 1

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
            except Exception:
                continue

            if r.verdict != "TRADE":
                continue

            proba = predict_proba(model, features, r, ob, asset, df_ltf=df_m1, df_d1=df_d1, mss_setups=mss_setups)
            if proba < threshold:
                continue

            # Setup VALIDE (pipeline + ML OK).
            grand_total_would_trade += 1
            valid_setups_all.append({"asset": asset, "ts": ts, "proba": proba})

    log.info("")
    log.info("-" * 100)
    log.info(f"TOTAL 24h : {grand_total_obs} OB analyses -> {grand_total_would_trade} passent pipeline+ML")
    log.info("")
    log.info("TEST RECENT_CUTOFF x LATENCE BROKER")
    log.info("Combien de trades on prend selon (cutoff, latence MT5) :")
    log.info(f"")
    log.info(f"{'CUTOFF':<12}{'lag=0min':<12}{'lag=1min':<12}{'lag=2min':<12}{'lag=3min':<12}{'lag=5min':<12}")
    log.info("-" * 70)
    # Pour chaque (cutoff, lag), nb de trades pris :
    # Live : a chaque scan, on a last_bar = realt_now - lag
    # Le bot voit l'OB quand last_bar >= validation_ts -> donc scan a ts+lag
    # Filtre : validation_ts >= last_bar - cutoff = (ts+lag) - cutoff
    # OK si: ts >= ts + lag - cutoff <=> cutoff >= lag
    for c in cutoff_tests:
        row = f"{c} min".ljust(12)
        for lag in [0, 1, 2, 3, 5]:
            pris = grand_total_would_trade if c >= lag else 0
            row += f"{pris}/{grand_total_would_trade}".ljust(12)
        log.info(row)
    log.info("=" * 90)
    log.info("CONCLUSION : cutoff = max(latence MT5 observee)")
    log.info("Si MT5 propage les bougies M1 en <1 min : cutoff=1 suffit")
    log.info("Si MT5 lag occasionnellement 2-3 min : cutoff=3 recommande")
    log.info("=" * 90)


# ========== MAIN LOOP ==========

def run_live(test_dry_run: bool = False):
    """Boucle principale du bot live.

    Args:
        test_dry_run: si True, ne place PAS d'ordres (juste detecte et log).
    """
    # Log path relatif au repertoire du script (compat PC dev + VPS prod)
    log_path = Path(__file__).resolve().parent.parent / "live.log"

    # V7.1 (2026-05-21) : logging optimise pour live.
    # - Fichier live.log : tout en INFO (debug complet conserve)
    # - Console : uniquement les messages IMPORTANTS (filtre custom) +
    #   tout ce qui est WARNING/ERROR. Reduit RAM/IO et garde le terminal
    #   lisible.
    #
    # Pour voir TOUS les logs en temps reel -> python tools/view_logs.py
    log.setLevel(logging.INFO)

    # File handler : full INFO
    fh = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(fh)

    # Console handler : WARNING+ par defaut, MAIS on laisse passer les
    # lignes importantes (CYCLE scan, SETUP, TRADE, balance, STATS) en INFO.
    class _ImportantOnlyFilter(logging.Filter):
        IMPORTANT_PREFIXES = (
            "BOT LIVE DEMARRAGE", "MT5 connecte", "Cash reel", "ProcessPool",
            "Buffers initialises", "DashboardPusher",
            "CYCLE scan", "SETUP ", "PENDING ORDER", "TRADE CLOSED",
            "STATS |", "Bot arrete", "Modele",
        )
        def filter(self, record: logging.LogRecord) -> bool:
            if record.levelno >= logging.WARNING:
                return True
            msg = record.getMessage()
            return any(msg.startswith(p) for p in self.IMPORTANT_PREFIXES)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.addFilter(_ImportantOnlyFilter())
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(ch)
    log.info("=" * 60)
    log.info("BOT LIVE DEMARRAGE")
    log.info("=" * 60)

    state = LiveState()
    state.log_event("START", f"Bot live demarre (dry_run={test_dry_run})")

    # V5.8 (2026-05-21) : dashboard pusher (Railway).
    # DASHBOARD_URL vide -> no-op (dev local).
    import uuid
    global _PUSHER
    _session_id = uuid.uuid4().hex[:8]
    _PUSHER = DashboardPusher(
        url=os.getenv("DASHBOARD_URL"),
        session_id=_session_id,
    )
    PUSHER = _PUSHER  # alias local pour le reste de run_live
    PUSHER.push_start(f"Bot demarre (dry_run={test_dry_run})")

    mt5_exec = MT5Executor()
    if not mt5_exec.initialize():
        log.error("MT5 init FAILED - bot ne demarre pas")
        state.log_event("ERROR", "MT5 init failed")
        PUSHER.push_error("MT5 init failed")
        PUSHER.shutdown()
        return

    log.info(f"Connecte au compte {mt5_exec.account_info.login} sur {mt5_exec.account_info.server}")
    cash_real = mt5_exec.get_balance()
    log.info(f"Cash reel : {cash_real:.2f} {mt5_exec.account_info.currency} + bonus 50€ = base calcul {cash_real + 50:.2f}€")
    if test_dry_run:
        log.info("** MODE DRY RUN - aucun ordre ne sera place **")

    # FIX V8 (2026-05-21) : BOT_START_TS aligne sur recent_cutoff (60min).
    # Avant : now - 1min -> au redemarrage, le bot ignorait pendant 1h tous les
    # OB que recent_cutoff lui montrait (ceux valides avant le boot) -> un OB
    # mur a ML 0.759 detecte comme SETUP mais jamais execute.
    # Maintenant : now - 60min -> le bot peut trader tout OB de la derniere
    # heure des le boot (coherent avec recent_cutoff). _seen_setups + le cap
    # recent_cutoff empechent de trader du vraiment vieux.
    BOT_START_TS = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=60)
    log.info(f"BOT_START_TS = {BOT_START_TS} (setups anterieurs a -60min ignores)")

    # V5.4 : init DataBuffer pour chaque actif (charge parquet 7mois + comble trou via MT5)
    log.info("=" * 70)
    log.info("INIT DATA BUFFERS (charge 7mois historique + comble trou via MT5)")
    log.info("=" * 70)
    _buffer_assets = list(LIVE_ASSETS)
    # Ajoute les correles SMT (XAGUSD, DXY, SPX500) si presents
    for asset in LIVE_ASSETS:
        for corr_name, _ in SMT_PAIRS.get(asset, []):
            if corr_name not in _buffer_assets:
                _buffer_assets.append(corr_name)
    for asset in _buffer_assets:
        try:
            buf = DataBuffer(asset=asset, mt5_exec=mt5_exec)
            if buf.load_and_fill():
                DATA_BUFFERS[asset] = buf
                log.info(f"  BUFFER {asset:<8} : OK ({len(buf.df_m1):,} M1)")
            else:
                log.warning(f"  BUFFER {asset:<8} : KO (fallback fetch MT5)")
        except Exception as e:
            log.error(f"  BUFFER {asset:<8} : exception {e}")
    log.info(f"Buffers initialises : {len(DATA_BUFFERS)} actifs")
    log.info("=" * 70)

    # === V5.7 (2026-05-21) : ProcessPool 4 workers pour scan parallele ===
    # Cree UNE fois (workers persistents) -> _models_cache reste chaud entre
    # cycles. Ferme dans le finally a la fin.
    from concurrent.futures import ProcessPoolExecutor
    POOL = ProcessPoolExecutor(max_workers=4)
    log.info("ProcessPool : 4 workers persistents crees")
    log.info("=" * 70)

    # === BOOT DIAGNOSTICS : audit complet avant de demarrer la boucle ===
    try:
        boot_diagnostics(mt5_exec, state)
    except Exception as e:
        log.exception(f"Boot diagnostics failed (continue quand meme) : {e}")

    # Force un DIAG au 1er scan pour ne pas attendre 5 min
    _force_first_diag = True

    try:
        while True:
            try:
                # User 2026-05-19 : bonus Vantage de 50€ ajoute au calcul du risk
                # ATTENTION : le bonus n'est pas du cash retirable, c'est juste pour gonfler
                # le capital de calcul. Si LOSS, c'est le cash reel qui est entame.
                BONUS_EUR = 50.0
                balance = mt5_exec.get_balance() + BONUS_EUR
                # V5 : seuil ML unifie 0.75 partout
                _ml_thr = "0.75 (V5)"
                active_assets = get_active_assets(balance)

                # Cleanup pending orders > 30 min (= max_bars_to_fill backtest)
                # Vantage ne supporte pas ORDER_TIME_SPECIFIED -> on cleanup manuellement
                from datetime import timezone as _tz
                _now = pd.Timestamp.now(tz=_tz.utc)
                _pending_all = mt5_exec.get_pending_orders(magic=BOT_MAGIC)
                for _po in _pending_all:
                    age_min = (_now - _po["time_setup"]).total_seconds() / 60
                    if age_min > 30:
                        if mt5_exec.cancel_pending_order(_po["ticket"]):
                            log.info(f"Cleanup pending vieux {_po['symbol']} ticket={_po['ticket']} age={age_min:.0f}min")

                n_open = mt5_exec.get_n_open_positions()
                # V2 : on compte aussi les pending orders (ils peuvent devenir des positions)
                n_pending = len(mt5_exec.get_pending_orders(magic=BOT_MAGIC))
                n_total = n_open + n_pending

                # Reconciliation des trades fermes
                reconcile_closed_trades(mt5_exec, state)

                # Si max concurrent atteint (open + pending), on attend
                if n_total >= MAX_CONCURRENT:
                    log.debug(f"Max {MAX_CONCURRENT} concurrents atteints "
                              f"({n_open} ouverts + {n_pending} pending), wait")
                    time.sleep(SCAN_INTERVAL_SEC)
                    continue

                # Scan tous les actifs actifs
                now = pd.Timestamp.now(tz="UTC")
                # FIX 2026-05-20 : dedup setups par validation_ts pour eviter reprises
                if not hasattr(state, "_seen_setups"):
                    state._seen_setups = {}
                # ALIGNEMENT OOS : 1 OB = 1 evaluation. {asset: set(ob_ts_str)}.
                if not hasattr(state, "_evaluated_obs"):
                    state._evaluated_obs = {}

                # DIAG : log diagnostic complet chaque ~5 min + 1er scan force
                _diag_now = _force_first_diag or (int(time.time()) % 300 < SCAN_INTERVAL_SEC)
                if _force_first_diag:
                    _force_first_diag = False
                    log.info(">>> DIAG FORCE (1er scan apres demarrage) <<<")

                # V5.7 (2026-05-21) : split fetch (main) / compute (ProcessPool 4).
                # Avant : scan sequentiel ~75s/cycle -> on rate ~30% des pullbacks
                # rapides (mesure empirique : 55% des OB sont retestes en <1min).
                # Maintenant : fetch ~4s + compute parallele ~5s -> cycle ~10-20s.
                _assets_to_scan = [a for a in active_assets if not state.is_in_cooldown(a, now, COOLDOWN_SEC)]

                _setups_by_asset: dict[str, list] = {}
                _cycle_t0 = time.time()

                # Phase 1 : fetch sequentiel (MT5 + DataBuffer dans le main)
                _t_fetch = time.time()
                payloads = []
                for _a in _assets_to_scan:
                    try:
                        _eval_keys = state._evaluated_obs.get(_a, set())
                        p = fetch_payload(mt5_exec, _a, balance, _diag_now,
                                          evaluated_keys=_eval_keys)
                        if p is not None:
                            payloads.append(p)
                        else:
                            _setups_by_asset[_a] = []
                    except Exception as _e:
                        log.error(f"fetch_payload {_a} exception : {_e}")
                        _setups_by_asset[_a] = []
                _fetch_s = time.time() - _t_fetch

                # Phase 2 : compute parallele (workers persistents)
                _t_compute = time.time()
                try:
                    results = list(POOL.map(compute_asset, payloads))
                except Exception as _e:
                    log.exception(f"POOL.map exception : {_e}")
                    results = []
                _compute_s = time.time() - _t_compute

                # Phase 3 : replay logs + rejets + collecte setups
                _cycle_latencies: dict[str, int] = {}
                _cycle_setups_pushed = 0
                _cycle_rejected_batch: list[dict] = []
                for r in results:
                    inst = r["instrument"]
                    if r.get("error"):
                        log.error(f"compute_asset {inst} exception : {r['error'][:200]}")
                        _setups_by_asset[inst] = []
                        continue
                    for line in r["diag_log"]:
                        log.info(line)
                    for entry in r["rejected_log"]:
                        # entry = (instrument, ts, direction, reason, ml_proba, score, extras_dict?)
                        # SQLite log_rejected n'accepte que les 6 premiers.
                        try:
                            state.log_rejected(*entry[:6])
                        except Exception:
                            pass
                        # Batch pour le dashboard avec extras (entry/SL/TP/RR/threshold)
                        try:
                            _extras = entry[6] if len(entry) > 6 and isinstance(entry[6], dict) else {}
                            _cycle_rejected_batch.append({
                                "instrument": entry[0],
                                "ts": str(entry[1]),
                                "direction": entry[2],
                                "reason": entry[3],
                                "ml_proba": entry[4] if len(entry) > 4 else None,
                                "score": entry[5] if len(entry) > 5 else None,
                                **_extras,  # threshold, entry, sl, tp, rr, ob_high, ob_low
                            })
                        except Exception:
                            pass
                    _setups_by_asset[inst] = r["setups"]

                    # ALIGNEMENT OOS : marque TOUS les OB evalues ce cycle
                    # (rejets + setups) -> ils ne seront plus re-evalues.
                    # 1 OB = 1 evaluation, exactement comme le backtest.
                    _ev = state._evaluated_obs.setdefault(inst, set())
                    for entry in r["rejected_log"]:
                        _ev.add(str(entry[1]))  # entry[1] = ts de l'OB
                    for s in r["setups"]:
                        _ev.add(str(s["ts"]))

                    # Latence pour le push CYCLE
                    if r.get("latency_ms"):
                        _cycle_latencies[inst] = r["latency_ms"]
                    # Push SETUP par setup ML-OK
                    for s in r["setups"]:
                        try:
                            PUSHER.push_setup(
                                instrument=inst,
                                ts=s["ts"],
                                direction=s["ob"].direction,
                                entry_price=s["r"].trade_setup.entry_price,
                                sl=s["r"].trade_setup.stop_loss,
                                tp=s["r"].trade_setup.take_profit,
                                rr=s["r"].trade_setup.rr,
                                score=s["r"].score,
                                ml_proba=s["proba"],
                                killzone=None,
                            )
                            _cycle_setups_pushed += 1
                        except Exception as _pe:
                            log.debug(f"push_setup fail {inst}: {_pe}")

                _cycle_total_s = time.time() - _cycle_t0
                log.info(
                    f"CYCLE scan {len(_assets_to_scan)} actifs en "
                    f"{_cycle_total_s:.1f}s (fetch={_fetch_s:.1f}s "
                    f"compute={_compute_s:.1f}s)"
                )
                # V5.8 : push CYCLE + rejets vers le dashboard (non-bloquant)
                # V5.9 : ajoute last_bar_ts pour afficher la derniere bougie M1
                # cote dashboard (on prend XAUUSD comme reference)
                try:
                    _last_bar_ts = None
                    try:
                        _ref_buf = DATA_BUFFERS.get("XAUUSD")
                        if _ref_buf is not None and _ref_buf.df_m1 is not None and len(_ref_buf.df_m1) > 0:
                            _last_bar_ts = _ref_buf.df_m1.index[-1]
                    except Exception:
                        pass
                    PUSHER.push_cycle(
                        actifs_scanned=len(_assets_to_scan),
                        total_s=_cycle_total_s,
                        fetch_s=_fetch_s,
                        compute_s=_compute_s,
                        latencies=_cycle_latencies,
                        last_bar_ts=_last_bar_ts,
                    )
                    if _cycle_rejected_batch:
                        PUSHER.push_rejected_batch(_cycle_rejected_batch)
                except Exception as _pe:
                    log.debug(f"push_cycle fail: {_pe}")

                for asset in active_assets:
                    setups = _setups_by_asset.get(asset, [])
                    if not setups:
                        continue

                    # FIX 2026-05-20 : itere sur TOUS les setups valides (pas juste le dernier)
                    # Avant : setup=setups[-1] -> on ratait 4/5 setups quand recent_cutoff=60min
                    # On trie du plus recent au plus vieux pour traiter les frais d'abord.
                    setups_sorted = sorted(setups, key=lambda s: s['ts'], reverse=True)
                    for setup in setups_sorted:
                        # Ignore les setups valides AVANT le demarrage du bot
                        if setup['ts'] < BOT_START_TS:
                            log.debug(f"SKIP setup pre-start {asset} ts={setup['ts']} < {BOT_START_TS}")
                            continue

                        setup_key = (asset, str(setup['ts']))

                        # Skip si deja vu il y a moins de 30 min
                        if setup_key in state._seen_setups:
                            last_seen = state._seen_setups[setup_key]
                            if (now - last_seen).total_seconds() < 1800:  # 30 min
                                log.debug(f"SKIP setup deja vu {asset} ts={setup['ts']}")
                                continue
                        state._seen_setups[setup_key] = now

                        log.info(
                            f"SETUP {asset} {setup['ob'].direction} "
                            f"ts={setup['ts']} ML={setup['proba']:.3f} score={setup['r'].score}"
                        )

                        if test_dry_run:
                            log.info(f"  [DRY RUN] order non place")
                            continue

                        # Re-check concurrent (peut avoir change entre temps)
                        n_open_now = mt5_exec.get_n_open_positions()
                        n_pending_now = len(mt5_exec.get_pending_orders(magic=BOT_MAGIC))
                        if n_open_now + n_pending_now >= MAX_CONCURRENT:
                            break

                        execute_setup(mt5_exec, state, setup, balance)
                        # Une fois un setup execute pour cet actif, on passe au suivant
                        # (cooldown 15min va bloquer les autres setups du meme actif)
                        break

                    # Cleanup vieilles entrees (> 2h)
                    state._seen_setups = {
                        k: v for k, v in state._seen_setups.items()
                        if (now - v).total_seconds() < 7200
                    }
                    # Cleanup _evaluated_obs : garde les 800 derniers ts/actif
                    # (un OB hors recent_cutoff ne sera plus jamais re-evalue,
                    # donc pas besoin de garder sa cle indefiniment).
                    _ev_asset = state._evaluated_obs.get(asset)
                    if _ev_asset and len(_ev_asset) > 800:
                        state._evaluated_obs[asset] = set(sorted(_ev_asset)[-800:])

                # V5.9 (2026-05-21) : Stats poussees a chaque cycle (~20s).
                # Avant : int(time()) % 300 < SCAN_INTERVAL_SEC -> fenetre de 5s
                # toutes les 5 min, ratee 95% du temps -> dashboard vide.
                # Maintenant : 1 STATS par cycle, le dashboard voit la balance
                # et les KPIs en quasi-temps reel.
                stats = state.get_stats()
                # Log INFO seulement toutes les ~5 min pour ne pas polluer
                _log_now = (int(time.time()) % 300) < SCAN_INTERVAL_SEC
                if _log_now:
                    log.info(
                        f"STATS | total={stats['total']} W={stats['wins']} L={stats['losses']} "
                        f"WR={stats['wr']:.1f}% PnL={stats['pnl_total']:+.2f}€ "
                        f"balance={balance:.2f}€"
                    )
                # Push STATS vers le dashboard a CHAQUE cycle
                try:
                    _eq = None
                    try:
                        _eq = mt5_exec.get_equity()
                    except Exception:
                        pass
                    PUSHER.push_stats(
                        total=stats.get("total", 0),
                        wins=stats.get("wins", 0),
                        losses=stats.get("losses", 0),
                        wr_pct=stats.get("wr", 0),
                        pnl_total=stats.get("pnl_total", 0),
                        balance=balance,
                        equity=_eq,
                        positions_open=n_open,
                        pending_orders=n_pending,
                    )
                except Exception as _pe:
                    log.debug(f"push_stats fail: {_pe}")

                # Push POSITIONS_SYNC : etat reel MT5 -> dashboard marque
                # FILLED (avec PnL flottant) / CANCELLED les ordres disparus.
                # On envoie les details (instrument/direction/entry/sl/tp/volume)
                # pour permettre au dashboard de recreer une ligne FILLED si elle
                # n'existe pas (orphan recovery : TRADE_EXECUTED perdu ou bot
                # redemarre).
                try:
                    _open_pos = mt5_exec.get_open_positions(magic=BOT_MAGIC)
                    _pend = mt5_exec.get_pending_orders(magic=BOT_MAGIC)
                    PUSHER.push_positions_sync(
                        open_positions=[
                            {
                                "ticket": p["ticket"],
                                "pnl": p["pnl"],
                                "symbol": p.get("symbol"),
                                "direction": ("bullish" if p.get("type") == 0
                                              else "bearish"),
                                "entry": p.get("price_open"),
                                "sl": p.get("sl"),
                                "tp": p.get("tp"),
                                "volume": p.get("volume"),
                            }
                            for p in _open_pos
                        ],
                        pending_tickets=[o["ticket"] for o in _pend],
                    )
                except Exception as _pe:
                    log.debug(f"push_positions_sync fail: {_pe}")

            except Exception as e:
                log.exception(f"Erreur dans la boucle : {e}")
                state.log_event("ERROR", str(e))
                try:
                    PUSHER.push_error(str(e))
                except Exception:
                    pass

            time.sleep(SCAN_INTERVAL_SEC)

    except KeyboardInterrupt:
        log.info("Arret manuel (Ctrl+C)")
        state.log_event("STOP", "Manual stop (Ctrl+C)")
    finally:
        try:
            POOL.shutdown(wait=False, cancel_futures=True)
            log.info("ProcessPool ferme")
        except Exception:
            pass
        # V5.8 : flush + shutdown du pusher dashboard
        try:
            PUSHER.push_stop("Bot arrete")
            PUSHER.shutdown()
        except Exception:
            pass
        mt5_exec.shutdown()
        state.close()
        log.info("Bot arrete proprement")


if __name__ == "__main__":
    import sys
    dry_run = "--dry-run" in sys.argv
    run_live(test_dry_run=dry_run)
