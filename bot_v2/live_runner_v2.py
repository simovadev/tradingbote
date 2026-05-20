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
import pickle
import time
from pathlib import Path
from typing import Any

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
from bot_v2.live_state import LiveState
from bot_v2.mt5_executor import MT5Executor
from bot_v2.pipeline import evaluate_ob
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
N_BARS_M1 = 2000   # 2000 minutes = ~33h
N_BARS_M15 = 500
N_BARS_H1 = 500
N_BARS_D1 = 100


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
    if instrument in _models_cache:
        return _models_cache[instrument]
    model_path = BOT_V2_DIR / f"ml_model_{instrument}.pkl"
    feat_path = BOT_V2_DIR / f"ml_features_{instrument}.json"
    if not model_path.exists():
        log.warning(f"Modele manquant pour {instrument}, skip")
        return None
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    features = json.load(open(feat_path))["features"]
    _models_cache[instrument] = (model, features)
    return model, features


def predict_proba(model, features, r, ob, instrument) -> float:
    feats = ml_filter._features_from_result(r, ob, instrument)
    X = pd.DataFrame([[feats.get(f, 0) for f in features]], columns=features)
    return float(model.predict_proba(X)[0, 1])


# ========== PIPELINE PAR ACTIF ==========

def scan_asset(mt5_exec: MT5Executor, instrument: str, state: LiveState, balance: float | None = None, debug_diag: bool = False) -> list[dict]:
    """Scanne un actif : fetch bougies + pipeline Vizion + ML filter.

    Args:
        balance: solde courant (pour seuil ML dynamique : <3K=0.55, >=3K=0.70)
        debug_diag: log diagnostic complet (bougies, OB, rejections) - 1 fois/5min.

    Returns:
        Liste de setups valides PRETS a etre executes (deja filtres ML, hors cooldown).
    """
    # 1. Fetch les bougies
    df_m1 = mt5_exec.get_bars(instrument, "M1", N_BARS_M1)
    if df_m1 is None or len(df_m1) < 200:
        if debug_diag:
            log.info(f"DIAG {instrument}: M1 KO (df_m1={None if df_m1 is None else len(df_m1)} bougies)")
        return []

    # FIX 2026-05-20 : virer la DERNIERE bougie M1 (souvent en cours, pas close)
    # Sinon le bot detecte des OB sur bougie incomplete -> setup change a chaque tick
    df_m1 = df_m1.iloc[:-1]

    df_m15 = mt5_exec.get_bars(instrument, "M15", N_BARS_M15)
    df_h1 = mt5_exec.get_bars(instrument, "H1", N_BARS_H1)
    if df_m15 is None or df_h1 is None:
        return []
    # FIX : pareil pour M15 et H1 (bougie courante en formation)
    df_m15 = df_m15.iloc[:-1]
    df_h1 = df_h1.iloc[:-1]

    # D1 : build depuis H1 (pas toujours dispo natif chez les brokers)
    try:
        df_d1 = mt5_exec.get_bars(instrument, "D1", N_BARS_D1)
        if df_d1 is None or len(df_d1) < 10:
            raise ValueError
    except Exception:
        df_d1 = build_d1_from_h1(df_h1)

    # HTF swings (D1, H4, H1)
    htf_dfs: dict[str, pd.DataFrame] = {"H1": df_h1, "D1": df_d1}
    try:
        df_h4 = mt5_exec.get_bars(instrument, "H4", 500)
        if df_h4 is not None and len(df_h4) > 0:
            htf_dfs["H4"] = df_h4
    except Exception:
        pass
    htf_swings = collect_htf_swings(htf_dfs, swing_strength=3)

    # SMT correles
    correlated_dfs = {}
    for corr_name, corr_type in SMT_PAIRS.get(instrument, []):
        try:
            df_c = mt5_exec.get_bars(corr_name, "M1", N_BARS_M1)
            if df_c is not None and len(df_c) > 0:
                correlated_dfs[corr_name] = (df_c, corr_type)
        except Exception:
            continue

    # 2. Detection OB+MSS
    sws = get_param(instrument, "swing_strength_m1", 2)
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

    mss_setups = detect_mss_setups(
        df_m1, structure_breaks=cache["structure_breaks"], swings=cache["swings_ltf"]
    )
    obs_confirmed = confirm_ob_with_mss(obs, mss_setups, window_bars=10)

    # 3. Filtre : on ne s'interesse qu'aux OB RECENTS
    # User 2026-05-20 : 60 min -> 15 min (compromis entre temps reel et marge MSS)
    now = df_m1.index[-1]
    recent_cutoff = now - pd.Timedelta(minutes=15)
    obs_recent = [ob for ob in obs_confirmed if df_m1.index[ob.validation_index] >= recent_cutoff]

    if debug_diag:
        from bot_v2.concepts.killzones import killzone_at
        last_ts = df_m1.index[-1]
        # Affiche aussi tous les OB confirmes (recents OU non) pour voir s'il y a eu activite
        latest_ob_ts = "aucun"
        if obs_confirmed:
            latest_ob_ts = df_m1.index[max(ob.validation_index for ob in obs_confirmed)]
        kz_now = killzone_at(last_ts) or "AUCUNE"
        # Compte aussi sur 60 min pour comparaison (vs notre fenetre 15 min)
        cutoff_60 = last_ts - pd.Timedelta(minutes=60)
        n_obs_60min = sum(1 for ob in obs_confirmed if df_m1.index[ob.validation_index] >= cutoff_60)
        log.info(
            f"DIAG {instrument}: M1={len(df_m1)} M15={len(df_m15)} H1={len(df_h1)} "
            f"last_bar={last_ts} kz={kz_now} | OB_brut={len(obs)} OB+MSS={len(obs_confirmed)} "
            f"latest_OB_MSS_ts={latest_ob_ts} OB_60min={n_obs_60min} OB_15min={len(obs_recent)}"
        )

    if not obs_recent:
        return []

    # 4. Pipeline Vizion + ML
    loaded = load_model(instrument)
    if loaded is None:
        return []
    model, features = loaded
    # V4.1 (user 2026-05-20) : seuil dynamique selon balance
    # < 3000E -> 0.55 (sprint, +volume) | >= 3000E -> 0.70 (conso, +qualite)
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

        proba = predict_proba(model, features, r, ob, instrument)
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

    return valid_setups


# ========== EXECUTION ==========

def execute_setup(mt5_exec: MT5Executor, state: LiveState, setup_dict: dict,
                  balance: float) -> bool:
    """Place l'ordre MT5 pour un setup."""
    instrument = setup_dict["instrument"]
    r = setup_dict["r"]
    ob = setup_dict["ob"]
    setup = r.trade_setup

    # === FIX 2026-05-20 : check derive prix + fraicheur setup ===
    # 1. Age du setup : si OB valide y'a >5 min, le marche a probablement bouge trop
    age_setup_min = (pd.Timestamp.now(tz="UTC") - ob.validation_ts).total_seconds() / 60
    if age_setup_min > 15:
        log.warning(f"SETUP TROP VIEUX {instrument} : validation il y a {age_setup_min:.1f} min, SKIP")
        return False

    # 2. Derive prix : verifie que entry n'est pas trop loin du prix actuel (max 0.3%)
    info = mt5_exec.symbol_info(instrument)
    if info is None:
        log.error(f"Symbol info None pour {instrument}")
        return False
    tick = mt5_exec.get_tick(instrument)
    if tick is None or tick.bid <= 0:
        log.error(f"Tick None pour {instrument}")
        return False
    current_price = (tick.bid + tick.ask) / 2
    entry_setup = float(setup.entry_price)
    sl_distance = abs(setup.entry_price - setup.stop_loss)
    derive = abs(current_price - entry_setup)
    # FIX 2026-05-20 : check absolu en % du SL distance (et plus en % du prix qui etait trop strict)
    # Reject si derive > 50% de la SL distance (= au-dela le RR casse a moitie)
    # Avant : 30% absolu sur prix -> rejetait XAUUSD avec SL 0.10% du prix
    if derive > sl_distance * 0.50:
        derive_pct = derive / current_price * 100
        sl_distance_pct = sl_distance / current_price * 100
        log.warning(
            f"DERIVE PRIX TROP GRANDE {instrument} : "
            f"entry={entry_setup:.5f} now={current_price:.5f} "
            f"derive={derive_pct:.3f}% > 50%*SL({sl_distance_pct:.3f}%), SKIP"
        )
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
        # Empeche d'avoir tous les fonds bloques en margin sur 1 trade
        try:
            margin_required = info.margin_initial * lots if info.margin_initial > 0 else None
            if margin_required is not None:
                if margin_required > balance * 0.8:
                    log.error(
                        f"REJET {instrument} : margin requis {margin_required:.2f}€ > 80% balance ({balance*0.8:.2f}€). SKIP."
                    )
                    return False
        except Exception:
            pass  # Pas critique si margin_initial pas dispo

    except Exception as e:
        log.error(f"Calc lots error {instrument}: {e}")
        return False

    # V2 : Place un ordre LIMIT au prix de l'OB (= comportement backtest).
    # Si prix touche entry_price -> fill au prix exact. Si non -> expire en 60min.
    result = mt5_exec.place_limit_order(
        symbol=instrument,
        direction=setup.direction,
        volume=lots,
        entry_price=entry_price,
        sl=sl_price,
        tp=tp_price,
        expiration_minutes=60,
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
                obs = detect_order_blocks(df_m1, swing_strength=sws, max_group_size=2)
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

    log.info("=" * 70)
    log.info("FIN BOOT DIAGNOSTICS")
    log.info("=" * 70)


# ========== MAIN LOOP ==========

def run_live(test_dry_run: bool = False):
    """Boucle principale du bot live.

    Args:
        test_dry_run: si True, ne place PAS d'ordres (juste detecte et log).
    """
    # Log path relatif au repertoire du script (compat PC dev + VPS prod)
    log_path = Path(__file__).resolve().parent.parent / "live.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(str(log_path)),
            logging.StreamHandler(),
        ],
    )
    log.info("=" * 60)
    log.info("BOT LIVE DEMARRAGE")
    log.info("=" * 60)

    state = LiveState()
    state.log_event("START", f"Bot live demarre (dry_run={test_dry_run})")

    mt5_exec = MT5Executor()
    if not mt5_exec.initialize():
        log.error("MT5 init FAILED - bot ne demarre pas")
        state.log_event("ERROR", "MT5 init failed")
        return

    log.info(f"Connecte au compte {mt5_exec.account_info.login} sur {mt5_exec.account_info.server}")
    cash_real = mt5_exec.get_balance()
    log.info(f"Cash reel : {cash_real:.2f} {mt5_exec.account_info.currency} + bonus 50€ = base calcul {cash_real + 50:.2f}€")
    if test_dry_run:
        log.info("** MODE DRY RUN - aucun ordre ne sera place **")

    # FIX 2026-05-20 (user: "lorsque je redemarrer le bot ca placer des order limite") :
    # On marque le timestamp de demarrage. Au scan, on IGNORE tout OB dont la validation
    # est anterieure a BOT_START_TS -> evite de re-placer des LIMIT sur des setups
    # deja passes (ou deja tradés avant restart).
    BOT_START_TS = pd.Timestamp.now(tz="UTC")
    log.info(f"BOT_START_TS = {BOT_START_TS} (setups anterieurs ignores)")

    # === BOOT DIAGNOSTICS : audit complet avant de demarrer la boucle ===
    try:
        boot_diagnostics(mt5_exec, state)
    except Exception as e:
        log.exception(f"Boot diagnostics failed (continue quand meme) : {e}")

    try:
        while True:
            try:
                # User 2026-05-19 : bonus Vantage de 50€ ajoute au calcul du risk
                # ATTENTION : le bonus n'est pas du cash retirable, c'est juste pour gonfler
                # le capital de calcul. Si LOSS, c'est le cash reel qui est entame.
                BONUS_EUR = 50.0
                balance = mt5_exec.get_balance() + BONUS_EUR
                # V4.1 : seuil ML dynamique selon balance
                _ml_thr = "0.55 (Sprint)" if balance < 3000 else "0.70 (Conso)"
                active_assets = get_active_assets(balance)

                # V4.1 FIX 2026-05-20 : cleanup pending orders > 15 min
                # (Vantage ne supporte pas ORDER_TIME_SPECIFIED -> on cleanup manuellement)
                # 15 min car au-dela le marche a trop bouge pour que l'OB reste pertinent
                from datetime import timezone as _tz
                _now = pd.Timestamp.now(tz=_tz.utc)
                _pending_all = mt5_exec.get_pending_orders(magic=BOT_MAGIC)
                for _po in _pending_all:
                    age_min = (_now - _po["time_setup"]).total_seconds() / 60
                    if age_min > 15:
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

                # DIAG : log diagnostic complet chaque ~5 min (1 fois par actif)
                _diag_now = int(time.time()) % 300 < SCAN_INTERVAL_SEC

                for asset in active_assets:
                    # Cooldown
                    if state.is_in_cooldown(asset, now, COOLDOWN_SEC):
                        continue

                    setups = scan_asset(mt5_exec, asset, state, balance=balance, debug_diag=_diag_now)
                    if not setups:
                        continue

                    # Garde le plus recent (le dernier valide)
                    setup = setups[-1]

                    # FIX 2026-05-20 : ignore les setups valides AVANT le demarrage du bot
                    # (sinon au restart le bot re-place des LIMIT sur de vieux OB)
                    if setup['ts'] < BOT_START_TS:
                        log.debug(f"SKIP setup pre-start {asset} ts={setup['ts']} < {BOT_START_TS}")
                        continue

                    setup_key = (asset, str(setup['ts']))

                    # FIX : skip si deja vu il y a moins de 30 min
                    if setup_key in state._seen_setups:
                        last_seen = state._seen_setups[setup_key]
                        if (now - last_seen).total_seconds() < 1800:  # 30 min
                            log.debug(f"SKIP setup deja vu {asset} ts={setup['ts']}")
                            continue
                    state._seen_setups[setup_key] = now

                    # Cleanup vieilles entrees (> 2h)
                    state._seen_setups = {
                        k: v for k, v in state._seen_setups.items()
                        if (now - v).total_seconds() < 7200
                    }

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

                # Stats periodiques (chaque ~5 min)
                if int(time.time()) % 300 < SCAN_INTERVAL_SEC:
                    stats = state.get_stats()
                    log.info(
                        f"STATS | total={stats['total']} W={stats['wins']} L={stats['losses']} "
                        f"WR={stats['wr']:.1f}% PnL={stats['pnl_total']:+.2f}€ "
                        f"balance={balance:.2f}€"
                    )

            except Exception as e:
                log.exception(f"Erreur dans la boucle : {e}")
                state.log_event("ERROR", str(e))

            time.sleep(SCAN_INTERVAL_SEC)

    except KeyboardInterrupt:
        log.info("Arret manuel (Ctrl+C)")
        state.log_event("STOP", "Manual stop (Ctrl+C)")
    finally:
        mt5_exec.shutdown()
        state.close()
        log.info("Bot arrete proprement")


if __name__ == "__main__":
    import sys
    dry_run = "--dry-run" in sys.argv
    run_live(test_dry_run=dry_run)
