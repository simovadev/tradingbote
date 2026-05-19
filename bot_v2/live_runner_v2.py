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
LIVE_ASSETS = [
    # Phase 1-3 (8 actifs valides)
    "XAUUSD", "NAS100", "GER40", "BTCUSD", "EURUSD", "GBPUSD", "AUDUSD", "USDJPY",
    # Phase 4 (5 indices + 2 forex)
    "SP500", "DJ30", "UK100", "FRA40", "JP225", "USDCAD", "USDCHF",
]

# Reduire la liste en mode test (5€ : pas assez pour BTC/NAS/GER lot min)
TEST_MODE_ASSETS = ["XAUUSD", "EURUSD", "GBPUSD", "AUDUSD", "USDJPY", "USDCAD", "USDCHF"]
TEST_MODE_THRESHOLD = 50.0  # Balance < 50€ = mode test

# Scan interval (user 2026-05-18 : 30s -> 15s pour plus de reactivite)
SCAN_INTERVAL_SEC = 15

# Cooldown par actif
COOLDOWN_SEC = 15 * 60

# Max trades concurrents (toutes assets)
MAX_CONCURRENT = 3

# MM progressif (user 2026-05-18 : sprint 30% jusqu'a 5000€ puis 5% safe)
# Objectif : atteindre 5000€ en ~2 semaines (vs 3 avec 20%)
# Risque DD max attendu : -87% sur 5 LOSS consecutifs (90% chance sur 8.5 mois)
# Aucune liquidation sur 200 simulations 480 trades.
THRESHOLD_SAFE_MODE = 5000.0  # balance >= 5000 -> mode 5%
RISK_PCT_AGGRESSIVE = 0.30    # 30% par trade jusqu'a 5000€ (sprint)
RISK_PCT_SAFE = 0.05
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

def scan_asset(mt5_exec: MT5Executor, instrument: str, state: LiveState) -> list[dict]:
    """Scanne un actif : fetch bougies + pipeline Vizion + ML filter.

    Returns:
        Liste de setups valides PRETS a etre executes (deja filtres ML, hors cooldown).
    """
    # 1. Fetch les bougies
    df_m1 = mt5_exec.get_bars(instrument, "M1", N_BARS_M1)
    if df_m1 is None or len(df_m1) < 200:
        return []

    df_m15 = mt5_exec.get_bars(instrument, "M15", N_BARS_M15)
    df_h1 = mt5_exec.get_bars(instrument, "H1", N_BARS_H1)
    if df_m15 is None or df_h1 is None:
        return []

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

    # 3. Filtre : on ne s'interesse qu'aux OB RECENTS (user 2026-05-18 : 10 min -> 60 min)
    # OB+MSS Vizion prennent souvent 15-30 min entre OB et confirmation MSS finale
    # Avec cooldown 15min/actif, pas de risque de re-prendre le meme setup
    now = df_m1.index[-1]
    recent_cutoff = now - pd.Timedelta(minutes=60)
    obs_recent = [ob for ob in obs_confirmed if df_m1.index[ob.validation_index] >= recent_cutoff]

    if not obs_recent:
        return []

    # 4. Pipeline Vizion + ML
    loaded = load_model(instrument)
    if loaded is None:
        return []
    model, features = loaded
    threshold = ml_filter.ML_THRESHOLDS.get(instrument, 0.55)

    valid_setups = []
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
            continue

        if r.verdict != "TRADE" or r.trade_setup is None:
            state.log_rejected(instrument, df_m1.index[ob.validation_index],
                              ob.direction, r.rejection_reason or "no_trade")
            continue

        proba = predict_proba(model, features, r, ob, instrument)
        if proba < threshold:
            state.log_rejected(instrument, df_m1.index[ob.validation_index],
                              ob.direction, f"ml_below_thr_{proba:.3f}",
                              ml_proba=proba, score=r.score)
            continue

        valid_setups.append({
            "instrument": instrument,
            "ts": df_m1.index[ob.validation_index],
            "ob": ob,
            "r": r,
            "proba": proba,
            "df_m1": df_m1,
        })

    return valid_setups


# ========== EXECUTION ==========

def execute_setup(mt5_exec: MT5Executor, state: LiveState, setup_dict: dict,
                  balance: float) -> bool:
    """Place l'ordre MT5 pour un setup."""
    instrument = setup_dict["instrument"]
    r = setup_dict["r"]
    ob = setup_dict["ob"]
    setup = r.trade_setup

    # Calcul lots selon balance (utilise les VRAIES valeurs MT5 du broker)
    risk_pct = get_risk_pct(balance)
    try:
        info = mt5_exec.symbol_info(instrument)
        if info is None:
            log.error(f"Symbol info None pour {instrument}")
            return False

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
        lots = risk_eur / risk_per_lot
        log.debug(f"{instrument} lots calc : sl_dist={sl_distance}, risk_per_lot={risk_per_lot:.2f}, "
                  f"lots={lots:.4f} [source={calcul_source}]")
        # Arrondi au volume_step
        vol_step = info.volume_step
        lots = max(info.volume_min, round(lots / vol_step) * vol_step)
        lots = round(lots, 2)

        if lots > info.volume_max:
            lots = info.volume_max
        if lots < info.volume_min:
            log.warning(f"Lots calcule {lots} < min {info.volume_min} sur {instrument}, SKIP")
            return False

        # === SECURITE 1 : Verifie que la perte max sur SL <= 1.5 x risk vise ===
        # Recalcul avec les MEMES valeurs que le calcul des lots (coherence)
        if tick_value_real > 0 and tick_size_real > 0:
            perte_si_sl = (sl_distance / tick_size_real) * tick_value_real * lots
        else:
            perte_si_sl = sl_distance * INSTRUMENTS[instrument]["tick_value"] * lots

        if perte_si_sl > risk_eur * 1.5:
            log.error(
                f"REJET {instrument} : perte SL={perte_si_sl:.2f} > 1.5x risk ({risk_eur*1.5:.2f}). "
                f"Lots={lots} probablement faux. SKIP."
            )
            state.log_event("WARN", f"Lots foireux {instrument} : skip")
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

    open_positions = {p["ticket"] for p in mt5_exec.get_positions()}
    closed_tickets = [t for t in pending if t not in open_positions]

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

    try:
        while True:
            try:
                # User 2026-05-19 : bonus Vantage de 50€ ajoute au calcul du risk
                # ATTENTION : le bonus n'est pas du cash retirable, c'est juste pour gonfler
                # le capital de calcul. Si LOSS, c'est le cash reel qui est entame.
                BONUS_EUR = 50.0
                balance = mt5_exec.get_balance() + BONUS_EUR
                active_assets = get_active_assets(balance)
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
                for asset in active_assets:
                    # Cooldown
                    if state.is_in_cooldown(asset, now, COOLDOWN_SEC):
                        continue

                    setups = scan_asset(mt5_exec, asset, state)
                    if not setups:
                        continue

                    # Garde le plus recent (le dernier valide)
                    setup = setups[-1]
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
