"""ML filter — applique le modele LightGBM comme filtre final apres Vizion.

Decision user 2026-05-16 :
- Seuil 0.50 : plus de trades (~25/mois sur 5 actifs), WR ~58% (vs 64% a 0.55).
- Garde la couche Vizion comme detection, le ML filtre la qualite.
"""
from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Any

import pandas as pd


_MODEL_PATH = Path("c:/Users/Shadow/TradingBot/bot_v2/ml_model.pkl")
_FEATURES_PATH = Path("c:/Users/Shadow/TradingBot/bot_v2/ml_features.json")

# Multi-TF (user 2026-05-16) : models specifiques par TF si existants
_MODELS_BY_TF: dict[str, Path] = {
    "M1": Path("c:/Users/Shadow/TradingBot/bot_v2/ml_model.pkl"),
    "M5": Path("c:/Users/Shadow/TradingBot/bot_v2/ml_model_M5.pkl"),
}
_FEATURES_BY_TF: dict[str, Path] = {
    "M1": Path("c:/Users/Shadow/TradingBot/bot_v2/ml_features.json"),
    "M5": Path("c:/Users/Shadow/TradingBot/bot_v2/ml_features_M5.json"),
}

# V4 (user 2026-05-20) : displacement dynamique. Fallback V3.5 si V4 absent.
_MBASE = "c:/Users/Shadow/TradingBot/bot_v2"

ALL_ASSETS = ["XAUUSD", "NAS100", "GER40", "BTCUSD",
              "EURUSD", "GBPUSD", "AUDUSD", "USDJPY",
              "SP500", "DJ30", "UK100", "FRA40",
              "USDCAD", "USDCHF"]

def _pick_model(asset: str) -> Path:
    """Cherche V5 > V4 > V3.5."""
    v5 = Path(f"{_MBASE}/ml_model_{asset}_admiral_v5.pkl")
    if v5.exists():
        return v5
    v4 = Path(f"{_MBASE}/ml_model_{asset}_admiral_v4.pkl")
    v3_5 = Path(f"{_MBASE}/ml_model_{asset}_admiral_v3_5.pkl")
    return v4 if v4.exists() else v3_5

def _pick_features(asset: str) -> Path:
    v5 = Path(f"{_MBASE}/ml_features_{asset}_admiral_v5.json")
    if v5.exists():
        return v5
    v4 = Path(f"{_MBASE}/ml_features_{asset}_admiral_v4.json")
    v3_5 = Path(f"{_MBASE}/ml_features_{asset}_admiral_v3_5.json")
    return v4 if v4.exists() else v3_5

_MODELS_BY_INSTRUMENT: dict[str, Path] = {a: _pick_model(a) for a in ALL_ASSETS}
_FEATURES_BY_INSTRUMENT: dict[str, Path] = {a: _pick_features(a) for a in ALL_ASSETS}

# Models specifiques par (INSTRUMENT, TF) — user 2026-05-16
# Fallback : modele instrument generique, puis modele TF generique
_MODELS_BY_INST_TF: dict[tuple[str, str], Path] = {
    ("NAS100", "M5"): Path("c:/Users/Shadow/TradingBot/bot_v2/ml_model_NAS100_M5.pkl"),
}
_FEATURES_BY_INST_TF: dict[tuple[str, str], Path] = {
    ("NAS100", "M5"): Path("c:/Users/Shadow/TradingBot/bot_v2/ml_features_NAS100_M5.json"),
}

# V11.3 thresholds (user 2026-05-22) - seuil 0.70 partout (retour valeur cible).
# Test session London 07-13 UTC simule :
#   @0.65 : 10 trades, WR 57% (4 WIN/3 LOSS/3 NO_FILL)
#   @0.70 : 7 trades,  WR 80% (4 WIN/1 LOSS/2 NO_FILL)  <- meilleur ratio
#   @0.75 : 2 trades,  WR 100% (trop strict)
# Avec V11.2 (cap age 30min aligne OOS), 0.70 donne le meilleur compromis
# volume/WR. Conforme aux chiffres OOS V11 (WR ~81% au seuil 0.70).
# V16 (2026-05-25, user) : seuil 0.55 pour DOUBLER le volume.
# OOS V16 : 0.55 = ~12 tr/jour WR ~71% vs 0.60 = ~7.5 tr/jour WR ~77%.
# 0.55 gagne en R total/jour (+44%) malgre WR plus faible. Risque : plus de
# drawdown (a 5% risk demo, attention aux SL consecutifs).
ML_THRESHOLDS: dict[str, float] = {
    "XAUUSD": 0.55,
    "NAS100": 0.55,
    "GER40":  0.55,
    "BTCUSD": 0.55,
    "EURUSD": 0.55,
    "GBPUSD": 0.55,
    "AUDUSD": 0.55,
    "USDJPY": 0.55,
    "SP500":  0.55,
    "DJ30":   0.55,
    "UK100":  0.55,
    "FRA40":  0.55,
    "USDCAD": 0.55,
    "USDCHF": 0.55,
}
# Seuil par (actif, TF) — surcharge ML_THRESHOLDS si present
ML_THRESHOLDS_BY_INST_TF: dict[tuple[str, str], float] = {
    # ("NAS100", "M5"): 0.55,  # a calibrer apres training
}
DEFAULT_THRESHOLD = 0.50

_model: Any | None = None
_features: list[str] | None = None
_models_cache: dict[str, tuple[Any, list[str]]] = {}
_models_by_inst_cache: dict[str, tuple[Any, list[str]]] = {}
_models_by_inst_tf_cache: dict[tuple[str, str], tuple[Any, list[str]]] = {}


def load_model_for_instrument_tf(instrument: str, tf: str) -> tuple[Any, list[str]] | None:
    """Charge le modele specifique (instrument, TF). None si absent.

    Cascade de fallback (gerse par l'appelant) : (inst,TF) > inst > TF > generique.
    """
    key = (instrument, tf)
    if key in _models_by_inst_tf_cache:
        return _models_by_inst_tf_cache[key]
    model_path = _MODELS_BY_INST_TF.get(key)
    features_path = _FEATURES_BY_INST_TF.get(key)
    if not model_path or not model_path.exists():
        return None
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    features = json.loads(features_path.read_text())["features"]
    _models_by_inst_tf_cache[key] = (model, features)
    return model, features


def load_model_for_instrument(instrument: str) -> tuple[Any, list[str]] | None:
    """Charge le modele specifique a un instrument. Fallback sur ml_model.pkl si absent."""
    if instrument in _models_by_inst_cache:
        return _models_by_inst_cache[instrument]
    model_path = _MODELS_BY_INSTRUMENT.get(instrument)
    features_path = _FEATURES_BY_INSTRUMENT.get(instrument)
    if not model_path or not model_path.exists():
        # Fallback: utilise ml_model.pkl (actuel)
        if _MODEL_PATH.exists():
            with open(_MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            features = json.loads(_FEATURES_PATH.read_text())["features"]
            _models_by_inst_cache[instrument] = (model, features)
            return model, features
        return None
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    features = json.loads(features_path.read_text())["features"]
    _models_by_inst_cache[instrument] = (model, features)
    return model, features


def _load() -> tuple[Any, list[str]]:
    global _model, _features
    if _model is None:
        if not _MODEL_PATH.exists():
            raise FileNotFoundError(f"Modele ML manquant : {_MODEL_PATH}")
        with open(_MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
        meta = json.loads(_FEATURES_PATH.read_text())
        _features = meta["features"]
    return _model, _features  # type: ignore[return-value]


def load_model_for_tf(tf: str) -> tuple[Any, list[str]] | None:
    """Charge le modele specifique a un TF. Retourne (model, features) ou None si absent."""
    if tf in _models_cache:
        return _models_cache[tf]
    model_path = _MODELS_BY_TF.get(tf)
    features_path = _FEATURES_BY_TF.get(tf)
    if not model_path or not model_path.exists():
        return None
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    meta = json.loads(features_path.read_text())
    features = meta["features"]
    _models_cache[tf] = (model, features)
    return model, features


def predict_proba_for_tf(r, ob, instrument: str, tf: str = "M1") -> float | None:
    """Predit la proba avec le modele du TF specifie.

    Cascade : (instrument, TF) > TF generique. None si aucun modele dispo.
    """
    # Priorite 1 : modele specifique (instrument, TF) — ex: NAS100_M5
    loaded = load_model_for_instrument_tf(instrument, tf)
    if loaded is None:
        # Priorite 2 : modele TF generique (ex: M5 base XAUUSD)
        loaded = load_model_for_tf(tf)
    if loaded is None:
        return None
    model, features = loaded
    feat_dict = _features_from_result(r, ob, instrument)
    X = pd.DataFrame([[feat_dict.get(f, 0) for f in features]], columns=features)
    return float(model.predict_proba(X)[0, 1])


# ============================================================================
# V13 (2026-05-23) : atr_regime par killzone
# ============================================================================

# Cap dur de lookback pour atr_regime_v13 : 10 jours = 14400 M1.
# En pratique 5 jours (7200) suffisent quand le marche est ouvert en continu.
_ATR_REGIME_V13_MAX_LOOKBACK = 14400
# Cible : 5 instances passees de la meme killzone
_ATR_REGIME_V13_N_TYPICAL = 5
# Fenetre courante pour le fallback hors KZ
_ATR_REGIME_V13_FALLBACK_WIN = 60

# V15.1 : version M15. 1 KZ ~= 8-16 M15. 5 KZ passees ~= 100 M15 + gaps weekends
# (max ~500 M15 = 5 jours utiles). Cap dur a 1000 M15 (10j) pour edge cases.
_ATR_REGIME_V15_MAX_LOOKBACK_M15 = 1000
# Fenetre fallback hors KZ : 4 M15 (= 1h) au lieu de 60 M1
_ATR_REGIME_V15_FALLBACK_WIN_M15 = 4


def _compute_atr_regime_v15_htf(df_htf, ob_ts, current_kz_name):
    """V15.1 : atr_regime calcule sur M15.

    Args:
        df_htf: DataFrame M15 (high/low + index UTC).
        ob_ts: timestamp UTC de validation de l'OB.
        current_kz_name: str ou None.

    Returns:
        float : ratio atr_now / atr_typical, ou 1.0.
    """
    if df_htf is None or len(df_htf) < 50:
        return 1.0
    try:
        from bot_v2.concepts.killzones import killzone_at
    except Exception:
        return 1.0

    # Index M15 <= ob_ts (strictement amont)
    try:
        ts_index = df_htf.index
        end_idx = ts_index.searchsorted(ob_ts, side="right")  # nb M15 closes <= ob_ts
        if end_idx < 20:
            return 1.0
        start_idx = max(0, end_idx - _ATR_REGIME_V15_MAX_LOOKBACK_M15)
    except Exception:
        return 1.0

    highs = df_htf["high"].values
    lows = df_htf["low"].values

    # --- Cas 1 : on est dans une killzone connue ---
    if current_kz_name is not None:
        # ATR de la KZ courante : bougies M15 de la meme KZ, remontant tant qu'on y est
        kz_start = end_idx
        for j in range(end_idx - 1, max(end_idx - 16, start_idx) - 1, -1):
            if killzone_at(ts_index[j]) == current_kz_name:
                kz_start = j
            else:
                break
        if kz_start >= end_idx:
            return 1.0
        atr_now = float((highs[kz_start:end_idx] - lows[kz_start:end_idx]).mean())
        # V17 FIX-17 : NaN check (NaN <= 0 = False -> sinon retourne NaN sur slice vide)
        import math
        if atr_now <= 0 or not math.isfinite(atr_now):
            return 1.0

        # Collecter 5 dernieres KZ passees du meme nom
        atr_history = []
        cur_end = None
        in_block = False
        j = kz_start - 1
        while j >= start_idx and len(atr_history) < _ATR_REGIME_V13_N_TYPICAL:
            if killzone_at(ts_index[j]) == current_kz_name:
                if not in_block:
                    cur_end = j + 1
                    in_block = True
                j -= 1
            else:
                if in_block:
                    blk_start = j + 1
                    if cur_end - blk_start >= 2:  # min 2 M15 = 30min
                        blk_atr = float((highs[blk_start:cur_end] - lows[blk_start:cur_end]).mean())
                        if blk_atr > 0:
                            atr_history.append(blk_atr)
                    in_block = False
                j -= 1
        if in_block and cur_end is not None:
            blk_start = max(j + 1, start_idx)
            if cur_end - blk_start >= 2:
                blk_atr = float((highs[blk_start:cur_end] - lows[blk_start:cur_end]).mean())
                if blk_atr > 0:
                    atr_history.append(blk_atr)

        if not atr_history:
            return 1.0
        atr_typical = sum(atr_history) / len(atr_history)
        return atr_now / atr_typical if atr_typical > 0 else 1.0

    # --- Cas 2 : hors KZ -> fenetre 4 M15 (= 1h) vs meme heure les 5 derniers jours ---
    win = _ATR_REGIME_V15_FALLBACK_WIN_M15
    if end_idx < win:
        return 1.0
    atr_now = float((highs[end_idx - win:end_idx] - lows[end_idx - win:end_idx]).mean())
    # V17 FIX-17 (idem) : NaN check
    import math
    if atr_now <= 0 or not math.isfinite(atr_now):
        return 1.0

    ts_now = ts_index[end_idx - 1]
    atr_history = []
    for d in range(1, _ATR_REGIME_V13_N_TYPICAL + 1):
        ts_target = ts_now - pd.Timedelta(days=d)
        try:
            j = ts_index.searchsorted(ts_target, side="right")
        except Exception:
            continue
        if j < win or j > end_idx:
            continue
        blk_atr = float((highs[j - win:j] - lows[j - win:j]).mean())
        if blk_atr > 0:
            atr_history.append(blk_atr)

    if not atr_history:
        return 1.0
    atr_typical = sum(atr_history) / len(atr_history)
    return atr_now / atr_typical if atr_typical > 0 else 1.0


def _compute_atr_regime_v13(df_ltf, idx, current_kz_name, df_htf=None, ob_ts=None):
    """ATR de la killzone courante / moyenne ATR des 5 dernieres MEMES KZ.

    V15.1 (2026-05-25) : utilise df_htf (M15) si dispo (recommande).
    M15 permet de couvrir 5 KZ passees avec ~500 bougies (= ~5 jours) au lieu
    de ~7200 M1. Aligne build (df_htf complet, pas de truncation au chunk) et
    live (df_htf via buffer N_BARS_M15=11000).

    Fallback sur df_ltf (M1) si df_htf indisponible (ancien comportement).

    Args:
        df_ltf: DataFrame M1 avec high/low/index ts (fallback).
        idx: int, position de validation de l'OB dans df_ltf.
        current_kz_name: nom de la killzone courante.
        df_htf: DataFrame M15 (recommande) avec high/low/index ts.
        ob_ts: timestamp UTC de la validation de l'OB (pour aligner sur df_htf).

    Returns:
        float : ratio atr_now / atr_typical, ou 1.0 si indeterminable.
    """
    # V15.1 : preferentiel M15 si on a df_htf + ob_ts
    if df_htf is not None and ob_ts is not None and len(df_htf) > 100:
        return _compute_atr_regime_v15_htf(df_htf, ob_ts, current_kz_name)

    # Fallback ancien comportement M1
    if df_ltf is None or idx is None or idx < 60:
        return 1.0
    try:
        from bot_v2.concepts.killzones import killzone_at
    except Exception:
        return 1.0

    # Fenetre de lookback : on prend au plus _MAX_LOOKBACK bougies M1 en arriere
    # (10 jours), mais sans depasser l'index 0.
    start = max(0, idx - _ATR_REGIME_V13_MAX_LOOKBACK)
    if idx - start < 240:
        # Pas assez d'historique pour calculer quoi que ce soit de sense.
        return 1.0

    highs = df_ltf["high"].values
    lows = df_ltf["low"].values
    ts_index = df_ltf.index

    # --- Cas 1 : on est dans une killzone connue ---
    if current_kz_name is not None:
        # ATR depuis le debut de la KZ courante (en remontant tant qu'on y est).
        # On limite a 240 bougies (4h) pour eviter de trainer si la KZ est tres longue
        # ou si on est tombe pile au demarrage d'une KZ qui chevauche un trou.
        kz_start = idx
        for j in range(idx - 1, max(idx - 240, start) - 1, -1):
            if killzone_at(ts_index[j]) == current_kz_name:
                kz_start = j
            else:
                break
        if kz_start >= idx:
            # Cas degenere : pas de bougie de la KZ courante avant idx
            return 1.0
        atr_now = float((highs[kz_start:idx] - lows[kz_start:idx]).mean())
        if atr_now <= 0:
            return 1.0

        # Collecter les ATR des 5 dernieres instances PASSEES de la meme KZ.
        # On scanne en arriere en groupant les bougies consecutives de la meme KZ.
        atr_history = []
        cur_end = None  # fin (exclusive) du bloc en cours
        in_block = False
        j = kz_start - 1
        while j >= start and len(atr_history) < _ATR_REGIME_V13_N_TYPICAL:
            if killzone_at(ts_index[j]) == current_kz_name:
                if not in_block:
                    cur_end = j + 1
                    in_block = True
                j -= 1
            else:
                if in_block:
                    # On vient de sortir d'un bloc -> calcul ATR de [j+1, cur_end)
                    blk_start = j + 1
                    if cur_end - blk_start >= 5:  # min 5 bougies pour etre valable
                        blk_atr = float((highs[blk_start:cur_end] - lows[blk_start:cur_end]).mean())
                        if blk_atr > 0:
                            atr_history.append(blk_atr)
                    in_block = False
                j -= 1
        # Si on est sorti de la boucle en plein bloc, le finaliser
        if in_block and cur_end is not None:
            blk_start = max(j + 1, start)
            if cur_end - blk_start >= 5:
                blk_atr = float((highs[blk_start:cur_end] - lows[blk_start:cur_end]).mean())
                if blk_atr > 0:
                    atr_history.append(blk_atr)

        if not atr_history:
            return 1.0
        atr_typical = sum(atr_history) / len(atr_history)
        return atr_now / atr_typical if atr_typical > 0 else 1.0

    # --- Cas 2 : hors killzone -> fallback fenetre 60 M1 a la meme heure UTC ---
    win = _ATR_REGIME_V13_FALLBACK_WIN
    atr_now = float((highs[idx - win:idx] - lows[idx - win:idx]).mean())
    if atr_now <= 0:
        return 1.0

    # Pour chacune des 5 derniers jours, prendre la meme fenetre [ts - N jours]
    # et calculer son ATR sur win bougies M1.
    ts_now = ts_index[idx]
    atr_history = []
    for d in range(1, _ATR_REGIME_V13_N_TYPICAL + 1):
        ts_target = ts_now - pd.Timedelta(days=d)
        # searchsorted retourne le 1er index >= ts_target
        try:
            j = ts_index.searchsorted(ts_target, side="right")
        except Exception:
            continue
        if j < win or j > idx:
            continue
        blk_atr = float((highs[j - win:j] - lows[j - win:j]).mean())
        if blk_atr > 0:
            atr_history.append(blk_atr)

    if not atr_history:
        return 1.0
    atr_typical = sum(atr_history) / len(atr_history)
    return atr_now / atr_typical if atr_typical > 0 else 1.0


def _features_from_result(r, ob, instrument: str, df_ltf=None, df_d1=None, mss_setups=None, df_htf=None) -> dict:
    """Reproduit EXACTEMENT les features V3.5/V4/V5 du dataset ML (ml_dataset.py _extract_features).

    FIX CRITIQUE 2026-05-20 : avant, 11 features etaient absentes en live (atr, dist_pdh,
    hour_of_day, etc.) -> le ML recevait 0 et donnait des probas catastrophiques (0.04-0.26).
    Maintenant aligne sur le training V5.

    V5 (2026-05-20) : 4 nouvelles features :
    - phase_reversal, phase_manipulation, vol_ratio_setup, has_mss_nearby
    (phase_expansion existe deja comme has_phase_expansion)
    """
    setup = r.trade_setup
    conf = r.confluences or []
    conf_text = " ".join(conf)

    f = {
        # V18.3 (Soufiane 2026-05-27 audit ML) :
        # 'score' et 'quality' RETIRES = combinaisons lineaires des autres features
        # → causent overfitting (train AUC 0.83 vs OOS 0.65 = gap +0.18)
        # On garde les composants atomiques : ob_strength, sweep_strength, etc.
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
        # Killzone one-hot (V15.1 : ajout London_Close ~17% des OB)
        "kz_london": int(r.killzone_name == "London"),
        "kz_ny_am": int(r.killzone_name == "NY_AM"),
        "kz_ny_pm": int(r.killzone_name == "NY_PM"),
        "kz_asia": int(r.killzone_name == "Asia"),
        "kz_ny_lunch": int(r.killzone_name == "NY_Lunch"),
        "kz_london_close": int(r.killzone_name == "London_Close"),
        # Confluences
        "has_smt": int("smt_" in conf_text),
        "has_feu_vert": int("feu_vert" in conf_text),
        "has_breaker_kz": int("breaker_in_KZ" in conf_text),
        "has_mss_fvg": int("MSS_with_FVG" in conf_text),
        "has_po3_dist": int("po3_distribution" in conf_text),
        "has_phase_expansion": int("phase_expansion" in conf_text),
        "has_open_midnight_respect": int("respecte_OpenMidnightNY" in conf_text),
        # V3.5 : indicateurs des filtres relaches
        "has_FVG_sync": int("OB_FVG_sync" in conf_text),
        "has_parent_ob": int("parent_ob_" in conf_text and "no_parent_ob_" not in conf_text),
        "has_grandparent_ob": int("grandparent_ob_" in conf_text and "no_grandparent" not in conf_text),
        "has_good_zone": int("mauvaise_zone" not in conf_text and "zone=" in conf_text),
        "has_session_direction": int("session_sans_direction" not in conf_text),
        # OB structure
        "ob_group_size": ob.group_size,
        # V15.1 FIX A13 : max(0, ...) car sweep_index peut etre > validation_index
        # sur OB replayed/rejected -> evite valeurs negatives parasites.
        "bars_sweep_to_validation": max(0, ob.validation_index - getattr(ob.sweep, "sweep_index", ob.group_start_index)),
        "bars_group_to_validation": max(0, ob.validation_index - ob.group_start_index),
        "is_bullish": int(ob.direction == "bullish"),
    }

    # Features V3 (temps)
    ts = ob.validation_ts
    f["hour_of_day"] = int(ts.hour)
    # BONUS 2 (2026-05-26) : encodage cyclique de l'heure. hour_of_day linaire
    # traite 23h et 00h comme distantes alors qu'elles sont voisines. sin/cos
    # encode la proximite circulaire. Garde hour_of_day pour compat / lisibilite.
    _hour = int(ts.hour)
    f["hour_sin"] = math.sin(2 * math.pi * _hour / 24)
    f["hour_cos"] = math.cos(2 * math.pi * _hour / 24)
    f["day_of_week"] = int(ts.dayofweek)
    # V17 FIX-9 : vraie formule. Avant: (ts.hour*60 + ts.minute) % 60 = ts.minute
    # (bug semantique, juste la minute de l'heure UTC). Maintenant: minutes
    # ecoulees depuis le debut de la KZ NY courante.
    if r.killzone_name:
        try:
            from bot_v2.concepts.killzones import to_ny_time, KILLZONES
            kz_def = next((k for k in KILLZONES if k.name == r.killzone_name), None)
            if kz_def is not None:
                ny_ts = to_ny_time(ts)
                kz_start = ny_ts.replace(hour=kz_def.start_hour,
                                          minute=kz_def.start_minute,
                                          second=0, microsecond=0)
                if kz_start > ny_ts:
                    kz_start -= pd.Timedelta(days=1)
                f["minutes_into_killzone"] = int((ny_ts - kz_start).total_seconds() / 60)
            else:
                f["minutes_into_killzone"] = -1
        except Exception:
            f["minutes_into_killzone"] = -1
    else:
        f["minutes_into_killzone"] = -1

    # V15 (2026-05-25) : volume_relatif = volume de la bougie OB / moyenne 20
    # dernieres bougies M1. Permet au ML de distinguer dead time (faible volume,
    # piege) vs heure active (fort volume, vrai interet institutionnel).
    # > 1.5 = forte participation, < 0.5 = mort. Calcul identique build/live.
    f["volume_relatif"] = 1.0  # defaut neutre si pas calculable
    if (df_ltf is not None and ob.validation_index is not None
            and ob.validation_index >= 20 and "volume" in df_ltf.columns):
        try:
            idx = ob.validation_index
            vol_ob = float(df_ltf["volume"].iloc[idx])
            vol_avg = float(df_ltf["volume"].iloc[idx - 20:idx].mean())
            if vol_avg > 0:
                f["volume_relatif"] = round(vol_ob / vol_avg, 3)
        except Exception:
            pass

    # Features V3 (volatilite ATR)
    if df_ltf is not None and ob.validation_index is not None and ob.validation_index >= 14:
        idx = ob.validation_index
        period_14 = 14
        period_100 = 100
        # ATR 14
        sl14 = df_ltf.iloc[max(0, idx - period_14):idx]
        if len(sl14) > 0:
            hl = sl14["high"] - sl14["low"]
            hc = (sl14["high"] - sl14["close"].shift(1)).abs()
            lc = (sl14["low"] - sl14["close"].shift(1)).abs()
            import pandas as _pd
            tr = _pd.concat([hl, hc, lc], axis=1).max(axis=1)
            atr14 = float(tr.mean()) if len(tr) > 0 else 0.0
        else:
            atr14 = 0.0
        # ATR 100
        sl100 = df_ltf.iloc[max(0, idx - period_100):idx]
        if len(sl100) > 0:
            hl = sl100["high"] - sl100["low"]
            hc = (sl100["high"] - sl100["close"].shift(1)).abs()
            lc = (sl100["low"] - sl100["close"].shift(1)).abs()
            import pandas as _pd
            tr = _pd.concat([hl, hc, lc], axis=1).max(axis=1)
            atr100 = float(tr.mean()) if len(tr) > 0 else 0.0
        else:
            atr100 = 0.0
        f["atr_at_setup"] = atr14
        f["atr_ratio_100"] = (atr14 / atr100) if atr100 > 0 else 1.0
    else:
        f["atr_at_setup"] = 0.0
        f["atr_ratio_100"] = 1.0

    # Features V3 (distance daily levels)
    # FIX V8 (2026-05-21) : data leakage corrige. Avant : df_d1.index < ts
    # incluait la D1 du jour en cours (complete en training, partielle en live)
    # -> features differentes training/live -> ML mal calibre.
    # Maintenant : on filtre sur ts.normalize() (00:00 du jour) pour exclure le D1
    # du jour de validation. yesterday = vraiment hier.
    entry_price = setup.entry_price if setup else (ob.ob_low + ob.ob_high) / 2
    pdh = pdl = d1_open = None
    if df_d1 is not None and len(df_d1) > 0:
        past = df_d1[df_d1.index < ts.normalize()]
        if len(past) >= 2:
            yesterday = past.iloc[-1]
            pdh = float(yesterday["high"])
            pdl = float(yesterday["low"])
            d1_open = float(yesterday["close"])  # today open ~= prev close
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

    # ==================== V18.7 FEATURES NORMALISEES STATIONNAIRES ====================
    # Adversarial validation V18.6 a revele que les features ci-dessus DRIFTENT
    # entre train (2018-2025) et OOS (2025-2026). Cause : prix absolu change (XAU
    # 1300->3500). Les % paraissent normalises mais ne le sont pas vraiment.
    # V18.7 ajoute des versions ATR-normalisees, vraiment stationnaires.
    # On garde les anciennes pour compat / ablation.

    # 1. ATR % du prix : stationnaire vs epoque
    #    XAU 2018 : atr=0.5 USD sur prix 1300 = 0.038%
    #    XAU 2025 : atr=12 USD sur prix 3500 = 0.343% (different mais comparable)
    #    Au moins, magnitudes restent dans la meme echelle
    atr_at_setup = f.get("atr_at_setup", 0.0) or 0.0
    if atr_at_setup > 0 and entry_price > 0:
        f["atr_pct_of_price"] = atr_at_setup / entry_price * 100
    else:
        f["atr_pct_of_price"] = 0.0

    # 2. risk_points en ATR : nb de ATR pour le risque (stationnaire)
    risk_pts = f.get("risk_points", 0) or 0
    if atr_at_setup > 0 and risk_pts > 0:
        f["risk_atr"] = risk_pts / atr_at_setup
    else:
        f["risk_atr"] = 0.0

    # 3. Versions LOG des % de distance (compresse les outliers, plus stable)
    # log(1+x) au lieu de x = transforme les distributions skewed
    import math as _math
    f["dist_pdh_log"] = _math.log1p(f["dist_to_pdh_pct"]) if f["dist_to_pdh_pct"] >= 0 else 0.0
    f["dist_pdl_log"] = _math.log1p(f["dist_to_pdl_pct"]) if f["dist_to_pdl_pct"] >= 0 else 0.0

    # 4. Range journalier en ATR (vs absolu)
    # Calcul via df_ltf (range jour courant) / ATR
    if df_ltf is not None and ob.validation_index is not None and atr_at_setup > 0:
        try:
            ts_day = ts.normalize()
            day_mask = (df_ltf.index >= ts_day) & (df_ltf.index <= ts)
            day_slice = df_ltf[day_mask]
            if len(day_slice) >= 5:
                day_range = float(day_slice["high"].max() - day_slice["low"].min())
                f["day_range_atr"] = day_range / atr_at_setup
            else:
                f["day_range_atr"] = 0.0
        except Exception:
            f["day_range_atr"] = 0.0
    else:
        f["day_range_atr"] = 0.0

    # 5. Distance entry au mid-OB en ATR (vs absolu)
    if atr_at_setup > 0:
        ob_mid = (ob.ob_low + ob.ob_high) / 2
        f["entry_to_ob_mid_atr"] = abs(entry_price - ob_mid) / atr_at_setup if ob_mid > 0 else 0.0
        # Taille de l'OB en ATR (stationnaire vs prix absolu)
        f["ob_size_atr"] = abs(ob.ob_high - ob.ob_low) / atr_at_setup
    else:
        f["entry_to_ob_mid_atr"] = 0.0
        f["ob_size_atr"] = 0.0

    # ==================== NEW V5 FEATURES (2026-05-20) ====================
    f["phase_reversal"] = int("phase_reversal" in conf_text)
    f["phase_manipulation"] = int("phase_manipulation" in conf_text)

    # Volatilite ratio courte/longue (a la validation OB)
    if df_ltf is not None and ob.validation_index is not None and ob.validation_index >= 100:
        h14_v = df_ltf.iloc[ob.validation_index-14:ob.validation_index]["high"]
        l14_v = df_ltf.iloc[ob.validation_index-14:ob.validation_index]["low"]
        h100_v = df_ltf.iloc[ob.validation_index-100:ob.validation_index]["high"]
        l100_v = df_ltf.iloc[ob.validation_index-100:ob.validation_index]["low"]
        a14_v = float((h14_v - l14_v).mean())
        a100_v = float((h100_v - l100_v).mean())
        f["vol_ratio_setup"] = (a14_v / a100_v) if a100_v > 0 else 1.0
    else:
        f["vol_ratio_setup"] = 1.0

    # Presence MSS proche (remplace le filtre dur confirm_ob_with_mss)
    # FIX V9 (2026-05-22) : data leakage corrige. Avant : abs(diff) <= 10 acceptait
    # un MSS dans les 10 bougies APRES validation. Maintenant : on ne regarde QUE
    # les MSS forme AU PLUS TARD a validation_index (passe + present).
    _mss_list = mss_setups or []
    f["has_mss_nearby"] = int(any(
        0 <= ob.validation_index - getattr(mss, "mss", mss).break_index <= 10
        for mss in _mss_list
    ))

    # ==================== NEW V10 FEATURES (2026-05-22) ====================
    # Objectif : donner au ML des valeurs continues riches (avant : flags 0/1).
    # L'analyse WR brut a montre que po3_dist (WR 46%) et le displacement sont
    # les vrais discriminants - mais le ML ne voyait que des flags.

    # --- Groupe A : PO3 enrichi (le signal en or) ---
    f["po3_body_pct"] = float(getattr(r, "po3_body_pct", 0.5))
    f["po3_upper_wick"] = float(getattr(r, "po3_upper_wick", 0.0))
    f["po3_lower_wick"] = float(getattr(r, "po3_lower_wick", 0.0))
    f["po3_aligned"] = int(getattr(r, "po3_aligned", 0))
    f["po3_htf2_aligned"] = int(getattr(r, "po3_htf2_aligned", 0))

    # --- Groupe B : displacement reel (deja calcule, jamais extrait avant) ---
    f["displacement_ratio"] = float(getattr(r, "disp_ratio", 0.0))
    f["fib_level"] = float(getattr(r, "fib_level", 0.5))

    # --- Groupe C : momentum / regime de marche ---
    idx = ob.validation_index
    if df_ltf is not None and idx is not None and idx >= 240:
        closes = df_ltf["close"]
        c_now = float(closes.iloc[idx])
        # Momentum : variation prix sur 60 et 240 bougies M1 (~1h et ~4h)
        c_60 = float(closes.iloc[idx - 60])
        c_240 = float(closes.iloc[idx - 240])
        f["mom_60"] = (c_now - c_60) / c_60 * 100 if c_60 else 0.0
        f["mom_240"] = (c_now - c_240) / c_240 * 100 if c_240 else 0.0
        # Body ratio recent : marche directionnel (corps grands) ou choppy
        # V17 FIX-15 : clip(0, 1) car body/range ne peut depasser 1.0 par definition.
        # Avant : 1e-9 sur doji parfait -> ratio ~1e+15 polluait la moyenne.
        sl20 = df_ltf.iloc[idx - 20:idx]
        rng = (sl20["high"] - sl20["low"])
        body = (sl20["close"] - sl20["open"]).abs()
        f["body_ratio_recent"] = float((body / rng.replace(0, 1e-9)).clip(0, 1).mean())
        # Momentum aligne avec la direction de l'OB ?
        mom_dir = 1 if f["mom_60"] > 0 else -1
        ob_dir = 1 if ob.direction == "bullish" else -1
        f["mom_aligned"] = int(mom_dir == ob_dir)
    else:
        f["mom_60"] = 0.0
        f["mom_240"] = 0.0
        f["body_ratio_recent"] = 0.5
        f["mom_aligned"] = 0

    # --- Groupe D : regime de volatilite (V13 2026-05-23 : par killzone) ---
    # V12 comparait ATR(4h) / ATR(30 jours = 43200 M1), forcant N_BARS_M1=88k.
    # V13 : ATR de la KZ courante / moyenne ATR des 5 dernieres MEMES KZ.
    # Lookback max : ~7200 M1 (5 jours), exceptionnellement 14400 (10j) si trous.
    # Hors KZ : fallback fenetre 60 M1 vs meme fenetre les 5 derniers jours.
    # Coherent avec une strategie M1 ICT/SMC (la volatilite depend de la session,
    # pas d'une moyenne 30j qui melange jours actifs et week-ends fermes).
    # V15.1 FIX A2 (2026-05-25) : atr_regime sur M15 au lieu M1.
    # M15 = 5 KZ passees tiennent dans ~500 M15 (deja dispo via N_BARS_M15=11000).
    # M1 = 5 KZ passees demande ~7200 M1 (probleme cote live N_BARS_M1=500).
    # Alignement : build et live ont tous deux acces a M15. Plus rapide + 0 divergence.
    f["atr_regime"] = _compute_atr_regime_v13(df_ltf, idx, r.killzone_name, df_htf=df_htf, ob_ts=ts)

    # --- Groupe E : distance aux niveaux daily en ATR (pas en %) ---
    # V17 FIX-14 : signe conserve (sous PDH = negatif, au-dessus = positif).
    # Avant: abs() perdait l'info "discount vs premium" critique en ICT.
    atr_ref = f.get("atr_at_setup", 0.0) or 0.0
    if atr_ref > 0 and pdh and entry_price:
        f["dist_pdh_atr"] = (entry_price - pdh) / atr_ref
    else:
        f["dist_pdh_atr"] = 0.0
    if atr_ref > 0 and pdl and entry_price:
        f["dist_pdl_atr"] = (entry_price - pdl) / atr_ref
    else:
        f["dist_pdl_atr"] = 0.0

    # ==================== V18.4 FEATURES ICT (2026-05-27) ====================
    # F1 : distance au round number (50/100/1000 selon actif) en ATR.
    # ICT : les liquidites campent sur les round numbers. Plus on est proche, plus
    # le sweep/raid est probable.
    if entry_price and atr_ref > 0:
        scale = 10 ** int(math.floor(math.log10(abs(entry_price))))
        # round number = entier le plus proche au scale (ex: 1.0850 -> 1.0900)
        rn_unit = scale / 10.0  # tick "naturel" : 0.001 EUR, 1 XAU, 100 BTC
        rn_above = math.ceil(entry_price / rn_unit) * rn_unit
        rn_below = math.floor(entry_price / rn_unit) * rn_unit
        dist_rn = min(abs(entry_price - rn_above), abs(entry_price - rn_below))
        f["dist_round_atr"] = dist_rn / atr_ref
    else:
        f["dist_round_atr"] = 1.0

    # F2 : asymetrie de liquidite (highs vs lows touches dans les 240 dernieres M1)
    # ICT : OB pertinent = ciblage d'un cote dominant (asymetrie elevee).
    # Si autant de highs que lows -> range -> setup moins pertinent.
    if df_ltf is not None and ob.validation_index is not None and ob.validation_index >= 240:
        idx = ob.validation_index
        sl = df_ltf.iloc[idx - 240:idx]
        h_max = sl["high"].max()
        l_min = sl["low"].min()
        # Compte combien de bougies ont swing high == h_max (et idem low)
        n_h = int((sl["high"] >= h_max * 0.9995).sum())
        n_l = int((sl["low"] <= l_min * 1.0005).sum())
        total = n_h + n_l
        if total > 0:
            # asymetrie [-1, 1] : 1 = uniquement highs, -1 = uniquement lows
            f["liq_asymmetry"] = (n_h - n_l) / total
        else:
            f["liq_asymmetry"] = 0.0
    else:
        f["liq_asymmetry"] = 0.0

    # F3 : ADR consumed % - range journalier deja consomme a la validation OB
    # Si > 80% : journee deja epuisee, mouvement supplementaire improbable.
    # Si < 30% : tot, beaucoup d'amplitude restante.
    if df_ltf is not None and ob.validation_index is not None:
        try:
            idx = ob.validation_index
            ts_day_start = ts.normalize()
            day_mask = (df_ltf.index >= ts_day_start) & (df_ltf.index <= ts)
            day_slice = df_ltf[day_mask]
            if len(day_slice) >= 5:
                day_range = float(day_slice["high"].max() - day_slice["low"].min())
                # ADR = average daily range sur les 14 derniers jours via df_d1
                if df_d1 is not None and len(df_d1) >= 14:
                    past_d1 = df_d1[df_d1.index < ts_day_start]
                    if len(past_d1) >= 14:
                        adr = float((past_d1["high"] - past_d1["low"]).iloc[-14:].mean())
                        f["adr_consumed_pct"] = day_range / adr if adr > 0 else 0.0
                    else:
                        f["adr_consumed_pct"] = 0.0
                else:
                    f["adr_consumed_pct"] = 0.0
            else:
                f["adr_consumed_pct"] = 0.0
        except Exception:
            f["adr_consumed_pct"] = 0.0
    else:
        f["adr_consumed_pct"] = 0.0

    # F4 : interactions (top combinaisons identifiees dans l'audit V18.1)
    # daily_bias_aligned + FVG_sync = +18pts WR cumule -> interaction puissante
    f["bias_x_fvg"] = f["daily_bias_aligned"] * f["has_FVG_sync"]
    # Confluence forte : bias + parent_ob + FVG
    f["bias_x_parent_x_fvg"] = (f["daily_bias_aligned"]
                                * f["has_parent_ob"]
                                * f["has_FVG_sync"])
    # NY_AM x bias = killzone premium
    f["kz_ny_am_x_bias"] = f["kz_ny_am"] * f["daily_bias_aligned"]
    # OB strength x sweep strength = qualite structurelle
    f["ob_x_sweep_strength"] = f["ob_strength"] * f["sweep_strength"]
    # Unicorn + KZ premium (NY)
    f["unicorn_x_ny"] = f["is_unicorn"] * (f["kz_ny_am"] + f["kz_ny_pm"])

    return f


def predict_proba(r, ob, instrument: str, df_ltf=None, df_d1=None, mss_setups=None) -> float:
    """Retourne la probabilite de WIN [0,1] selon le modele specifique a l'actif."""
    # Multi-asset (user 2026-05-16) : utilise le modele par instrument si dispo
    loaded = load_model_for_instrument(instrument)
    if loaded is not None:
        model, features = loaded
    else:
        model, features = _load()
    feat_dict = _features_from_result(r, ob, instrument, df_ltf=df_ltf, df_d1=df_d1, mss_setups=mss_setups)
    X = pd.DataFrame([[feat_dict.get(f, 0) for f in features]], columns=features)
    return float(model.predict_proba(X)[0, 1])


def get_dynamic_threshold(instrument: str, balance: float | None = None) -> float:
    """Seuil ML V5 (user 2026-05-20) : 0.75 partout (WR ~80% min, qualite maximale).

    V5 modeles : AUC 0.819-0.836, WR @0.75 entre 76% et 87% sur OOS 12 mois.
    User accepte volume reduit (cap par actif/jour gere en amont) pour
    maximiser WR et reduire drawdown.
    """
    return ML_THRESHOLDS.get(instrument, DEFAULT_THRESHOLD)


def should_take(r, ob, instrument: str, balance: float | None = None,
                df_ltf=None, df_d1=None, mss_setups=None) -> tuple[bool, float]:
    """Decide si on prend le trade selon le ML.

    Args:
        balance: compte actuel (optionnel). Si fourni, applique seuil dynamique.
        df_ltf, df_d1, mss_setups: passes a _features_from_result (V5 features).

    Returns:
        (accept, proba) : accept=True si proba >= threshold pour l'actif.
    """
    proba = predict_proba(r, ob, instrument, df_ltf=df_ltf, df_d1=df_d1, mss_setups=mss_setups)
    threshold = get_dynamic_threshold(instrument, balance)
    return proba >= threshold, proba
