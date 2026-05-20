"""ML filter — applique le modele LightGBM comme filtre final apres Vizion.

Decision user 2026-05-16 :
- Seuil 0.50 : plus de trades (~25/mois sur 5 actifs), WR ~58% (vs 64% a 0.55).
- Garde la couche Vizion comme detection, le ML filtre la qualite.
"""
from __future__ import annotations

import json
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

# V4 (user 2026-05-20) : Models V3.5 Admiral 8 ans pour 14 actifs
# WR @ 0.70 entre 73% et 80%, AUC 0.78-0.81
_MBASE = "c:/Users/Shadow/TradingBot/bot_v2"
_MODELS_BY_INSTRUMENT: dict[str, Path] = {
    "XAUUSD": Path(f"{_MBASE}/ml_model_XAUUSD_admiral_v3_5.pkl"),
    "NAS100": Path(f"{_MBASE}/ml_model_NAS100_admiral_v3_5.pkl"),
    "GER40":  Path(f"{_MBASE}/ml_model_GER40_admiral_v3_5.pkl"),
    "BTCUSD": Path(f"{_MBASE}/ml_model_BTCUSD_admiral_v3_5.pkl"),
    "EURUSD": Path(f"{_MBASE}/ml_model_EURUSD_admiral_v3_5.pkl"),
    "GBPUSD": Path(f"{_MBASE}/ml_model_GBPUSD_admiral_v3_5.pkl"),
    "AUDUSD": Path(f"{_MBASE}/ml_model_AUDUSD_admiral_v3_5.pkl"),
    "USDJPY": Path(f"{_MBASE}/ml_model_USDJPY_admiral_v3_5.pkl"),
    "SP500":  Path(f"{_MBASE}/ml_model_SP500_admiral_v3_5.pkl"),
    "DJ30":   Path(f"{_MBASE}/ml_model_DJ30_admiral_v3_5.pkl"),
    "UK100":  Path(f"{_MBASE}/ml_model_UK100_admiral_v3_5.pkl"),
    "FRA40":  Path(f"{_MBASE}/ml_model_FRA40_admiral_v3_5.pkl"),
    "USDCAD": Path(f"{_MBASE}/ml_model_USDCAD_admiral_v3_5.pkl"),
    "USDCHF": Path(f"{_MBASE}/ml_model_USDCHF_admiral_v3_5.pkl"),
    # JP225 desactive (ecart Duka/Vantage trop grand)
}
_FEATURES_BY_INSTRUMENT: dict[str, Path] = {
    "XAUUSD": Path(f"{_MBASE}/ml_features_XAUUSD_admiral_v3_5.json"),
    "NAS100": Path(f"{_MBASE}/ml_features_NAS100_admiral_v3_5.json"),
    "GER40":  Path(f"{_MBASE}/ml_features_GER40_admiral_v3_5.json"),
    "BTCUSD": Path(f"{_MBASE}/ml_features_BTCUSD_admiral_v3_5.json"),
    "EURUSD": Path(f"{_MBASE}/ml_features_EURUSD_admiral_v3_5.json"),
    "GBPUSD": Path(f"{_MBASE}/ml_features_GBPUSD_admiral_v3_5.json"),
    "AUDUSD": Path(f"{_MBASE}/ml_features_AUDUSD_admiral_v3_5.json"),
    "USDJPY": Path(f"{_MBASE}/ml_features_USDJPY_admiral_v3_5.json"),
    "SP500":  Path(f"{_MBASE}/ml_features_SP500_admiral_v3_5.json"),
    "DJ30":   Path(f"{_MBASE}/ml_features_DJ30_admiral_v3_5.json"),
    "UK100":  Path(f"{_MBASE}/ml_features_UK100_admiral_v3_5.json"),
    "FRA40":  Path(f"{_MBASE}/ml_features_FRA40_admiral_v3_5.json"),
    "USDCAD": Path(f"{_MBASE}/ml_features_USDCAD_admiral_v3_5.json"),
    "USDCHF": Path(f"{_MBASE}/ml_features_USDCHF_admiral_v3_5.json"),
}

# Models specifiques par (INSTRUMENT, TF) — user 2026-05-16
# Fallback : modele instrument generique, puis modele TF generique
_MODELS_BY_INST_TF: dict[tuple[str, str], Path] = {
    ("NAS100", "M5"): Path("c:/Users/Shadow/TradingBot/bot_v2/ml_model_NAS100_M5.pkl"),
}
_FEATURES_BY_INST_TF: dict[tuple[str, str], Path] = {
    ("NAS100", "M5"): Path("c:/Users/Shadow/TradingBot/bot_v2/ml_features_NAS100_M5.json"),
}

# V3.5 thresholds (user 2026-05-20) - calibre sur OOS 12 mois Admiral 8 ans
# Strategie : seuil 0.70 minimum partout (WR >= 73% partout), montee 0.75 si
# WR <75% au seuil 0.70.
ML_THRESHOLDS: dict[str, float] = {
    "XAUUSD": 0.55,  # TEST WR 73.0% @0.70, AUC 0.778
    "NAS100": 0.55,  # TEST WR 76.3% @0.70, AUC 0.813
    "GER40":  0.55,  # AUC 0.804
    "BTCUSD": 0.55,  # TEST WR 73.8% @0.70, AUC 0.810
    "EURUSD": 0.55,  # TEST WR 79.4% @0.70, AUC 0.808
    "GBPUSD": 0.55,  # TEST WR 74.4% @0.70, AUC 0.804
    "AUDUSD": 0.55,  # TEST WR 79.8% @0.70, AUC 0.811
    "USDJPY": 0.55,  # TEST WR 75.0% @0.70, AUC 0.804
    "SP500":  0.55,  # TEST WR 80.5% @0.70, AUC 0.801
    "DJ30":   0.55,  # TEST WR 73.4% @0.70, AUC 0.792
    "UK100":  0.55,  # TEST WR 72.8% @0.70, AUC 0.787
    "FRA40":  0.55,  # TEST WR 75.1% @0.70, AUC 0.799
    "USDCAD": 0.55,  # TEST WR 80.0% @0.70, AUC 0.786
    "USDCHF": 0.55,  # TEST WR 77.7% @0.70, AUC 0.806
    # JP225 desactive (ecart Duka/Vantage trop grand)
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


def _features_from_result(r, ob, instrument: str) -> dict:
    """Reproduit les features du dataset ML a partir d'un PipelineResult Vizion.

    Doit rester synchronise avec _extract_features dans ml_dataset.py.
    """
    setup = r.trade_setup
    conf = r.confluences or []
    conf_text = " ".join(conf)
    return {
        "score": r.score,
        "quality": r.quality.total_quality_score if r.quality else 0,
        "ob_strength": r.quality.ob_strength if r.quality else 0,
        "sweep_strength": r.quality.sweep_strength if r.quality else 0,
        "retest_count": r.quality.retest_count if r.quality else 0,
        "is_unicorn": int(r.quality.is_unicorn) if r.quality else 0,
        "multi_liq_sweep": int(r.quality.multi_liquidity_sweep) if r.quality else 0,
        "rr": setup.rr if setup else 0,
        "risk_points": setup.risk_points if setup else 0,
        "tp_source_htf": int("htf" in (setup.tp_source if setup else "")),
        "tp_source_capped": int("capped" in (setup.tp_source if setup else "")),
        "daily_bias_aligned": int(r.daily_bias_ok is True),
        "daily_bias_neutral": int(r.daily_bias is not None and r.daily_bias.bias == "neutral"),
        "kz_london": int(r.killzone_name == "London"),
        "kz_ny_am": int(r.killzone_name == "NY_AM"),
        "kz_ny_pm": int(r.killzone_name == "NY_PM"),
        "kz_asia": int(r.killzone_name == "Asia"),
        "kz_ny_lunch": int(r.killzone_name == "NY_Lunch"),
        "kz_none": int(r.killzone_name is None),
        "has_sync_fvg": int("OB_FVG_sync" in conf_text),
        "has_smt": int("smt_" in conf_text),
        "has_feu_vert": int("feu_vert" in conf_text),
        "has_breaker_kz": int("breaker_in_KZ" in conf_text),
        "has_mss_fvg": int("MSS_with_FVG" in conf_text),
        "has_grandparent": int("grandparent_ob_" in conf_text),
        "has_po3_dist": int("po3_distribution" in conf_text),
        "has_phase_expansion": int("phase_expansion" in conf_text),
        "has_phase_reversal": int("phase_reversal" in conf_text),
        "has_open_midnight_respect": int("respecte_OpenMidnightNY" in conf_text),
        "ob_group_size": ob.group_size,
        "bars_sweep_to_validation": ob.validation_index - getattr(ob.sweep, "sweep_index", ob.group_start_index),
        "bars_group_to_validation": ob.validation_index - ob.group_start_index,
        "is_bullish": int(ob.direction == "bullish"),
        "is_mss_setup": int(getattr(ob, "is_mss_setup", False)),
        "has_mss_confirmation": 1,
    }


def predict_proba(r, ob, instrument: str) -> float:
    """Retourne la probabilite de WIN [0,1] selon le modele specifique a l'actif."""
    # Multi-asset (user 2026-05-16) : utilise le modele par instrument si dispo
    loaded = load_model_for_instrument(instrument)
    if loaded is not None:
        model, features = loaded
    else:
        model, features = _load()
    feat_dict = _features_from_result(r, ob, instrument)
    X = pd.DataFrame([[feat_dict.get(f, 0) for f in features]], columns=features)
    return float(model.predict_proba(X)[0, 1])


def get_dynamic_threshold(instrument: str, balance: float | None = None) -> float:
    """Seuil ML (user 2026-05-20 : revert au seuil 0.70 fixe apres test).

    Strategie 0.55 sprint / 0.70 conso revertee : on garde 0.70 partout
    pour qualite maximale (WR 73-80% sur OOS).
    """
    return ML_THRESHOLDS.get(instrument, DEFAULT_THRESHOLD)


def should_take(r, ob, instrument: str, balance: float | None = None) -> tuple[bool, float]:
    """Decide si on prend le trade selon le ML.

    Args:
        balance: compte actuel (optionnel). Si fourni, applique seuil dynamique.

    Returns:
        (accept, proba) : accept=True si proba >= threshold pour l'actif.
    """
    proba = predict_proba(r, ob, instrument)
    threshold = get_dynamic_threshold(instrument, balance)
    return proba >= threshold, proba
