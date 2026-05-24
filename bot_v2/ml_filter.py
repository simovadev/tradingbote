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
# V14 (2026-05-25) : seuil descendu 0.70 -> 0.60.
# OOS V14 a seuil 0.60 : WR moyen 66.3% / 1169 trades sur 6 mois (~9/jour).
# Expectancy theorique (RR=2) = +0.99R/trade vs +1.26R a 0.70.
# Mais volume x3.2 -> total expectancy/jour +160% vs 0.70.
# Avec RR reel V14 (~3 via EXEC pres SL), expectancy/trade = +1.65R.
# Cf decision user 2026-05-25 apres analyse OOS V14 complete.
ML_THRESHOLDS: dict[str, float] = {
    "XAUUSD": 0.60,
    "NAS100": 0.60,
    "GER40":  0.60,
    "BTCUSD": 0.60,
    "EURUSD": 0.60,
    "GBPUSD": 0.60,
    "AUDUSD": 0.60,
    "USDJPY": 0.60,
    "SP500":  0.60,
    "DJ30":   0.60,
    "UK100":  0.60,
    "FRA40":  0.60,
    "USDCAD": 0.60,
    "USDCHF": 0.60,
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


def _compute_atr_regime_v13(df_ltf, idx, current_kz_name):
    """ATR de la killzone courante / moyenne ATR des 5 dernieres MEMES KZ.

    Retourne 1.0 (neutre) si pas assez d'historique ou erreur.

    Args:
        df_ltf: DataFrame M1 avec colonnes high/low et index timestamp.
        idx: int, position de validation de l'OB dans df_ltf.
        current_kz_name: nom de la killzone courante (str ou None).
                         Si None : fallback fenetre 60 M1 a la meme heure UTC.

    Returns:
        float : ratio atr_now / atr_typical, ou 1.0 si indeterminable.
    """
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


def _features_from_result(r, ob, instrument: str, df_ltf=None, df_d1=None, mss_setups=None) -> dict:
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
        # Killzone one-hot
        "kz_london": int(r.killzone_name == "London"),
        "kz_ny_am": int(r.killzone_name == "NY_AM"),
        "kz_ny_pm": int(r.killzone_name == "NY_PM"),
        "kz_asia": int(r.killzone_name == "Asia"),
        "kz_ny_lunch": int(r.killzone_name == "NY_Lunch"),
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
        "bars_sweep_to_validation": ob.validation_index - getattr(ob.sweep, "sweep_index", ob.group_start_index),
        "bars_group_to_validation": ob.validation_index - ob.group_start_index,
        "is_bullish": int(ob.direction == "bullish"),
    }

    # Features V3 (temps)
    ts = ob.validation_ts
    f["hour_of_day"] = int(ts.hour)
    f["day_of_week"] = int(ts.dayofweek)
    f["minutes_into_killzone"] = int(ts.hour * 60 + ts.minute) % 60 if r.killzone_name else -1

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
        sl20 = df_ltf.iloc[idx - 20:idx]
        rng = (sl20["high"] - sl20["low"])
        body = (sl20["close"] - sl20["open"]).abs()
        f["body_ratio_recent"] = float((body / rng.replace(0, 1e-9)).mean())
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
    f["atr_regime"] = _compute_atr_regime_v13(df_ltf, idx, r.killzone_name)

    # --- Groupe E : distance aux niveaux daily en ATR (pas en %) ---
    atr_ref = f.get("atr_at_setup", 0.0) or 0.0
    if atr_ref > 0 and pdh and entry_price:
        f["dist_pdh_atr"] = abs(entry_price - pdh) / atr_ref
    else:
        f["dist_pdh_atr"] = 0.0
    if atr_ref > 0 and pdl and entry_price:
        f["dist_pdl_atr"] = abs(entry_price - pdl) / atr_ref
    else:
        f["dist_pdl_atr"] = 0.0

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
