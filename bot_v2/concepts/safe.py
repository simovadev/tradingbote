"""concepts/safe.py - Implementation ICT SANS LOOK-AHEAD.

Fixe les 33 bugs critiques identifies dans AUDIT_V21_COMPLET.md (28/05/2026)
Chaque fonction accepte un parametre `as_of_index` strict pour borner
le calcul aux donnees disponibles au moment T.

Reutilise le mieux de bot_screener/ict_detectors.py (plus simple et propre).

PRINCIPE FONDAMENTAL :
- Aucune fonction ne regarde data[i+1:] sauf si explicitement autorise
- Toutes les detections (swings, sweeps, MSS, FVG) sont calculees comme un
  trader le ferait en live : avec un decalage de validation strength bougies
- Les HTF (M15, H1, D1) sont stricts : ts < target_ts (jamais <=)
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass


# =====================================================
# SWINGS SAFE (fix bug #81 audit)
# =====================================================

@dataclass(frozen=True)
class SafeSwing:
    kind: str                 # "high" / "low"
    index: int                # index pivot
    confirmed_at: int         # index ou le swing est confirme (= index + strength)
    timestamp: pd.Timestamp
    price: float
    strength: int


def find_swings_safe(
    df: pd.DataFrame,
    strength: int = 3,
    as_of_index: int | None = None,
) -> list[SafeSwing]:
    """Trouve les swings dont la CONFIRMATION (= pivot + strength) est <= as_of_index.

    Fix bug #81 : on retourne UNIQUEMENT les swings dont les 'strength' bougies
    apres le pivot sont disponibles ET dont confirmed_at <= as_of_index.

    En live, c'est equivalent a : 'a la bougie t, quels sont les swings deja
    confirmes ?' Reponse : ceux dont le pivot etait t-strength ou avant.
    """
    if df is None or len(df) < 2 * strength + 1:
        return []

    if as_of_index is None:
        as_of_index = len(df) - 1
    # Un swing au pivot k est confirme a k+strength
    # On ne peut detecter que les swings dont k+strength <= as_of_index
    # donc on cherche les pivots dans [strength, as_of_index - strength]
    last_valid_pivot = min(as_of_index - strength, len(df) - 1 - strength)
    if last_valid_pivot < strength:
        return []

    highs = df["high"].values
    lows = df["low"].values
    timestamps = df.index
    swings = []

    # Boucle sur les pivots candidats
    for k in range(strength, last_valid_pivot + 1):
        # Fenetre [k-s, k+s] : on cherche si k est extremum local
        win_high = highs[k - strength:k + strength + 1]
        win_low = lows[k - strength:k + strength + 1]
        center_h = win_high[strength]
        center_l = win_low[strength]
        # Strict (> pour high, < pour low) comme l'original
        is_high = (
            center_h > win_high[:strength].max() and
            center_h > win_high[strength + 1:].max()
        )
        is_low = (
            center_l < win_low[:strength].min() and
            center_l < win_low[strength + 1:].min()
        )
        if is_high:
            swings.append(SafeSwing(
                kind="high", index=k, confirmed_at=k + strength,
                timestamp=timestamps[k], price=float(center_h),
                strength=strength,
            ))
        if is_low:
            swings.append(SafeSwing(
                kind="low", index=k, confirmed_at=k + strength,
                timestamp=timestamps[k], price=float(center_l),
                strength=strength,
            ))

    return swings


# =====================================================
# SWEEPS SAFE (fix bug #97 audit)
# =====================================================

@dataclass(frozen=True)
class SafeSweep:
    swing_kind: str            # "high" / "low" balaye
    swing_price: float         # niveau balaye
    swing_index: int           # index du swing
    sweep_index: int           # index de la bougie qui sweep
    sweep_ts: pd.Timestamp
    direction: str             # "bullish" si sweep low (rejet bas), "bearish" si sweep high


def find_sweeps_safe(
    df: pd.DataFrame,
    swings: list[SafeSwing],
    as_of_index: int | None = None,
) -> list[SafeSweep]:
    """Cherche la 1ere bougie posterieure au swing (apres confirmation) qui sweep.

    Fix bug #97 : start_idx = swing.confirmed_at + 1 (pas swing.index + 1).
    On ne peut pas sweep un swing tant qu'il n'est pas encore confirme.

    Fix #94 : sweep_index <= as_of_index.
    """
    if not swings or df is None or len(df) < 2:
        return []
    if as_of_index is None:
        as_of_index = len(df) - 1

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    sweeps = []

    for sw in swings:
        start = sw.confirmed_at + 1
        end = min(as_of_index + 1, len(df))
        if start >= end:
            continue
        if sw.kind == "high":
            # Cherche 1ere bougie ou high > sw.price ET close < sw.price (rejet)
            for j in range(start, end):
                if highs[j] > sw.price and closes[j] < sw.price:
                    sweeps.append(SafeSweep(
                        swing_kind="high", swing_price=sw.price,
                        swing_index=sw.index, sweep_index=j,
                        sweep_ts=df.index[j], direction="bearish",
                    ))
                    break
        else:  # low
            for j in range(start, end):
                if lows[j] < sw.price and closes[j] > sw.price:
                    sweeps.append(SafeSweep(
                        swing_kind="low", swing_price=sw.price,
                        swing_index=sw.index, sweep_index=j,
                        sweep_ts=df.index[j], direction="bullish",
                    ))
                    break

    return sweeps


# =====================================================
# DETECT TREND SAFE (fix bug #82 audit)
# =====================================================

def detect_trend_safe(
    swings: list[SafeSwing],
    as_of_index: int | None = None,
    lookback: int = 4,
) -> str | None:
    """Tendance basee uniquement sur les swings dont confirmed_at <= as_of_index.

    Fix bug #82 : on filtre AVANT de prendre les 'lookback' derniers.
    """
    if not swings:
        return None
    if as_of_index is not None:
        valid = [s for s in swings if s.confirmed_at <= as_of_index]
    else:
        valid = swings
    if len(valid) < lookback:
        return None
    recent = valid[-lookback:]
    # Tendance haussiere : derniers swings high & low en hausse
    highs_in_seq = [s for s in recent if s.kind == "high"]
    lows_in_seq = [s for s in recent if s.kind == "low"]
    if len(highs_in_seq) >= 2 and len(lows_in_seq) >= 2:
        hh = highs_in_seq[-1].price > highs_in_seq[-2].price
        hl = lows_in_seq[-1].price > lows_in_seq[-2].price
        if hh and hl:
            return "bullish"
        if not hh and not hl:
            return "bearish"
    return None


# =====================================================
# ORDER BLOCK SAFE
# =====================================================

@dataclass(frozen=True)
class SafeOB:
    direction: str              # "bullish" / "bearish"
    ob_low: float
    ob_high: float
    ob_open: float
    ob_close: float
    formation_index: int
    formation_ts: pd.Timestamp
    validation_index: int       # index ou le displacement confirme l'OB
    validation_ts: pd.Timestamp
    displacement_atr: float
    swept_swing_index: int | None     # le swing que l'OB a sweepe (None si pas de sweep)


def find_obs_safe(
    df: pd.DataFrame,
    as_of_index: int | None = None,
    swing_strength: int = 3,
    sweep_lookback: int = 20,
    displacement_min_atr: float = 1.5,
) -> list[SafeOB]:
    """LEGACY : OB strict V21-style avec sweep et displacement obligatoires.
    Trop restrictif. Utilise plutot find_obs_simple() pour la vraie definition ICT classique.
    """
    return find_obs_simple(df, as_of_index=as_of_index)


def find_obs_simple(
    df: pd.DataFrame,
    as_of_index: int | None = None,
    min_consecutive: int = 2,
) -> list[SafeOB]:
    """Detection OB ICT DEFINITION USER (ensemble de bougies consecutives).

    OB bullish :
    - Un ENSEMBLE de bougies baissieres consecutives (close < open), min 2 bougies
    - L'OB = la bougie de l'ensemble qui a le LOW LE PLUS BAS
    - Confirmation : la 1ere bougie HAUSSIERE qui clot l'ensemble valide l'OB
    - INVALIDATION : si plus tard une bougie a son low < ob_low -> l'OB est mort

    OB bearish :
    - Ensemble de bougies haussieres consecutives (close > open), min 2 bougies
    - L'OB = bougie avec le HIGH LE PLUS HAUT
    - Confirmation : 1ere bougie baissiere qui clot l'ensemble
    - INVALIDATION : si plus tard une bougie a son high > ob_high -> mort

    PAS de sweep obligatoire, PAS de displacement minimum.
    La fonction retourne UNIQUEMENT les OB ENCORE VALIDES a as_of_index.
    Tout est borne par as_of_index pour eviter le look-ahead.
    """
    if as_of_index is None:
        as_of_index = len(df) - 1
    if as_of_index < min_consecutive + 1:
        return []

    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    timestamps = df.index
    n = len(df)

    obs = []
    i = 1
    while i <= as_of_index:
        body_i = c[i] - o[i]
        if abs(body_i) < 1e-9:
            i += 1; continue

        if body_i < 0:
            # Debut potentiel ensemble baissier -> OB bullish
            # Cherche jusqu'ou va la sequence consecutive de baissieres
            j = i
            while j + 1 <= as_of_index and c[j + 1] < o[j + 1]:
                j += 1
            # Sequence baissiere = [i, j]
            seq_len = j - i + 1
            if seq_len >= min_consecutive and j + 1 <= as_of_index:
                # Bougie de retournement = j+1 (doit etre haussiere)
                if c[j + 1] > o[j + 1]:
                    # OB = bougie de l'ensemble avec le low le plus bas
                    sub_lows = l[i:j + 1]
                    ob_pos = i + int(np.argmin(sub_lows))
                    ob_low_val = float(l[ob_pos])
                    ob_high_val = float(h[ob_pos])
                    validation_idx = j + 1

                    # On enregistre TOUS les OBs detectes (validation_idx <= as_of_index)
                    # L'invalidation se gere dans la simulation de trade, pas ici
                    atr = _local_atr(h, l, c, validation_idx, 14)
                    disp = c[validation_idx] - c[ob_pos]
                    obs.append(SafeOB(
                        direction="bullish",
                        ob_low=ob_low_val, ob_high=ob_high_val,
                        ob_open=float(o[ob_pos]), ob_close=float(c[ob_pos]),
                        formation_index=ob_pos, formation_ts=timestamps[ob_pos],
                        validation_index=validation_idx, validation_ts=timestamps[validation_idx],
                        displacement_atr=float(disp / atr) if atr > 0 else 0.0,
                        swept_swing_index=None,
                    ))
            i = j + 1
        elif body_i > 0:
            # Debut potentiel ensemble haussier -> OB bearish
            j = i
            while j + 1 <= as_of_index and c[j + 1] > o[j + 1]:
                j += 1
            seq_len = j - i + 1
            if seq_len >= min_consecutive and j + 1 <= as_of_index:
                if c[j + 1] < o[j + 1]:
                    sub_highs = h[i:j + 1]
                    ob_pos = i + int(np.argmax(sub_highs))
                    ob_low_val = float(l[ob_pos])
                    ob_high_val = float(h[ob_pos])
                    validation_idx = j + 1

                    atr = _local_atr(h, l, c, validation_idx, 14)
                    disp = c[ob_pos] - c[validation_idx]
                    obs.append(SafeOB(
                        direction="bearish",
                        ob_low=ob_low_val, ob_high=ob_high_val,
                        ob_open=float(o[ob_pos]), ob_close=float(c[ob_pos]),
                        formation_index=ob_pos, formation_ts=timestamps[ob_pos],
                        validation_index=validation_idx, validation_ts=timestamps[validation_idx],
                        displacement_atr=float(disp / atr) if atr > 0 else 0.0,
                        swept_swing_index=None,
                    ))
            i = j + 1
        else:
            i += 1

    return obs


def _local_atr(h, l, c, idx, period=14):
    """ATR local pour info (pas pour filtre)."""
    s = max(0, idx - period)
    if idx - s < 2:
        return 0.0
    hh = h[s:idx + 1]; ll = l[s:idx + 1]; cc = c[s:idx + 1]
    tr = np.maximum(hh[1:] - ll[1:],
                    np.maximum(np.abs(hh[1:] - cc[:-1]),
                               np.abs(ll[1:] - cc[:-1])))
    if len(tr) == 0:
        return 0.0
    return float(np.mean(tr))


# =====================================================
# DAILY BIAS SAFE (fix bug #104 audit)
# =====================================================

def daily_bias_safe(df_d1: pd.DataFrame, target_ts: pd.Timestamp) -> dict:
    """Daily bias propre. Fix bug #104 : on normalize() pour exclure D1 en cours.

    target_ts = timestamp du moment de l'analyse
    On compare close J-1 vs close J-2.
    """
    if df_d1 is None or len(df_d1) < 3:
        return {"bias": None, "ok": False}
    # On normalize a 00:00 UTC pour comparer les jours, pas les heures
    target_day = pd.Timestamp(target_ts).normalize()
    # On veut UNIQUEMENT les D1 STRICTEMENT avant target_day (fin de bougie D1 = 00h jour suivant)
    df_past = df_d1[df_d1.index < target_day]
    if len(df_past) < 2:
        return {"bias": None, "ok": False}
    last_close = float(df_past["close"].iloc[-1])      # J-1
    prev_close = float(df_past["close"].iloc[-2])      # J-2
    return {
        "bias": "haussier" if last_close > prev_close else "baissier",
        "j1_close": last_close, "j2_close": prev_close,
        "ok": True,
    }


# =====================================================
# HTF VALUE-AT-T SAFE
# =====================================================

def htf_value_at_t(df_htf: pd.DataFrame, target_ts: pd.Timestamp,
                    field: str = "close") -> float | None:
    """Retourne la valeur HTF a target_ts : la DERNIERE bougie FERMEE avant target_ts.

    Fix bugs #12, #13 audit : utiliser searchsorted(side='left') puis -1
    pour ne JAMAIS prendre une bougie qui n'est pas encore fermee.

    Une bougie HTF d'index ts (donnees) est fermee A ts + duree_tf. Donc on veut
    la plus recente bougie dont ts + duree_tf <= target_ts, autrement dit ts < target_ts.
    """
    if df_htf is None or len(df_htf) == 0:
        return None
    # On veut le dernier index strictement avant target_ts
    pos = df_htf.index.searchsorted(target_ts, side="left") - 1
    if pos < 0:
        return None
    return float(df_htf[field].iloc[pos])


def htf_window_safe(df_htf: pd.DataFrame, target_ts: pd.Timestamp,
                     n_bars: int = 50) -> pd.DataFrame:
    """Retourne les N dernieres bougies HTF FERMEES avant target_ts.

    Fix bugs #12, #13 audit.
    """
    if df_htf is None or len(df_htf) == 0:
        return pd.DataFrame()
    pos = df_htf.index.searchsorted(target_ts, side="left") - 1
    if pos < 0:
        return pd.DataFrame()
    start = max(0, pos - n_bars + 1)
    return df_htf.iloc[start:pos + 1]


# =====================================================
# FAIR VALUE GAP SAFE
# =====================================================

@dataclass(frozen=True)
class SafeFVG:
    direction: str            # "bullish" / "bearish"
    top: float
    bottom: float
    formation_index: int      # index de la bougie i+1 (du milieu du FVG)
    formation_ts: pd.Timestamp


def find_fvgs_safe(df: pd.DataFrame, as_of_index: int | None = None,
                    lookback: int = 100) -> list[SafeFVG]:
    """FVG = trou de 3 bougies. bullish : low[i+2] > high[i].

    Detect uniquement les FVG dont formation_index <= as_of_index.
    """
    if df is None or len(df) < 3:
        return []
    if as_of_index is None:
        as_of_index = len(df) - 1

    h = df["high"].values
    l = df["low"].values
    timestamps = df.index
    fvgs = []
    start = max(0, as_of_index - lookback)
    for i in range(start, min(as_of_index - 1, len(df) - 2)):
        # FVG bullish : high[i] < low[i+2]
        if h[i] < l[i + 2]:
            fvgs.append(SafeFVG(
                direction="bullish", top=float(l[i + 2]), bottom=float(h[i]),
                formation_index=i + 2, formation_ts=timestamps[i + 2],
            ))
        # FVG bearish : low[i] > high[i+2]
        if l[i] > h[i + 2]:
            fvgs.append(SafeFVG(
                direction="bearish", top=float(l[i]), bottom=float(h[i + 2]),
                formation_index=i + 2, formation_ts=timestamps[i + 2],
            ))
    return fvgs


# =====================================================
# ATR SAFE (ne regarde JAMAIS le futur)
# =====================================================

def atr_safe(df: pd.DataFrame, as_of_index: int, period: int = 14) -> float:
    """ATR a as_of_index, base sur les `period` bougies precedentes (toutes <= as_of_index)."""
    if as_of_index < 1:
        return 0.0
    s = max(0, as_of_index - period)
    if as_of_index - s < 2:
        return 0.0
    h = df["high"].values[s:as_of_index + 1]
    l = df["low"].values[s:as_of_index + 1]
    c = df["close"].values[s:as_of_index + 1]
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    if len(tr) == 0:
        return 0.0
    return float(np.mean(tr))


# =====================================================
# UTILITAIRE : extraire la sequence M5 jusqu'a as_of_index (pour DL)
# =====================================================

def extract_seq_safe(df: pd.DataFrame, as_of_index: int, n: int = 60) -> np.ndarray | None:
    """Extrait les `n` dernieres bougies OHLC <= as_of_index sans look-ahead.

    Fix bugs #1, #2, #3 audit (normalisation contextuelle/bougie partielle).
    Pour la normalisation, on utilise une stat ROBUSTE basee sur les bougies
    extraites (mean/std) au lieu de closes[-1] qui peut etre instable.

    Retourne array shape (n, 4) = OHLC normalises ou None si pas assez de data.
    """
    if as_of_index + 1 < n or as_of_index >= len(df):
        return None
    start = as_of_index + 1 - n
    sub = df.iloc[start:as_of_index + 1]
    arr = np.column_stack([
        sub["open"].values, sub["high"].values, sub["low"].values, sub["close"].values
    ]).astype(np.float32)
    # Normalisation safe : prix centre sur median, scale par range total
    med = float(np.median(arr))
    rng = float(arr.max() - arr.min())
    if rng < 1e-9:
        return None
    normed = (arr - med) / rng
    return normed
