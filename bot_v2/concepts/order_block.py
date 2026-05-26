"""Order Block — detection STRICTE Vizion (bible §2 + decision user 2026-05-15).

Definition stricte :
- **OB bullish** = N bougies BAISSIERES CONSECUTIVES (N>=1) qui prennent une
  liquidite externe (sweep d'un swing low). Validation = cloture corps de bougie
  AU-DESSUS du HIGH du groupe.
  ⚠ Aucune bougie haussiere "intruse" toleree au milieu du groupe (decision user).

- **OB bearish** = symetrique avec N bougies HAUSSIERES consecutives, sweep d'un
  swing high, cloture corps SOUS le LOW du groupe.

Stop Loss : MECHE (decision user 2026-05-15).
- OB bullish : SL sous la meche basse du groupe (low du groupe).
- OB bearish : SL au-dessus de la meche haute du groupe (high du groupe).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from bot_v2.concepts.liquidity import Sweep, Swing, find_sweeps, find_swings

# V14 (2026-05-25) : mode global via env var. Si OB_VALIDATION_MODE=mitigation,
# tous les appels a detect_order_blocks utilisent ce mode (sauf override explicite).
# Permet de switcher V13 (BOS) <-> V14 (mitigation) sans toucher tous les call sites.
# Lu a chaque appel (pas au module import) pour que load_model V14 puisse l'activer
# a runtime.
def _get_global_validation_mode() -> str:
    return os.environ.get("OB_VALIDATION_MODE", "bos")


OBDirection = Literal["bullish", "bearish"]


@dataclass(frozen=True)
class OrderBlock:
    """Un Order Block valide selon les regles Vizion strictes."""
    direction: OBDirection                # "bullish" | "bearish"
    # Bornes du GROUPE de bougies inverses (l'OB lui-meme)
    group_start_index: int                # iloc premiere bougie du groupe
    group_end_index: int                  # iloc derniere bougie du groupe (inclusif)
    group_start_ts: pd.Timestamp
    group_end_ts: pd.Timestamp
    # Niveaux de prix de l'OB
    ob_high: float                        # plus haut du groupe (meches incluses)
    ob_low: float                         # plus bas du groupe (meches incluses)
    ob_open: float                        # open de la 1ere bougie du groupe
    ob_close: float                       # close de la derniere bougie du groupe
    # La bougie de VALIDATION (clos corps au-dessus du high pour bullish)
    validation_index: int
    validation_ts: pd.Timestamp
    validation_close: float
    # Sweep qui a precede / declenche l'OB (liquidite externe prise)
    sweep: Sweep
    # Taille du groupe en bougies
    group_size: int


def _is_bearish(row) -> bool:
    """Une bougie est baissiere si close < open."""
    return row["close"] < row["open"]


def _is_bullish(row) -> bool:
    return row["close"] > row["open"]


def detect_order_blocks(
    df: pd.DataFrame,
    swings: list[Swing] | None = None,
    sweeps: list[Sweep] | None = None,
    swing_strength: int = 2,
    max_group_size: int = 5,
    min_group_size: int = 1,
    max_bars_after_sweep: int = 30,
    min_sweep_depth_atr: float = 0.0,
    validation_mode: str = "bos",
) -> list[OrderBlock]:
    """Detecte tous les OB valides dans le DataFrame.

    Logique :
    1. Pour chaque sweep (prise de liquidite) :
       - Si sweep BULLISH (low pris) -> on cherche un OB bullish
         = groupe de bougies BAISSIERES consecutives juste avant le retournement,
           puis cloture corps au-dessus du HIGH du groupe.
       - Si sweep BEARISH (high pris) -> symetrique.
    2. Le groupe doit etre 100% du bon sens (decision user : strict, pas d'intruse).
    3. La validation doit arriver dans les `max_bars_after_sweep` bougies apres le sweep.

    Args:
        df: DataFrame avec colonnes open, high, low, close.
        swings, sweeps: optionnels, recalcules sinon.
        swing_strength: passage a find_swings si recalcul.
        max_group_size: garde-fou (bible : pas plus de 5 bougies sinon = range).
        max_bars_after_sweep: validation doit arriver vite, sinon OB ignore.
    """
    if swings is None:
        swings = find_swings(df, strength=swing_strength)
    if sweeps is None:
        # V18.2 : min_sweep_depth optimisable via env var
        _depth = float(os.environ.get("V18_MIN_SWEEP_DEPTH", str(min_sweep_depth_atr)))
        sweeps = find_sweeps(df, swings, min_depth_atr=_depth)

    # Si l'appelant n'a pas force le mode, on lit l'env var globale a CHAQUE appel
    # (defaut "bos" = comportement V13). Lu dynamiquement pour que load_model V14
    # puisse l'activer apres le module import.
    effective_mode = validation_mode if validation_mode != "bos" else _get_global_validation_mode()

    obs: list[OrderBlock] = []

    for sweep in sweeps:
        if sweep.direction == "bullish":
            ob = _try_bullish_ob(df, sweep, max_group_size, max_bars_after_sweep, min_group_size,
                                 validation_mode=effective_mode)
        else:
            ob = _try_bearish_ob(df, sweep, max_group_size, max_bars_after_sweep, min_group_size,
                                 validation_mode=effective_mode)
        if ob is not None:
            obs.append(ob)

    return deduplicate_obs(obs)


def deduplicate_obs(obs: list[OrderBlock]) -> list[OrderBlock]:
    """Supprime les OB doublons (meme groupe de bougies, meme direction).

    On considere 2 OB doublons s'ils ont :
    - Meme direction
    - Meme group_start_index et group_end_index
    OU
    - Chevauchement complet et meme validation_index

    On garde le premier (= celui qui vient du sweep le plus pertinent : le 1er detecte).
    """
    seen: set[tuple] = set()
    out: list[OrderBlock] = []
    for ob in obs:
        # Cle unique = direction + bornes du groupe + validation
        key = (ob.direction, ob.group_start_index, ob.group_end_index, ob.validation_index)
        if key in seen:
            continue
        seen.add(key)
        out.append(ob)
    return out


def _try_bullish_ob(
    df: pd.DataFrame,
    sweep: Sweep,
    max_group_size: int,
    max_bars_after_sweep: int,
    min_group_size: int = 1,
    validation_mode: str = "bos",
) -> OrderBlock | None:
    """Cherche un OB bullish autour d'un sweep bullish (low pris).

    Structure attendue :
        [N bougies baissieres consecutives] [bougie qui sweep le low] [validation : close > high du groupe]
    OU
        [N bougies baissieres incluant la sweep] [validation]

    Note Vizion 5fv2MjuPKE4 : c'est souvent LA bougie qui sweep qui FORME la fin
    du groupe baissier. Sa meche depasse le swing low, mais elle CLOS au-dessus
    (= rejet). Cette bougie peut etre baissiere ou haussiere.

    On retient : le GROUPE = bougies baissieres consecutives terminant la
    descente. La bougie de sweep peut etre dedans (si baissiere) ou juste apres.
    """
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values

    sweep_idx = sweep.sweep_index

    # ===== V19 ICT PUR (ICT 2024, version "cluster" = ensemble de bougies) =====
    # Definition ICT pratique (cf Soufiane + sources ICT secondaires) :
    #   OB BULLISH = SERIE de bougies BAISSIERES consecutives avant impulsion haussiere.
    #   Les "last bearish candle OR series of candles before a significant bullish move".
    #
    # Validation ICT "engulfing in 1 candle" (LA difference cle vs V18) :
    #   La bougie SUIVANTE doit en UNE SEULE bougie :
    #   - sweep : low descend SOUS le low du groupe (mèches incluses)
    #   - engulf : close cloture > high CORPS du groupe (max(open,close))
    #   - etre HAUSSIERE (close > open)
    #
    # Bornes ICT : CORPS uniquement (cohorent avec engulf body-to-body)
    #   ob_high = MAX(open, close) sur tout le groupe
    #   ob_low  = MIN(open, close) sur tout le groupe
    #
    # Difference cle vs V18 :
    #   V18 : validation par close > ob_high APRES N bougies, rejection si meche < ob_low
    #   V19 : validation par close > ob_high EN UNE BOUGIE (la bougie d'engulf = sweep_idx)
    if validation_mode == "v19_ict":
        engulf_idx = sweep_idx
        # La bougie d'engulf doit etre HAUSSIERE
        if closes[engulf_idx] <= opens[engulf_idx]:
            return None
        # Trouve le GROUPE de baissieres consecutives strictement AVANT engulf_idx
        # (la derniere baissiere doit etre a engulf_idx - 1)
        if engulf_idx - 1 < 0:
            return None
        if closes[engulf_idx - 1] >= opens[engulf_idx - 1]:
            return None  # pas de baissiere juste avant -> pas d'OB
        group_end = engulf_idx - 1
        group_start = group_end
        while group_start > 0 and closes[group_start - 1] < opens[group_start - 1]:
            group_start -= 1
        group_size = group_end - group_start + 1
        # min 1 bougie pour ICT (a la difference de V18 qui exige 2)
        if group_size < 1:
            return None
        # Bornes ICT cluster (corps uniquement)
        ob_high_ict = float(max(
            max(opens[i], closes[i]) for i in range(group_start, group_end + 1)
        ))
        ob_low_ict = float(min(
            min(opens[i], closes[i]) for i in range(group_start, group_end + 1)
        ))
        if ob_high_ict <= ob_low_ict:
            return None
        # CHECK 1 : sweep — low[engulf] doit descendre sous low_corps_min (ou meche du groupe)
        # On utilise low du groupe (meches incluses) pour le sweep ICT classique
        group_low_wick = float(min(lows[i] for i in range(group_start, group_end + 1)))
        if lows[engulf_idx] >= group_low_wick:
            return None
        # CHECK 2 : engulf — close[engulf] > ob_high (max corps)
        if closes[engulf_idx] <= ob_high_ict:
            return None
        return OrderBlock(
            direction="bullish",
            group_start_index=int(group_start),
            group_end_index=int(group_end),
            group_start_ts=df.index[group_start],
            group_end_ts=df.index[group_end],
            ob_high=ob_high_ict,
            ob_low=ob_low_ict,
            ob_open=float(opens[group_start]),
            ob_close=float(closes[group_end]),
            validation_index=int(engulf_idx),
            validation_ts=df.index[engulf_idx],
            validation_close=float(closes[engulf_idx]),
            sweep=sweep,
            group_size=int(group_size),
        )

    # ===== V18 STRICT MODE (Soufiane 2026-05-26 + ajustements V18.1 2026-05-27) =====
    # Sweep bullish -> bougie de sweep DOIT etre baissiere, sinon PAS d'OB
    # V18.1 : min_group=1 (au lieu de 2) pour capturer + d'OB ICT classiques
    # max_group=illimite, bornes mecheSweep/maxCorps, validation close > ob_high,
    # rejection si meche < ob_low pendant le pending.
    if validation_mode == "v18":
        # Etape 1 : bougie de sweep doit etre BAISSIERE
        if closes[sweep_idx] >= opens[sweep_idx]:
            return None
        # Etape 2 : groupe = sweep + baissieres consecutives juste avant (pas de max)
        group_end = sweep_idx
        group_start = group_end
        while group_start > 0 and closes[group_start - 1] < opens[group_start - 1]:
            group_start -= 1
        group_size = group_end - group_start + 1
        # V18.1 : min_group=1 (etait 2). Permet OB ICT "1 derniere bougie baissiere"
        if group_size < 1:
            return None
        # Etape 3 : bornes V18
        # ob_low = low de la meche du sweep (meche INCLUSE en bas)
        ob_low_v18 = float(lows[sweep_idx])
        # ob_high = max(open, close) du groupe (corps uniquement, meches haut EXCLUES)
        ob_high_v18 = float(max(
            max(opens[i], closes[i]) for i in range(group_start, group_end + 1)
        ))
        if ob_high_v18 <= ob_low_v18:
            return None
        # Etape 4-5 : pending + validation + rejection
        # V18 : Soufiane = pending indefini. On override max_bars_after_sweep en boucle
        # jusqu'a fin du df (ou trouve validation/rejection en chemin).
        val_end = len(df)
        validation_idx = None
        for j in range(sweep_idx + 1, val_end):
            # Rejection (priorite) : meche < ob_low -> OB mort
            if lows[j] < ob_low_v18:
                return None
            # Validation : close > ob_high
            if closes[j] > ob_high_v18:
                validation_idx = j
                break
        if validation_idx is None:
            return None
        # Construction OrderBlock V18 (ob_high/ob_low V18, autres champs cohorents)
        return OrderBlock(
            direction="bullish",
            group_start_index=int(group_start),
            group_end_index=int(group_end),
            group_start_ts=df.index[group_start],
            group_end_ts=df.index[group_end],
            ob_high=ob_high_v18,  # V18 : max corps
            ob_low=ob_low_v18,    # V18 : meche sweep
            ob_open=float(opens[group_start]),
            ob_close=float(closes[group_end]),
            validation_index=int(validation_idx),
            validation_ts=df.index[validation_idx],
            validation_close=float(closes[validation_idx]),
            sweep=sweep,
            group_size=int(group_size),
        )

    # 1. Trouver le GROUPE baissier qui se termine a (ou juste avant) sweep_idx
    # On remonte depuis sweep_idx tant que les bougies sont baissieres.
    group_end = sweep_idx
    # La bougie de sweep est-elle baissiere ?
    if closes[sweep_idx] >= opens[sweep_idx]:
        # Bougie de sweep haussiere : le groupe finit a sweep_idx - 1
        group_end = sweep_idx - 1
    if group_end < 0:
        return None
    if closes[group_end] >= opens[group_end]:
        # Pas de bougie baissiere juste avant -> pas de groupe -> pas d'OB
        return None

    # Remonte les baissieres consecutives
    group_start = group_end
    while group_start > 0 and closes[group_start - 1] < opens[group_start - 1]:
        if (group_end - (group_start - 1) + 1) > max_group_size:
            break
        group_start -= 1

    group_size = group_end - group_start + 1
    if group_size < min_group_size or group_size > max_group_size:
        return None

    # 2. Niveaux du groupe (meches incluses)
    ob_high = float(highs[group_start:group_end + 1].max())
    ob_low = float(lows[group_start:group_end + 1].min())

    # 3. Validation selon le mode :
    # - "bos" (V12) : close > ob_high (cassure de structure)
    # - "mitigation" (V14) : low <= ob_high (le prix revient toucher la zone OB)
    val_start = max(sweep_idx, group_end) + 1
    val_end = min(val_start + max_bars_after_sweep, len(df))
    validation_idx = None
    if validation_mode == "mitigation":
        for j in range(val_start, val_end):
            if lows[j] <= ob_high:
                validation_idx = j
                break
    else:  # "bos" (default = V12 comportement actuel)
        for j in range(val_start, val_end):
            if closes[j] > ob_high:
                validation_idx = j
                break

    if validation_idx is None:
        return None

    return OrderBlock(
        direction="bullish",
        group_start_index=int(group_start),
        group_end_index=int(group_end),
        group_start_ts=df.index[group_start],
        group_end_ts=df.index[group_end],
        ob_high=ob_high,
        ob_low=ob_low,
        ob_open=float(opens[group_start]),
        ob_close=float(closes[group_end]),
        validation_index=int(validation_idx),
        validation_ts=df.index[validation_idx],
        validation_close=float(closes[validation_idx]),
        sweep=sweep,
        group_size=int(group_size),
    )


def _try_bearish_ob(
    df: pd.DataFrame,
    sweep: Sweep,
    max_group_size: int,
    max_bars_after_sweep: int,
    min_group_size: int = 1,
    validation_mode: str = "bos",
) -> OrderBlock | None:
    """Symetrique : sweep d'un high -> OB bearish (bougies haussieres consecutives)."""
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values

    sweep_idx = sweep.sweep_index

    # ===== V19 ICT BEARISH (symetrique cluster) =====
    # OB BEARISH = SERIE de bougies HAUSSIERES consecutives avant impulsion baissiere.
    # Validation ICT "engulfing in 1 candle" :
    #   bougie SUIVANTE doit en UNE bougie :
    #   - sweep : high > high_meches du groupe
    #   - engulf : close < min(open,close) du groupe
    #   - etre BAISSIERE
    # Bornes : CORPS uniquement
    if validation_mode == "v19_ict":
        engulf_idx = sweep_idx
        if closes[engulf_idx] >= opens[engulf_idx]:
            return None  # engulf doit etre baissiere
        if engulf_idx - 1 < 0:
            return None
        if closes[engulf_idx - 1] <= opens[engulf_idx - 1]:
            return None  # pas de haussiere juste avant
        group_end = engulf_idx - 1
        group_start = group_end
        while group_start > 0 and closes[group_start - 1] > opens[group_start - 1]:
            group_start -= 1
        group_size = group_end - group_start + 1
        if group_size < 1:
            return None
        # Bornes corps
        ob_high_ict = float(max(
            max(opens[i], closes[i]) for i in range(group_start, group_end + 1)
        ))
        ob_low_ict = float(min(
            min(opens[i], closes[i]) for i in range(group_start, group_end + 1)
        ))
        if ob_high_ict <= ob_low_ict:
            return None
        # CHECK 1 : sweep — high[engulf] > group_high_wick
        group_high_wick = float(max(highs[i] for i in range(group_start, group_end + 1)))
        if highs[engulf_idx] <= group_high_wick:
            return None
        # CHECK 2 : engulf — close[engulf] < ob_low
        if closes[engulf_idx] >= ob_low_ict:
            return None
        return OrderBlock(
            direction="bearish",
            group_start_index=int(group_start),
            group_end_index=int(group_end),
            group_start_ts=df.index[group_start],
            group_end_ts=df.index[group_end],
            ob_high=ob_high_ict,
            ob_low=ob_low_ict,
            ob_open=float(opens[group_start]),
            ob_close=float(closes[group_end]),
            validation_index=int(engulf_idx),
            validation_ts=df.index[engulf_idx],
            validation_close=float(closes[engulf_idx]),
            sweep=sweep,
            group_size=int(group_size),
        )

    # ===== V18 STRICT MODE (Soufiane 2026-05-26 + V18.1 ajustements) =====
    if validation_mode == "v18":
        # Bougie de sweep doit etre HAUSSIERE
        if closes[sweep_idx] <= opens[sweep_idx]:
            return None
        # Groupe = sweep + haussieres consecutives avant
        group_end = sweep_idx
        group_start = group_end
        while group_start > 0 and closes[group_start - 1] > opens[group_start - 1]:
            group_start -= 1
        group_size = group_end - group_start + 1
        # V18.1 : min_group=1
        if group_size < 1:
            return None
        # Bornes V18 bearish
        # ob_high = high de la meche du sweep (meche INCLUSE en haut)
        ob_high_v18 = float(highs[sweep_idx])
        # ob_low = min(open, close) du groupe (corps uniquement, meches bas EXCLUES)
        ob_low_v18 = float(min(
            min(opens[i], closes[i]) for i in range(group_start, group_end + 1)
        ))
        if ob_high_v18 <= ob_low_v18:
            return None
        # Pending + validation + rejection (V18 = indefini)
        val_end = len(df)
        validation_idx = None
        for j in range(sweep_idx + 1, val_end):
            # Rejection : meche > ob_high -> OB mort
            if highs[j] > ob_high_v18:
                return None
            # Validation : close < ob_low
            if closes[j] < ob_low_v18:
                validation_idx = j
                break
        if validation_idx is None:
            return None
        return OrderBlock(
            direction="bearish",
            group_start_index=int(group_start),
            group_end_index=int(group_end),
            group_start_ts=df.index[group_start],
            group_end_ts=df.index[group_end],
            ob_high=ob_high_v18,  # V18 : meche sweep
            ob_low=ob_low_v18,    # V18 : min corps
            ob_open=float(opens[group_start]),
            ob_close=float(closes[group_end]),
            validation_index=int(validation_idx),
            validation_ts=df.index[validation_idx],
            validation_close=float(closes[validation_idx]),
            sweep=sweep,
            group_size=int(group_size),
        )

    group_end = sweep_idx
    if closes[sweep_idx] <= opens[sweep_idx]:
        group_end = sweep_idx - 1
    if group_end < 0:
        return None
    if closes[group_end] <= opens[group_end]:
        return None

    group_start = group_end
    while group_start > 0 and closes[group_start - 1] > opens[group_start - 1]:
        if (group_end - (group_start - 1) + 1) > max_group_size:
            break
        group_start -= 1

    group_size = group_end - group_start + 1
    if group_size < min_group_size or group_size > max_group_size:
        return None

    ob_high = float(highs[group_start:group_end + 1].max())
    ob_low = float(lows[group_start:group_end + 1].min())

    # Validation selon le mode :
    # - "bos" (V12) : close < ob_low (cassure de structure)
    # - "mitigation" (V14) : high >= ob_low (le prix revient toucher la zone OB)
    val_start = max(sweep_idx, group_end) + 1
    val_end = min(val_start + max_bars_after_sweep, len(df))
    validation_idx = None
    if validation_mode == "mitigation":
        for j in range(val_start, val_end):
            if highs[j] >= ob_low:
                validation_idx = j
                break
    else:  # "bos" (default = V12 comportement actuel)
        for j in range(val_start, val_end):
            if closes[j] < ob_low:
                validation_idx = j
                break

    if validation_idx is None:
        return None

    return OrderBlock(
        direction="bearish",
        group_start_index=int(group_start),
        group_end_index=int(group_end),
        group_start_ts=df.index[group_start],
        group_end_ts=df.index[group_end],
        ob_high=ob_high,
        ob_low=ob_low,
        ob_open=float(opens[group_start]),
        ob_close=float(closes[group_end]),
        validation_index=int(validation_idx),
        validation_ts=df.index[validation_idx],
        validation_close=float(closes[validation_idx]),
        sweep=sweep,
        group_size=int(group_size),
    )


# ============ HELPERS ============

SL_LOOKBACK_BARS = 5  # decision user 2026-05-16 : 5 bougies AVANT le groupe pour SL protecteur


def ob_stop_loss(ob: OrderBlock, df: 'pd.DataFrame | None' = None) -> float:
    """SL = extreme depuis le SWEEP jusqu'a la VALIDATION (user 2026-05-17).

    BUG CRITIQUE FIX : avant on s'arretait a group_end, mais les bougies entre
    group_end et validation_index peuvent atteindre des extremes BIEN PLUS HAUTS
    (ex: BTCUSD SELL 2025-08-20 : group_end high=113316, validation+1 high=113378).
    Si SL ne couvre pas cette zone, il est PRIS en live avant la validation.

    Logique ICT pro :
    - Bougie de SWEEP (point ou la liquidite a ete prise)
    - + toutes les bougies entre sweep et VALIDATION (incl. group + bougies de
      poussee/manipulation avant cassure)
    - + padding SL_LOOKBACK_BARS=5 bougies AVANT le sweep
    - SL = pire extreme dans cette fenetre

    - OB bullish  : SL = MIN des lows
    - OB bearish  : SL = MAX des highs

    Si df n'est pas fourni : fallback meche du groupe seul (compat ancien).
    """
    if df is None:
        # Fallback : meche groupe seul (compat)
        if ob.direction == "bullish":
            return ob.ob_low
        return ob.ob_high

    # Point de depart : le sweep (si dispo) sinon group_start
    sweep_idx = getattr(ob.sweep, "sweep_index", None) if ob.sweep is not None else None
    if sweep_idx is None:
        sweep_idx = ob.group_start_index

    # Fenetre : N bougies avant sweep -> validation_index (inclusif)
    # Couvre group + bougies de manipulation entre group et validation
    # V18.2 : SL_LOOKBACK_BARS optimisable via env var
    _sl_lookback = int(os.environ.get("V18_SL_LOOKBACK", str(SL_LOOKBACK_BARS)))
    lookback_start = max(0, sweep_idx - _sl_lookback)
    end_idx = max(ob.group_end_index, ob.validation_index)
    sub = df.iloc[lookback_start:end_idx + 1]
    if len(sub) == 0:
        # Fallback meche
        if ob.direction == "bullish":
            return ob.ob_low
        return ob.ob_high

    # V18.2 (Soufiane 2026-05-27) : padding SL adaptatif ATR.
    # Justification : analyse forensique 474k trades V18.1 -> Q4 risk_points (large SL)
    # WR=38.9% vs Q1 (serre) WR=35.4% (+3.4pts). SL trop serre = touche au bruit.
    # Env var V18_SL_ATR_MULT=N => ajoute N x ATR au SL (defaut 0 = comportement V18.0).
    sl_atr_mult = float(os.environ.get("V18_SL_ATR_MULT", "0"))
    if sl_atr_mult > 0 and len(df) >= 20:
        # ATR rapide M1 : moyenne range sur 14 dernieres bougies avant validation
        atr_window = df.iloc[max(0, ob.validation_index - 14):ob.validation_index]
        if len(atr_window) > 0:
            atr = float((atr_window["high"] - atr_window["low"]).mean())
            padding = atr * sl_atr_mult
        else:
            padding = 0
    else:
        padding = 0

    if ob.direction == "bullish":
        return float(sub["low"].min()) - padding
    return float(sub["high"].max()) + padding


def ob_entry(ob: OrderBlock) -> float:
    """Entree de l'OB : prix d'entree quand le price revient dans la zone.

    Vizion privilégie l'entree au CORPS de bougie (open du groupe = milieu de l'OB).
    Pour l'instant on prend le HIGH (pour un bullish OB) ou le LOW (pour un bearish OB)
    comme niveau d'entree limite : le price doit retomber dans la zone OB.
    """
    if ob.direction == "bullish":
        return ob.ob_high
    return ob.ob_low


def ob_is_invalidated(ob: OrderBlock, df: pd.DataFrame) -> bool:
    """Verifie si l'OB a ete invalide apres sa validation.

    Bullish : invalide si une bougie clos SOUS le low du groupe (corps, pas meche).
    Bearish : invalide si une bougie clos AU-DESSUS du high du groupe.
    """
    after = df.iloc[ob.validation_index + 1:]
    if ob.direction == "bullish":
        return bool((after["close"] < ob.ob_low).any())
    return bool((after["close"] > ob.ob_high).any())


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent))
    from bot_v2.data_loader import load

    df = load("XAUUSD", "M5")
    print(f"XAUUSD M5 : {len(df)} bougies")

    # Test sur 7 derniers jours
    mask = df.index >= (df.index.max() - pd.Timedelta(days=7))
    df_week = df[mask]
    print(f"Derniers 7 jours : {len(df_week)} bougies")

    obs = detect_order_blocks(df_week)
    print(f"\nOrder Blocks detectes : {len(obs)}")
    print(f"  - Bullish : {sum(1 for o in obs if o.direction == 'bullish')}")
    print(f"  - Bearish : {sum(1 for o in obs if o.direction == 'bearish')}")

    print(f"\nGroup size repartition :")
    from collections import Counter
    sizes = Counter(o.group_size for o in obs)
    for s, c in sorted(sizes.items()):
        print(f"  {s} bougies : {c} OB")

    print(f"\n=== 5 derniers OB ===")
    for ob in obs[-5:]:
        sl = ob_stop_loss(ob)
        entry = ob_entry(ob)
        risk = abs(entry - sl)
        print(
            f"  {ob.validation_ts} | {ob.direction:8s} | "
            f"group={ob.group_size}b | entry={entry:.3f} | sl={sl:.3f} | risk={risk:.3f}"
        )

    # Verification stricte : tous les OB ont des bougies du bon sens
    print("\n=== Verification stricte (decision user 2026-05-15) ===")
    n_strict_ok = 0
    for ob in obs:
        group = df_week.iloc[ob.group_start_index:ob.group_end_index + 1]
        if ob.direction == "bullish":
            # Toutes les bougies du groupe doivent etre baissieres
            all_bearish = (group["close"] < group["open"]).all()
            if all_bearish:
                n_strict_ok += 1
        else:
            all_bullish = (group["close"] > group["open"]).all()
            if all_bullish:
                n_strict_ok += 1
    print(f"OB strictement valides : {n_strict_ok}/{len(obs)}")
    if n_strict_ok != len(obs):
        print("⚠ ATTENTION : certains OB ont des bougies intruses (bug)")
