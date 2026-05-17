"""SL/TP dynamique style ICT pur (decision user 2026-05-16).

Probleme detecte par user via dashboard replay :
- SL trop large (sur mèche du groupe OB) -> stop hunts faciles
- TP fixe RR=2/3 -> ignore les vraies liquidites du marche

Solution ICT :
- SL = au-dela du **prochain swing non sweepe** = la VRAIE liquidite que les pros
  iraient chasser. Si on est buy, on place SL sous le low non-sweepe le plus proche
  -> les pros doivent casser une liquidite pour nous sortir.
- TP = **prochaine liquidite externe** (swing HTF + sessions H/L) dans la direction.
  Le marche ira chercher ces niveaux mecaniquement.

Le RR devient dynamique : 1.2 a 4+ selon ce que le marche propose.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from bot_v2.concepts.liquidity import Swing, Sweep
from bot_v2.concepts.htf_swings import HTFSwing
from bot_v2.concepts.order_block import OrderBlock


Direction = Literal["bullish", "bearish"]


@dataclass(frozen=True)
class SLTPResult:
    """Resultat du calcul SL/TP dynamique."""
    sl: float
    tp: float
    sl_source: str               # "swing_low_unswept" | "ob_meche" | "atr_buffer"
    tp_source: str               # "session_high_London" | "htf_H1_swing" | "swing_high_unswept"
    risk_points: float
    reward_points: float
    rr: float


def _atr_buffer(df: pd.DataFrame, idx: int, lookback: int = 14, mult: float = 0.2) -> float:
    """ATR(14) * mult comme buffer minimum au-dela d'une liquidite."""
    if idx < lookback:
        return 0.0
    sub = df.iloc[idx - lookback:idx]
    atr = float((sub["high"] - sub["low"]).mean())
    return atr * mult


def _live_swings_before(swings: list[Swing], sweeps: list[Sweep], ref_idx: int) -> list[Swing]:
    """Filtre les swings encore LIVE (non sweepes) AVANT ref_idx."""
    swept_ids = {(sw.swing.kind, sw.swing.index) for sw in sweeps}
    return [
        s for s in swings
        if s.index < ref_idx and (s.kind, s.index) not in swept_ids
    ]


def find_sl_liquidity_aware(
    ob: OrderBlock,
    df_ltf: pd.DataFrame,
    swings: list[Swing],
    sweeps: list[Sweep],
    entry: float,
    fallback_meche: float,
    max_sl_distance_pct: float = 0.005,  # 0.5% max distance entry -> SL
) -> tuple[float, str]:
    """SL = au-dela du prochain swing live (non sweepe) AVANT l'OB.

    Logique ICT :
    - Buy : SL sous le swing LOW live le plus proche EN DESSOUS de ob_low
    - Sell : SL au-dessus du swing HIGH live le plus proche AU DESSUS de ob_high

    Fallback : meche du groupe OB (comportement v1).
    Filet : si SL > max_sl_distance_pct du prix d'entree, on revient au meche.
    """
    buffer = _atr_buffer(df_ltf, ob.validation_index)
    live = _live_swings_before(swings, sweeps, ob.group_start_index)

    if ob.direction == "bullish":
        # Cherche le swing LOW live le plus proche EN DESSOUS de ob.ob_low
        candidates = [s for s in live if s.kind == "low" and s.price < ob.ob_low]
        if candidates:
            # Le plus PROCHE (= max price parmi ceux qui sont sous ob.ob_low)
            target = max(candidates, key=lambda s: s.price)
            sl = target.price - buffer
            # Filet : si trop loin du entry, fallback meche
            distance = (entry - sl) / entry
            if distance <= max_sl_distance_pct and sl < entry:
                return sl, f"swing_low_live_{target.strength}"
        return fallback_meche, "meche_ob_fallback"

    # bearish
    candidates = [s for s in live if s.kind == "high" and s.price > ob.ob_high]
    if candidates:
        target = min(candidates, key=lambda s: s.price)
        sl = target.price + buffer
        distance = (sl - entry) / entry
        if distance <= max_sl_distance_pct and sl > entry:
            return sl, f"swing_high_live_{target.strength}"
    return fallback_meche, "meche_ob_fallback"


def find_tp_liquidity_external(
    ob: OrderBlock,
    entry: float,
    sl: float,
    htf_swings: list[HTFSwing] | None,
    ltf_swings: list[Swing],
    sweeps: list[Sweep],
    session_levels: list[tuple[str, float]] | None = None,
    min_rr: float = 1.5,
    max_rr_search: float = 5.0,
) -> tuple[float, str]:
    """TP = prochaine liquidite externe dans la direction.

    Ordre de priorite ICT :
    1. Session high/low precedente non touche (London/Asia/NY)
    2. Swing HTF live (D1 > H4 > H1)
    3. Swing LTF live (M1)
    4. Fallback : RR=2 fixe

    Filtre : seul un TP qui donne RR >= min_rr est retenu.
    """
    risk = abs(entry - sl)
    if risk == 0:
        return entry, "invalid_zero_risk"

    candidates: list[tuple[float, str]] = []

    # 1. Session levels (highs/lows de session terminee)
    if session_levels:
        for src, lvl in session_levels:
            if ob.direction == "bullish" and lvl > entry:
                candidates.append((lvl, f"session_{src}"))
            elif ob.direction == "bearish" and lvl < entry:
                candidates.append((lvl, f"session_{src}"))

    # 2. HTF swings live (D1 > H4 > H1 priorite)
    if htf_swings:
        tf_priority = {"D1": 0, "H4": 1, "H1": 2}
        sorted_htf = sorted(htf_swings, key=lambda s: tf_priority.get(s.tf_name, 99))
        for hs in sorted_htf:
            if not hs.is_live:
                continue
            p = hs.swing.price
            if ob.direction == "bullish" and hs.swing.kind == "high" and p > entry:
                candidates.append((p, f"htf_{hs.tf_name}_swing_high"))
            elif ob.direction == "bearish" and hs.swing.kind == "low" and p < entry:
                candidates.append((p, f"htf_{hs.tf_name}_swing_low"))

    # 3. LTF swings live au-dela de l'OB
    swept_ids = {(sw.swing.kind, sw.swing.index) for sw in sweeps}
    ltf_live = [
        s for s in ltf_swings
        if s.index > ob.validation_index and (s.kind, s.index) not in swept_ids
    ]
    for s in ltf_live:
        p = s.price
        if ob.direction == "bullish" and s.kind == "high" and p > entry:
            candidates.append((p, f"ltf_swing_high_{s.strength}"))
        elif ob.direction == "bearish" and s.kind == "low" and p < entry:
            candidates.append((p, f"ltf_swing_low_{s.strength}"))

    # Cherche la PREMIERE liquidite dans la direction qui donne RR >= min_rr
    # mais aussi RR <= max_rr_search (sinon TP trop loin = miss souvent)
    if ob.direction == "bullish":
        # Trier par prix CROISSANT (plus proche d'abord), on cherche la 1ere >= min_rr
        candidates_sorted = sorted(candidates, key=lambda c: c[0])
        for lvl, src in candidates_sorted:
            reward = lvl - entry
            rr = reward / risk
            if min_rr <= rr <= max_rr_search:
                return lvl, src
    else:
        # bearish : trier par prix DECROISSANT (plus proche d'abord)
        candidates_sorted = sorted(candidates, key=lambda c: -c[0])
        for lvl, src in candidates_sorted:
            reward = entry - lvl
            rr = reward / risk
            if min_rr <= rr <= max_rr_search:
                return lvl, src

    # Fallback : RR=2 fixe
    if ob.direction == "bullish":
        return entry + risk * 2.0, "fixed_rr2_fallback"
    return entry - risk * 2.0, "fixed_rr2_fallback"


def compute_session_levels(
    df_ltf: pd.DataFrame,
    ref_ts: pd.Timestamp,
    lookback_hours: int = 24,
) -> list[tuple[str, float]]:
    """Calcule les H/L des sessions terminees dans les `lookback_hours` precedentes.

    Sessions UTC :
    - Asia    : 00:00 - 08:00 UTC
    - London  : 07:00 - 16:00 UTC
    - NY      : 13:00 - 21:00 UTC

    Retourne liste de (session_name, level) avec les H et L de chaque session.
    """
    if df_ltf.empty:
        return []
    start_window = ref_ts - pd.Timedelta(hours=lookback_hours)
    df_w = df_ltf.loc[(df_ltf.index >= start_window) & (df_ltf.index < ref_ts)]
    if df_w.empty:
        return []

    sessions = {
        "Asia": (0, 8),
        "London": (7, 16),
        "NY": (13, 21),
    }

    levels: list[tuple[str, float]] = []
    # Groupe par date pour avoir 1 session par jour
    for date, day_df in df_w.groupby(df_w.index.date):
        for sname, (h_start, h_end) in sessions.items():
            sess = day_df[(day_df.index.hour >= h_start) & (day_df.index.hour < h_end)]
            if len(sess) < 5:
                continue
            levels.append((f"{sname}_H_{date}", float(sess["high"].max())))
            levels.append((f"{sname}_L_{date}", float(sess["low"].min())))

    return levels


def build_dynamic_sl_tp(
    ob: OrderBlock,
    df_ltf: pd.DataFrame,
    swings: list[Swing],
    sweeps: list[Sweep],
    htf_swings: list[HTFSwing] | None,
    entry: float,
    fallback_meche_sl: float,
    min_rr: float = 1.5,
    max_rr_search: float = 5.0,
    max_sl_distance_pct: float = 0.005,
) -> SLTPResult | None:
    """Pipeline complete : calcule SL dynamique + TP liquidite + verifie RR.

    Retourne None si pas de TP qui donne RR >= min_rr.
    """
    # SL
    sl, sl_src = find_sl_liquidity_aware(
        ob, df_ltf, swings, sweeps, entry, fallback_meche_sl, max_sl_distance_pct
    )
    risk = abs(entry - sl)
    if risk == 0:
        return None

    # Session levels
    session_levels = compute_session_levels(df_ltf, ob.validation_ts, lookback_hours=24)

    # TP
    tp, tp_src = find_tp_liquidity_external(
        ob, entry, sl, htf_swings, swings, sweeps,
        session_levels=session_levels,
        min_rr=min_rr, max_rr_search=max_rr_search,
    )

    reward = abs(tp - entry)
    rr = reward / risk
    if rr < min_rr:
        return None

    return SLTPResult(
        sl=float(sl), tp=float(tp),
        sl_source=sl_src, tp_source=tp_src,
        risk_points=float(risk), reward_points=float(reward), rr=float(rr),
    )
