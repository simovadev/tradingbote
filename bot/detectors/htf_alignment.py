"""Score d'alignement HTF complet (0-100).

Verifie 5 dimensions ICT/SMC pour savoir si l'OB M1 a le contexte favorable :

1. H4 swing structure (max 20 pts)
   Dernier swing H4 (HH/HL ou LL/LH) dans le sens du trade
2. H1 OB confluent (max 20 pts)
   Y a-t-il un OB H1 qui chevauche l'OB M1 dans le bon sens ?
3. Daily bias (max 20 pts)
   La liquidite du jour precedent a-t-elle ete prise dans le bon sens ?
4. Liquidite avant TP (max 20 pts)
   Pas de liquidite opposee majeure entre entry et TP ?
5. Killzone forte (max 20 pts)
   LN open / NY open / LN-NY overlap = meilleurs moments
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time, timedelta
from typing import Literal

import pandas as pd

from bot.detectors.fvg import detect_fvgs
from bot.detectors.swings import detect_swings
from bot.killzones import current_killzone


@dataclass
class AlignmentComponent:
    name: str
    score: int             # points obtenus (0 a max_pts)
    max_pts: int
    status: Literal["ok", "warning", "fail", "n/a"]
    detail: str


@dataclass
class HTFAlignment:
    total_score: int                    # 0-100
    components: list[AlignmentComponent] = field(default_factory=list)
    verdict: str = ""                    # premium / good / mediocre / weak

    def to_dict(self) -> dict:
        return {
            "total_score": self.total_score,
            "verdict": self.verdict,
            "components": [
                {"name": c.name, "score": c.score, "max": c.max_pts,
                 "status": c.status, "detail": c.detail}
                for c in self.components
            ],
        }


# ============ Composantes ============

def _component_h4_swing(df_h4: pd.DataFrame | None, trade_direction: str, asof: pd.Timestamp) -> AlignmentComponent:
    max_pts = 20
    if df_h4 is None or df_h4.empty:
        return AlignmentComponent("H4 swing", 0, max_pts, "n/a", "Pas de H4 dispo")
    sub = df_h4.loc[df_h4.index <= asof].tail(50)
    if len(sub) < 10:
        return AlignmentComponent("H4 swing", 0, max_pts, "n/a", "Pas assez de bougies H4")
    swings = detect_swings(sub, left=2, right=2)
    highs = [s for s in swings if s.kind == "high"][-2:]
    lows = [s for s in swings if s.kind == "low"][-2:]
    if len(highs) < 2 or len(lows) < 2:
        return AlignmentComponent("H4 swing", 5, max_pts, "warning", "Structure H4 incertaine")
    last_h, prev_h = highs[-1].price, highs[-2].price
    last_l, prev_l = lows[-1].price, lows[-2].price
    if last_h > prev_h and last_l > prev_l:
        h4_bias = "bullish"
    elif last_h < prev_h and last_l < prev_l:
        h4_bias = "bearish"
    else:
        return AlignmentComponent("H4 swing", 10, max_pts, "warning", "H4 range/mixte")

    if h4_bias == trade_direction:
        return AlignmentComponent("H4 swing", max_pts, max_pts, "ok", f"H4 {h4_bias} aligne")
    else:
        return AlignmentComponent("H4 swing", 0, max_pts, "fail", f"H4 {h4_bias} CONTRE le trade")


def _component_h1_ob_confluence(
    df_h1: pd.DataFrame | None,
    trade_direction: str,
    ob_high: float, ob_low: float,
    asof: pd.Timestamp,
) -> AlignmentComponent:
    """Verifie si une bougie H1 du bon sens englobe l'OB M1."""
    max_pts = 20
    if df_h1 is None or df_h1.empty:
        return AlignmentComponent("H1 OB confluence", 0, max_pts, "n/a", "Pas de H1 dispo")
    sub = df_h1.loc[df_h1.index <= asof].tail(50)
    if len(sub) < 5:
        return AlignmentComponent("H1 OB confluence", 0, max_pts, "n/a", "Pas assez de H1")

    # Cherche une bougie H1 de la bonne couleur dont le range englobe au moins partiellement l'OB M1
    target_bull = trade_direction == "bullish"   # LONG -> on cherche bougies rouges (OB haussier H1)
    target_bull_color = not target_bull  # OB long = bougie rouge

    for i in range(len(sub) - 1, max(-1, len(sub) - 20), -1):
        row = sub.iloc[i]
        is_bull_candle = row["close"] > row["open"]
        if is_bull_candle != target_bull_color:
            continue
        # Verifie chevauchement avec OB M1
        if row["high"] < ob_low or row["low"] > ob_high:
            continue
        # Bougie H1 du bon type qui chevauche
        return AlignmentComponent("H1 OB confluence", max_pts, max_pts, "ok",
                                  f"Bougie H1 {('rouge' if not is_bull_candle else 'verte')} @ {sub.index[i].strftime('%H:%M')} englobe OB M1")

    return AlignmentComponent("H1 OB confluence", 0, max_pts, "fail", "Aucune bougie H1 confluente trouvee")


def _component_daily_bias(
    df_m1: pd.DataFrame,
    trade_direction: str,
    asof: pd.Timestamp,
) -> AlignmentComponent:
    """Daily bias : ou la liquidite de la veille a-t-elle ete prise ?
    Si veille HIGH balayee aujourd'hui = bias bearish (chasse les longs)
    Si veille LOW balayee aujourd'hui = bias bullish (chasse les shorts)
    """
    max_pts = 20
    today_start = asof.normalize()
    prev_start = today_start - timedelta(days=1)
    while prev_start.weekday() >= 5:
        prev_start -= timedelta(days=1)
    prev_end = prev_start + timedelta(days=1)
    prev_day = df_m1.loc[(df_m1.index >= prev_start) & (df_m1.index < prev_end)]
    today_data = df_m1.loc[(df_m1.index >= today_start) & (df_m1.index < asof)]

    if prev_day.empty or today_data.empty:
        return AlignmentComponent("Daily bias", 5, max_pts, "n/a", "Pas assez de data jour")

    pdh = float(prev_day["high"].max())
    pdl = float(prev_day["low"].min())
    today_high = float(today_data["high"].max())
    today_low = float(today_data["low"].min())

    pdh_swept = today_high > pdh
    pdl_swept = today_low < pdl

    if pdh_swept and not pdl_swept:
        # PDH sweep = bias bearish (apres prise de liquidite haute)
        bias = "bearish"
    elif pdl_swept and not pdh_swept:
        bias = "bullish"
    elif pdh_swept and pdl_swept:
        # Les deux sweep = range/incertain
        return AlignmentComponent("Daily bias", 10, max_pts, "warning", "PDH ET PDL sweep = range")
    else:
        return AlignmentComponent("Daily bias", 10, max_pts, "warning", "Aucune liquidite veille prise encore")

    if bias == trade_direction:
        return AlignmentComponent("Daily bias", max_pts, max_pts, "ok", f"Daily bias {bias} aligne")
    else:
        return AlignmentComponent("Daily bias", 0, max_pts, "fail", f"Daily bias {bias} CONTRE")


def _component_no_opposite_liquidity(
    df_m1: pd.DataFrame,
    trade_direction: str,
    entry_price: float,
    take_profit: float,
    asof: pd.Timestamp,
) -> AlignmentComponent:
    """Verifie qu'il n'y a pas de liquidite opposee MAJEURE entre entry et TP."""
    max_pts = 20
    today_start = asof.normalize()
    today_data = df_m1.loc[(df_m1.index >= today_start) & (df_m1.index < asof)]
    if today_data.empty:
        return AlignmentComponent("Pas de liq oppose", 10, max_pts, "n/a", "Pas de data")

    # Pour LONG : on regarde si entre entry et TP il y a un HIGH du jour deja fait
    # qui n'a pas ete repris (= liquidite qui va attirer le prix)
    swings = detect_swings(today_data, left=5, right=5)
    if trade_direction == "bullish":
        # Liquidite OPPOSEE pour un LONG = un LOW non-pris dans la zone (entre entry et TP)
        # Mais on cherche surtout : un HIGH AVANT le TP que le marche va chercher
        # Pour LONG : pas de HIGH MAJEUR avant le TP qui freine
        # Ici on simplifie : on regarde s'il y a des LOWS non-repris ENTRE entry et TP
        lows_between = [s for s in swings if s.kind == "low" and entry_price > s.price > take_profit]
        if not lows_between:
            return AlignmentComponent("Pas de liq oppose", max_pts, max_pts, "ok", "Chemin libre vers TP")
        # Il y a des liquidites attirantes entre = warning
        return AlignmentComponent("Pas de liq oppose", 8, max_pts, "warning",
                                  f"{len(lows_between)} swing low(s) entre entry et TP")
    else:
        highs_between = [s for s in swings if s.kind == "high" and entry_price < s.price < take_profit]
        if not highs_between:
            return AlignmentComponent("Pas de liq oppose", max_pts, max_pts, "ok", "Chemin libre vers TP")
        return AlignmentComponent("Pas de liq oppose", 8, max_pts, "warning",
                                  f"{len(highs_between)} swing high(s) entre entry et TP")


def _component_fvg_htf_confluence(
    df_h1: pd.DataFrame | None,
    df_h4: pd.DataFrame | None,
    df_m15: pd.DataFrame | None,
    trade_direction: str,
    ob_high: float, ob_low: float,
    asof: pd.Timestamp,
) -> AlignmentComponent:
    """SIGNAL PREMIUM Vizion : FVG ou iFVG en HTF (H4, H1, ou M15) qui chevauche l'OB M1.
    C'est le pattern le plus probabilisant ICT. Poids ELEVE = 30 pts.
    """
    max_pts = 30
    # On verifie M15 d'abord (le plus pertinent pour M1), puis H1, puis H4
    found = None
    found_tf = None
    found_type = None  # "fvg" ou "ifvg"

    for tf_name, df_tf in [("M15", df_m15), ("H1", df_h1), ("H4", df_h4)]:
        if df_tf is None or df_tf.empty:
            continue
        sub = df_tf.loc[df_tf.index <= asof].tail(80)
        if len(sub) < 10:
            continue
        fvgs = detect_fvgs(sub)
        # On cherche un FVG ou iFVG de la bonne direction qui chevauche l'OB M1
        for fvg in fvgs:
            # Direction de la FVG doit matcher direction du trade
            if fvg.kind != trade_direction:
                continue
            # Chevauchement avec OB M1
            if fvg.zone_high < ob_low or fvg.zone_low > ob_high:
                continue
            # iFVG (deja inverte) = signal encore plus fort
            if fvg.inverted:
                found = fvg
                found_tf = tf_name
                found_type = "iFVG"
                break   # on prend ce premier match
            elif not fvg.filled:
                # FVG actif non rempli = top
                if found is None:
                    found = fvg
                    found_tf = tf_name
                    found_type = "FVG actif"
        if found is not None:
            break   # on a trouve sur ce TF, on s'arrete

    if found is None:
        return AlignmentComponent("FVG HTF confluence", 0, max_pts, "n/a",
                                  "Aucune FVG/iFVG HTF en confluence")

    # iFVG sur HTF haut = signal PREMIUM
    if found_type == "iFVG":
        return AlignmentComponent("FVG HTF confluence", max_pts, max_pts, "ok",
                                  f"iFVG {found_tf} en confluence avec OB M1 (signal PREMIUM)")
    else:
        # FVG actif non rempli sur HTF
        pts = max_pts if found_tf in ("H1", "H4") else int(max_pts * 0.75)
        return AlignmentComponent("FVG HTF confluence", pts, max_pts, "ok",
                                  f"FVG {found_tf} actif en confluence avec OB M1")


def _component_killzone(asof: pd.Timestamp) -> AlignmentComponent:
    """Killzone : London & NY = max, Asia random = peu."""
    max_pts = 20
    kz = current_killzone(asof)
    if kz is None:
        return AlignmentComponent("Killzone", 0, max_pts, "fail", "HORS killzone")
    weights = {
        "london":       max_pts,              # 20
        "ny":           max_pts,              # 20
        "london_close": int(max_pts * 0.75),  # 15
        "asia":         int(max_pts * 0.5),   # 10
    }
    pts = weights.get(kz.name, 5)
    status = "ok" if pts >= 15 else ("warning" if pts >= 8 else "fail")
    return AlignmentComponent("Killzone", pts, max_pts, status, f"{kz.label}")


# ============ Aggregation ============

def compute_alignment(
    df_m1: pd.DataFrame,
    df_h1: pd.DataFrame | None,
    df_h4: pd.DataFrame | None,
    trade_direction: str,
    ob_high: float, ob_low: float,
    entry_price: float, take_profit: float,
    asof: pd.Timestamp,
    df_m15: pd.DataFrame | None = None,
) -> HTFAlignment:
    """Calcule le score HTF complet sur 130 points (5 composantes x 20 + 1 x 30)."""
    if asof.tz is None:
        asof = asof.tz_localize("UTC")

    components = [
        _component_h4_swing(df_h4, trade_direction, asof),
        _component_h1_ob_confluence(df_h1, trade_direction, ob_high, ob_low, asof),
        _component_daily_bias(df_m1, trade_direction, asof),
        _component_no_opposite_liquidity(df_m1, trade_direction, entry_price, take_profit, asof),
        _component_killzone(asof),
        _component_fvg_htf_confluence(df_h1, df_h4, df_m15, trade_direction, ob_high, ob_low, asof),
    ]

    total = sum(c.score for c in components)
    max_total = sum(c.max_pts for c in components)   # 130
    # Normalise sur 100
    total_norm = int(round(total / max_total * 100))

    if total_norm >= 85:
        verdict = "premium"
    elif total_norm >= 65:
        verdict = "good"
    elif total_norm >= 45:
        verdict = "mediocre"
    else:
        verdict = "weak"

    return HTFAlignment(total_score=total_norm, components=components, verdict=verdict)
