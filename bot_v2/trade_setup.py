"""Trade setup : Entry / Stop Loss / Take Profit / RR.

Bible §13 + decisions user 2026-05-15 :
- SL = MECHE TOUJOURS (OB, breaker, FVG).
- RR min = 1.5, RR cible = 2-6.
- Risque par trade = 1% du capital par defaut.
- TP base sur prochain swing H/L pertinent (HTF prefere).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from bot_v2.concepts.htf_swings import HTFSwing, find_next_htf_target
from bot_v2.concepts.liquidity import Sweep, Swing
from bot_v2.concepts.order_block import OrderBlock, ob_entry, ob_stop_loss
from bot_v2.concepts.sl_tp_dynamic import build_dynamic_sl_tp
from bot_v2.config import (
    INITIAL_BALANCE_USD,
    INSTRUMENTS,
    RISK_PER_TRADE_PCT,
    RR_MAX,
    RR_MIN,
    RR_TARGET,
    get_param,
)


Direction = Literal["bullish", "bearish"]


@dataclass(frozen=True)
class TradeSetup:
    """Un setup de trade complet, calcule a partir d'un OB."""
    instrument: str
    direction: Direction          # bullish = long, bearish = short
    entry_price: float
    stop_loss: float              # toujours en MECHE (decision user)
    take_profit: float
    rr: float                     # risk/reward ratio
    risk_points: float            # distance entry -> SL en points
    reward_points: float          # distance entry -> TP en points
    risk_usd: float               # perte si SL touche (a balance fixe)
    reward_usd: float             # gain si TP touche
    position_size_lots: float     # taille de position calculee
    # Origine du setup
    ob_validation_ts: pd.Timestamp
    tp_source: str                # description : "next_swing_high", "ATR_target", etc.


def find_next_swing_target(
    df: pd.DataFrame,
    entry_index: int,
    direction: Direction,
    swings: list[Swing],
    min_distance_pct: float = 0.001,
) -> tuple[float, str] | None:
    """Cherche le prochain swing H/L PERTINENT comme TP.

    Bullish : on cherche le prochain SWING HIGH non encore pris au-dessus de l'entry.
    Bearish : prochain swing LOW sous l'entry.

    Retourne (prix_TP, source_description) ou None.
    """
    if entry_index >= len(df):
        return None
    entry_price = float(df.iloc[entry_index]["close"])

    # On regarde les swings AVANT entry_index (= swings historiques pertinents)
    candidates = [s for s in swings if s.index < entry_index]

    if direction == "bullish":
        # Swings HIGH au-dessus de entry, non pris encore
        targets = [s for s in candidates if s.kind == "high" and s.price > entry_price * (1 + min_distance_pct)]
        if not targets:
            return None
        # Le plus PROCHE au-dessus
        best = min(targets, key=lambda s: s.price)
        return float(best.price), f"swing_high@{best.timestamp.date()}"
    else:
        targets = [s for s in candidates if s.kind == "low" and s.price < entry_price * (1 - min_distance_pct)]
        if not targets:
            return None
        best = max(targets, key=lambda s: s.price)
        return float(best.price), f"swing_low@{best.timestamp.date()}"


def compute_position_size(
    entry: float,
    stop_loss: float,
    instrument: str,
    balance: float = INITIAL_BALANCE_USD,
    risk_pct: float = RISK_PER_TRADE_PCT,
) -> tuple[float, float]:
    """Calcule la taille de position en LOTS pour risquer `risk_pct` du balance.

    Returns:
        (lots, risk_usd) — lots arrondi a 0.01.
    """
    if instrument not in INSTRUMENTS:
        raise ValueError(f"Instrument inconnu : {instrument}")

    risk_usd = balance * risk_pct
    risk_points = abs(entry - stop_loss)
    if risk_points == 0:
        return 0.0, 0.0

    tick_value = INSTRUMENTS[instrument]["tick_value"]
    # Risque par lot = risk_points * tick_value
    risk_per_lot = risk_points * tick_value
    if risk_per_lot == 0:
        return 0.0, 0.0

    lots = risk_usd / risk_per_lot
    return round(lots, 2), risk_usd


def build_setup_from_ob(
    df: pd.DataFrame,
    ob: OrderBlock,
    swings: list[Swing],
    instrument: str,
    balance: float = INITIAL_BALANCE_USD,
    risk_pct: float = RISK_PER_TRADE_PCT,
    rr_min: float = RR_MIN,
    rr_target: float = RR_TARGET,
    htf_swings: list[HTFSwing] | None = None,
    sweeps: list[Sweep] | None = None,
    use_dynamic_sl_tp: bool = False,
) -> TradeSetup | None:
    """Construit un TradeSetup complet a partir d'un OB.

    Etapes :
    1. Entry = limite zone OB.
    2. SL = mèche (classique) OU swing live non sweepe (dynamique).
    3. TP = swing HTF (classique) OU prochaine liquidite externe (dynamique).
    4. Calcule taille position pour risquer risk_pct du balance.

    Args:
        use_dynamic_sl_tp: si True, utilise sl_tp_dynamic.build_dynamic_sl_tp
            (SL au-dela prochain swing non sweepe, TP sur liquidite externe).
            Nouveau mode ICT (decision user 2026-05-16).
    """
    entry = ob_entry(ob)
    sl_fallback = ob_stop_loss(ob, df)  # SL protecteur : groupe + 5 bougies avant
    risk_points = abs(entry - sl_fallback)
    if risk_points == 0:
        return None

    # ===== MODE DYNAMIQUE ICT (decision user 2026-05-16) =====
    if use_dynamic_sl_tp:
        dynamic = build_dynamic_sl_tp(
            ob, df, swings, sweeps or [], htf_swings,
            entry=entry, fallback_meche_sl=sl_fallback,
            min_rr=rr_min,
        )
        if dynamic is None:
            return None
        sl = dynamic.sl
        tp = dynamic.tp
        rr = dynamic.rr
        risk_points = dynamic.risk_points
        reward_points = dynamic.reward_points
        tp_source = f"{dynamic.tp_source}|sl={dynamic.sl_source}"

        lots, risk_usd = compute_position_size(entry, sl, instrument, balance, risk_pct)
        tick_value = INSTRUMENTS[instrument]["tick_value"]
        reward_usd = abs(tp - entry) * tick_value * lots

        return TradeSetup(
            instrument=instrument, direction=ob.direction,
            entry_price=float(entry), stop_loss=float(sl), take_profit=float(tp),
            rr=float(rr), risk_points=float(risk_points), reward_points=float(reward_points),
            risk_usd=float(risk_usd), reward_usd=float(reward_usd),
            position_size_lots=float(lots),
            ob_validation_ts=ob.validation_ts,
            tp_source=tp_source,
        )

    # ===== MODE CLASSIQUE (v2 actuel) =====
    sl = sl_fallback

    tp = None
    tp_source = ""

    min_dist_tp = get_param(instrument, "min_distance_tp_pct", 0.0005)

    # Decision user : RR plafonne par actif (NAS100=2, autres=RR_MAX=3).
    rr_max_inst = get_param(instrument, "rr_max", RR_MAX)
    capped_tp = entry + (risk_points * rr_max_inst) if ob.direction == "bullish" else entry - (risk_points * rr_max_inst)

    # PRIORITE 1 : swing HTF pertinent (D1 > H4 > H1), CAPPED a RR_MAX
    if htf_swings:
        htf_target = find_next_htf_target(
            htf_swings, entry, ob.direction, min_distance_pct=min_dist_tp,
        )
        if htf_target is not None:
            tp_candidate = float(htf_target.swing.price)
            if ob.direction == "bullish":
                # Plafonne au plus proche : min(swing_htf, capped_tp)
                tp_capped = min(tp_candidate, capped_tp)
                reward_points = tp_capped - entry
            else:
                tp_capped = max(tp_candidate, capped_tp)
                reward_points = entry - tp_capped
            rr_candidate = reward_points / risk_points
            if rr_candidate >= rr_min:
                tp = tp_capped
                tp_source = f"htf_{htf_target.tf_name}_swing_capped" if tp_capped == capped_tp else f"htf_{htf_target.tf_name}_swing"
                rr = rr_candidate

    # PRIORITE 2 : swing LTF, CAPPED a RR_MAX
    if tp is None:
        next_target = find_next_swing_target(
            df, ob.validation_index, ob.direction, swings,
            min_distance_pct=min_dist_tp,
        )
        if next_target is not None:
            tp_swing, tp_src = next_target
            if ob.direction == "bullish":
                tp_capped = min(tp_swing, capped_tp)
                reward_points = tp_capped - entry
            else:
                tp_capped = max(tp_swing, capped_tp)
                reward_points = entry - tp_capped
            rr_candidate = reward_points / risk_points
            if rr_candidate >= rr_min:
                tp = tp_capped
                tp_source = tp_src + ("_capped" if tp_capped == capped_tp else "")
                rr = rr_candidate

    # PRIORITE 3 : fixed_rr = RR_TARGET (=2.0)
    if tp is None:
        reward_points = risk_points * rr_target
        tp = entry + reward_points if ob.direction == "bullish" else entry - reward_points
        tp_source = f"fixed_rr{rr_target}"
        rr = rr_target

    if rr < rr_min:
        return None

    lots, risk_usd = compute_position_size(entry, sl, instrument, balance, risk_pct)
    tick_value = INSTRUMENTS[instrument]["tick_value"]
    reward_usd = abs(tp - entry) * tick_value * lots

    return TradeSetup(
        instrument=instrument,
        direction=ob.direction,
        entry_price=float(entry),
        stop_loss=float(sl),
        take_profit=float(tp),
        rr=float(rr),
        risk_points=float(risk_points),
        reward_points=float(reward_points),
        risk_usd=float(risk_usd),
        reward_usd=float(reward_usd),
        position_size_lots=float(lots),
        ob_validation_ts=ob.validation_ts,
        tp_source=tp_source,
    )


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent))
    from bot_v2.concepts.liquidity import find_swings
    from bot_v2.concepts.order_block import detect_order_blocks
    from bot_v2.data_loader import load

    instrument = "XAUUSD"
    df = load(instrument, "M5")
    mask = df.index >= (df.index.max() - pd.Timedelta(days=7))
    df_week = df[mask]
    print(f"{instrument} M5 (7j) : {len(df_week)} bougies\n")

    swings = find_swings(df_week)
    obs = detect_order_blocks(df_week, swings=swings)
    print(f"OB detectes : {len(obs)}")

    setups = []
    rejected = 0
    for ob in obs:
        setup = build_setup_from_ob(df_week, ob, swings, instrument)
        if setup:
            setups.append(setup)
        else:
            rejected += 1

    print(f"Setups valides (RR >= {RR_MIN}) : {len(setups)}")
    print(f"Rejetes : {rejected}")

    print(f"\n=== 5 derniers setups ===")
    for s in setups[-5:]:
        print(
            f"  {s.ob_validation_ts} | {s.direction:8s} | "
            f"entry={s.entry_price:.3f} sl={s.stop_loss:.3f} tp={s.take_profit:.3f} "
            f"| RR={s.rr:.2f} | lots={s.position_size_lots:.2f} "
            f"| risk=${s.risk_usd:.0f} reward=${s.reward_usd:.0f} | {s.tp_source}"
        )
