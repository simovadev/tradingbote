"""MSS Setup — un MSS+FVG transforme en setup tradable (decision user 2026-05-16).

Bible §10 video 07 ICT 2022 :
- Tendance baissiere -> bougie cloture au-dessus d'un swing high = MSS bullish
- Cette cassure contient (idealement) un FVG dans son displacement
- Le prix revient retester ce FVG = ENTRY long

Differences vs OB classique :
- L'OB cherche un groupe inverse + sweep + cloture
- Le MSS cherche un CHANGEMENT DE STRUCTURE confirme

Trade :
- Entry = milieu de la FVG du displacement
- SL = au-dela du swing point casse (avec petit buffer)
- TP = prochain swing HTF dans la direction, capé a RR_MAX
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from bot_v2.concepts.fvg import FVG, detect_fvg
from bot_v2.concepts.liquidity import Swing, find_swings
from bot_v2.concepts.structure import StructureBreak, detect_structure_breaks


Direction = Literal["bullish", "bearish"]


@dataclass(frozen=True)
class MSSSetup:
    """Un setup MSS tradable : MSS confirme + FVG dans le displacement."""
    direction: Direction
    # Le MSS d'origine
    mss: StructureBreak
    # La FVG du displacement (zone de retest)
    fvg: FVG
    # Index/ts ou le retest commence (la 1ere bougie qui touche la FVG)
    retest_index: int | None = None
    retest_ts: pd.Timestamp | None = None
    # Niveaux du setup
    entry_price: float = 0.0     # milieu FVG
    stop_loss: float = 0.0       # au-dela du swing casse
    fvg_top: float = 0.0
    fvg_bottom: float = 0.0


def _displacement_ratio(df: pd.DataFrame, break_idx: int, lookback: int = 14) -> float:
    """Mesure le ratio (taille bougie break) / ATR(14)."""
    if break_idx < lookback:
        return 0.0
    sub = df.iloc[break_idx - lookback:break_idx]
    atr = float((sub["high"] - sub["low"]).mean())
    if atr == 0:
        return 0.0
    bar = df.iloc[break_idx]
    body = abs(float(bar["close"]) - float(bar["open"]))
    return body / atr


def detect_mss_setups(
    df: pd.DataFrame,
    structure_breaks: list[StructureBreak] | None = None,
    swings: list[Swing] | None = None,
    swing_strength: int = 3,        # +strict (5 -> 7 bougies)
    max_bars_to_retest: int = 30,
    sl_buffer_pct: float = 0.0005,
    min_displacement_atr: float = 0.5,   # bougie de break >= 0.5 * ATR
    min_swing_age_bars: int = 5,         # le swing casse doit etre vieux d'au moins 5 bougies
) -> list[MSSSetup]:
    """Detecte les setups MSS tradables.

    Pour chaque MSS avec FVG dans son displacement :
    1. Identifie la FVG du displacement
    2. Cherche le retest (1ere bougie qui touche la FVG)
    3. Construit entry/SL/TP

    Args:
        max_bars_to_retest: si pas de retest dans N bougies, le setup est skip.
        sl_buffer_pct: marge en % au-dela du swing casse pour le SL.
    """
    if structure_breaks is None:
        structure_breaks = detect_structure_breaks(df, swings=swings, swing_strength=swing_strength)

    fvgs = detect_fvg(df)
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values

    setups: list[MSSSetup] = []
    seen_keys: set = set()  # dedup par (direction, break_index)

    for br in structure_breaks:
        # On veut UNIQUEMENT les MSS avec FVG displacement
        if br.kind != "MSS" or not br.has_displacement_fvg:
            continue

        # Filtre 1 : swing age - le swing doit etre etabli (pas un micro-swing recent)
        if (br.break_index - br.swing.index) < min_swing_age_bars:
            continue

        # Filtre 2 : displacement franc sur la bougie de break
        disp_ratio = _displacement_ratio(df, br.break_index)
        if disp_ratio < min_displacement_atr:
            continue

        # Filtre 3 : dedup par (direction, break_index) - un seul MSS par bougie
        key = (br.direction, br.break_index)
        if key in seen_keys:
            continue
        seen_keys.add(key)

        # Trouve la FVG la plus proche dans la fenetre [break-3, break+3] et meme direction
        window_start = max(0, br.break_index - 3)
        window_end = min(len(df), br.break_index + 4)
        candidate_fvgs = [
            f for f in fvgs
            if window_start <= f.center_index < window_end
            and f.direction == br.direction
        ]
        if not candidate_fvgs:
            continue
        # La FVG la plus proche en temps du break_index
        fvg = min(candidate_fvgs, key=lambda f: abs(f.center_index - br.break_index))

        # Niveaux ICT pour MSS bullish (le prix retourne dans la FVG par le HAUT) :
        #   Entry = TOP de la FVG (le prix touche en premier le top du gap en revenant)
        #   SL    = sous le BOTTOM de la FVG + buffer
        # Pour MSS bearish (le prix retourne dans la FVG par le BAS) :
        #   Entry = BOTTOM de la FVG
        #   SL    = au-dessus du TOP de la FVG + buffer
        if br.direction == "bullish":
            entry = fvg.top
            sl = fvg.bottom * (1 - sl_buffer_pct)
            if sl >= entry:
                continue
        else:
            entry = fvg.bottom
            sl = fvg.top * (1 + sl_buffer_pct)
            if sl <= entry:
                continue

        # Cherche le retest dans les max_bars_to_retest bougies suivantes
        retest_idx = None
        retest_ts = None
        scan_start = br.break_index + 1
        scan_end = min(len(df), scan_start + max_bars_to_retest)
        for j in range(scan_start, scan_end):
            if br.direction == "bullish":
                # Retest = bougie qui touche la zone FVG par le haut (low touche fvg.top)
                if lows[j] <= fvg.top and lows[j] >= fvg.bottom:
                    retest_idx = j
                    retest_ts = df.index[j]
                    break
                # Cas ou le prix saute carrement dans la FVG
                if lows[j] < fvg.bottom:
                    retest_idx = j
                    retest_ts = df.index[j]
                    break
            else:
                if highs[j] >= fvg.bottom and highs[j] <= fvg.top:
                    retest_idx = j
                    retest_ts = df.index[j]
                    break
                if highs[j] > fvg.top:
                    retest_idx = j
                    retest_ts = df.index[j]
                    break

        if retest_idx is None:
            continue  # Pas de retest = pas de setup tradable

        setups.append(MSSSetup(
            direction=br.direction,
            mss=br,
            fvg=fvg,
            retest_index=retest_idx,
            retest_ts=retest_ts,
            entry_price=float(entry),
            stop_loss=float(sl),
            fvg_top=float(fvg.top),
            fvg_bottom=float(fvg.bottom),
        ))

    return setups


def mss_setup_to_ob_like(setup: MSSSetup):
    """Adapter : convertit un MSSSetup en objet 'OB-like' pour reutiliser le pipeline.

    Retourne un namespace avec les memes attributs qu'un OrderBlock pour que
    evaluate_ob() puisse l'evaluer sans modification majeure.
    """
    from types import SimpleNamespace

    # On simule un OB avec les bornes de la FVG
    return SimpleNamespace(
        direction=setup.direction,
        group_start_index=setup.mss.swing.index,
        group_end_index=setup.mss.break_index,
        group_start_ts=setup.mss.swing.timestamp,
        group_end_ts=setup.mss.break_ts,
        ob_high=setup.fvg_top,
        ob_low=setup.fvg_bottom,
        ob_open=setup.entry_price,
        ob_close=setup.mss.break_close,
        validation_index=setup.retest_index,
        validation_ts=setup.retest_ts,
        validation_close=float(setup.mss.break_close),
        sweep=setup.mss.swing,  # pas vraiment un sweep mais on reutilise la struct
        group_size=setup.mss.break_index - setup.mss.swing.index,
        is_mss_setup=True,  # flag pour distinguer dans le pipeline
    )


def confirm_ob_with_mss(
    obs: list,
    mss_setups: list[MSSSetup],
    window_bars: int = 10,
) -> list:
    """Garde uniquement les OB CONFIRMES par un MSS recent (setup premium ICT).

    Pour chaque OB, on cherche s'il existe un MSS dans la meme direction
    qui a casse la structure dans une fenetre de [-window_bars, +window_bars]
    autour de la validation de l'OB.

    Args:
        obs: liste d'OrderBlock detectes
        mss_setups: liste de MSSSetup detectes
        window_bars: tolerance en bougies (defaut 10 = ~10 min en M1)

    Returns:
        Sous-ensemble d'obs : ceux qui ont un MSS confirme proche.
    """
    confirmed: list = []
    for ob in obs:
        # Cherche un MSS de meme direction dans la fenetre temporelle
        for mss in mss_setups:
            if mss.direction != ob.direction:
                continue
            # MSS break_index proche de l'OB validation_index
            delta = abs(mss.mss.break_index - ob.validation_index)
            if delta <= window_bars:
                # Setup PREMIUM : OB+MSS confirmes
                confirmed.append(ob)
                break
    return confirmed


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent))
    from bot_v2.data_loader import load
    from bot_v2.concepts.order_block import detect_order_blocks

    df = load("XAUUSD", "M1")
    mask = df.index >= (df.index.max() - pd.Timedelta(days=30))
    df_w = df[mask]
    print(f"XAUUSD M1 (30 derniers jours) : {len(df_w)} bougies")

    obs = detect_order_blocks(df_w, swing_strength=2, max_group_size=2)
    setups = detect_mss_setups(df_w)
    confirmed = confirm_ob_with_mss(obs, setups, window_bars=10)

    print(f"\nOB classiques     : {len(obs)} (~{len(obs)/30:.1f}/jour)")
    print(f"MSS setups        : {len(setups)} (~{len(setups)/30:.1f}/jour)")
    print(f"OB confirmes MSS  : {len(confirmed)} (~{len(confirmed)/30:.1f}/jour)")
    print(f"  - Bullish : {sum(1 for ob in confirmed if ob.direction == 'bullish')}")
    print(f"  - Bearish : {sum(1 for ob in confirmed if ob.direction == 'bearish')}")
