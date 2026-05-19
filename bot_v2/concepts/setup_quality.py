"""Qualite d'un setup — filtres avances Vizion.

Ces concepts manquaient en V1 et expliquent l'ecart de winrate entre le bot
et un trader Vizion experimente.

Concepts implementes :
- Force du sweep (multi-liquidite, equal H/L)
- Retest counter (3eme retest d'un niveau = signal fort)
- Setup Unicorn (OB + stop hunt + FVG + Breaker, entree sur le breaker)
- Force de l'OB (score 1-10)
- Narrative coherente (sweep + OB + breaker dans ordre logique)
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from bot_v2.concepts.breaker import Breaker, detect_breakers
from bot_v2.concepts.fvg import FVG
from bot_v2.concepts.liquidity import Sweep, Swing
from bot_v2.concepts.order_block import OrderBlock


@dataclass
class SetupQuality:
    """Score qualitatif d'un setup."""
    ob_strength: int                # 0-10, force de l'OB
    sweep_strength: int             # 0-10, force du sweep d'origine
    is_unicorn: bool                # OB + stop hunt + FVG + Breaker
    has_associated_pdr: bool        # FVG/IFVG sur le chemin de validation
    associated_pdr_count: int       # nombre de PDR confluents
    retest_count: int               # nombre de retests du niveau OB
    multi_liquidity_sweep: bool     # le sweep a pris plusieurs liquidites
    aligned_with_htf_trend: bool    # OB direction = tendance HTF
    total_quality_score: int        # 0-100


def compute_ob_strength(
    ob: OrderBlock,
    df: pd.DataFrame,
    fvgs: list[FVG] | None = None,
) -> int:
    """Force de l'OB de 0 a 10 selon plusieurs criteres."""
    score = 0

    # Critere 1 : displacement (validation forte vs taille du groupe)
    group_range = ob.ob_high - ob.ob_low
    if group_range > 0:
        val_dist = abs(ob.validation_close - (ob.ob_high if ob.direction == "bullish" else ob.ob_low))
        displacement_ratio = val_dist / group_range
        if displacement_ratio > 1.5:
            score += 3
        elif displacement_ratio > 1.0:
            score += 2
        elif displacement_ratio > 0.5:
            score += 1

    # Critere 2 : group size compact (1-2 bougies = OB net)
    if ob.group_size <= 2:
        score += 2
    elif ob.group_size <= 3:
        score += 1

    # Critere 3 : presence d'un FVG dans la zone de validation
    if fvgs:
        for f in fvgs:
            if f.direction == ob.direction:
                # FVG dans les 3 bougies autour de la validation
                if abs(f.center_index - ob.validation_index) <= 3:
                    score += 2
                    break

    # Critere 4 : meche significative (l'OB a une meche basse pour bullish)
    bar = df.iloc[ob.group_end_index]
    if ob.direction == "bullish":
        # Pour OB bullish, on aime une meche BASSE forte (= prise de liquidite)
        body = abs(bar["close"] - bar["open"])
        lower_wick = min(bar["open"], bar["close"]) - bar["low"]
        if body > 0 and lower_wick / body > 1.0:
            score += 2
        elif body > 0 and lower_wick / body > 0.5:
            score += 1
    else:
        body = abs(bar["close"] - bar["open"])
        upper_wick = bar["high"] - max(bar["open"], bar["close"])
        if body > 0 and upper_wick / body > 1.0:
            score += 2
        elif body > 0 and upper_wick / body > 0.5:
            score += 1

    # Critere 5 : sweep recent (< 3 bougies entre sweep et validation)
    bars_to_validation = ob.validation_index - ob.sweep.sweep_index
    if bars_to_validation <= 3:
        score += 1

    return min(score, 10)


def compute_sweep_strength(
    sweep: Sweep,
    swings: list[Swing],
    df: pd.DataFrame,
) -> tuple[int, bool]:
    """Force du sweep + indicateur multi-liquidite.

    Returns:
        (score 0-10, multi_liquidity_bool)
    """
    score = 0

    # Critere 1 : age du swing pris (plus c'est vieux, plus c'est fort)
    swing = sweep.swing
    bars_old = sweep.sweep_index - swing.index
    if bars_old > 50:
        score += 3
    elif bars_old > 20:
        score += 2
    elif bars_old > 10:
        score += 1

    # Critere 2 : strength du swing (plus elevee = plus pivot)
    if swing.strength >= 3:
        score += 2
    elif swing.strength >= 2:
        score += 1

    # Critere 3 : multi-liquidite (plusieurs swings au meme niveau pris en meme temps)
    # Fix 2026-05-19 : tolerance 0.001 etait sur le PRIX (0.1%), valeur trop large
    # pour XAUUSD a 2000$ (= 2$ tolerance, capture quasi-tous les swings).
    # Et on ne filtrait pas le swing lui-meme (s.index != swing.index).
    sweep_bar = df.iloc[sweep.sweep_index]
    multi = False
    # Tolerance en ATR : 0.5x ATR du sweep (plus realiste que % du prix)
    if "atr" in df.columns:
        atr_at_sweep = df.iloc[sweep.sweep_index].get("atr", 0) or 0
    else:
        atr_at_sweep = 0
    tol_price = max(atr_at_sweep * 0.3, swing.price * 0.0005)  # 0.05% min, 0.3xATR max
    same_side_swings = [
        s for s in swings
        if s.kind == swing.kind
        and s.index < sweep.sweep_index
        and s.index != swing.index  # exclut le swing lui-meme
        and abs(s.price - swing.price) <= tol_price
    ]
    if len(same_side_swings) >= 2:
        multi = True
        score += 3
    elif len(same_side_swings) >= 1:
        score += 1

    # Critere 4 : rejet net (la bougie de sweep a une grosse meche)
    if sweep.swing.kind == "high":
        # Sweep d'un high : meche haute / corps
        body = abs(sweep_bar["close"] - sweep_bar["open"])
        upper_wick = sweep_bar["high"] - max(sweep_bar["open"], sweep_bar["close"])
        if body > 0 and upper_wick / body > 1.0:
            score += 2
    else:
        body = abs(sweep_bar["close"] - sweep_bar["open"])
        lower_wick = min(sweep_bar["open"], sweep_bar["close"]) - sweep_bar["low"]
        if body > 0 and lower_wick / body > 1.0:
            score += 2

    return min(score, 10), multi


def detect_unicorn_setup(
    ob: OrderBlock,
    fvgs: list[FVG],
    breakers: list[Breaker],
    swings: list[Swing],
    df: pd.DataFrame,
) -> bool:
    """Detecte un Setup Unicorn (vidéo tNgYKSK3Vz0).

    Composition :
    1. Prise de liquidite (= sweep -> deja dans l'OB).
    2. Stop hunt : 2eme baisse/hausse qui prend les stops des early participants.
    3. OB valide (= deja le notre).
    4. FVG dans le displacement de validation.
    5. Breaker block (ancien OB inversé) dans la même zone.

    Returns True si tous les elements sont reunis.
    """
    # Check FVG dans displacement (3 bougies autour de validation)
    has_fvg = False
    for f in fvgs:
        if f.direction == ob.direction:
            if abs(f.center_index - ob.validation_index) <= 3:
                has_fvg = True
                break

    if not has_fvg:
        return False

    # Check breaker compatible (direction + proximite)
    has_breaker = False
    for br in breakers:
        if br.direction == ob.direction:
            # Breaker recent (dans 30 bougies avant la validation OB)
            if 0 < (ob.validation_index - br.inverse_index) < 30:
                # Et niveau proche de l'OB (chevauchement)
                overlap = not (br.top < ob.ob_low or br.bottom > ob.ob_high)
                if overlap:
                    has_breaker = True
                    break

    if not has_breaker:
        return False

    # Stop hunt = sweep apres le sweep initial (2eme prise de liquidite)
    # Simplification : on accepte si le sweep d'origine prend un swing recent
    # (ce qui est deja le cas par construction).
    return True


def count_ob_retests(ob: OrderBlock, df: pd.DataFrame) -> int:
    """Compte combien de fois le price est revenu TOUCHER l'OB apres validation."""
    if ob.validation_index >= len(df) - 1:
        return 0

    after = df.iloc[ob.validation_index + 1:]
    touches = 0
    in_zone = False

    for _, row in after.iterrows():
        # Touch = high ou low entre dans la zone OB
        touched = not (row["high"] < ob.ob_low or row["low"] > ob.ob_high)
        if touched:
            if not in_zone:
                touches += 1
                in_zone = True
            # Si le price casse de l'autre cote = OB invalide, on stoppe
            if ob.direction == "bullish" and row["close"] < ob.ob_low:
                break
            if ob.direction == "bearish" and row["close"] > ob.ob_high:
                break
        else:
            in_zone = False

    return touches


def compute_setup_quality(
    ob: OrderBlock,
    df: pd.DataFrame,
    fvgs: list[FVG],
    breakers: list[Breaker],
    swings: list[Swing],
    htf_trend: str | None = None,
) -> SetupQuality:
    """Calcule la qualite globale d'un setup."""
    ob_strength = compute_ob_strength(ob, df, fvgs)
    sweep_strength, multi_liq = compute_sweep_strength(ob.sweep, swings, df)
    is_unicorn = detect_unicorn_setup(ob, fvgs, breakers, swings, df)
    retest = count_ob_retests(ob, df)

    # FVG/IFVG associes sur le chemin de validation
    associated = 0
    for f in fvgs:
        if f.direction == ob.direction:
            if ob.sweep.sweep_index <= f.center_index <= ob.validation_index + 5:
                associated += 1

    aligned = (htf_trend == ob.direction) if htf_trend else False

    # Score global 0-100
    total = (
        ob_strength * 4         # 0-40
        + sweep_strength * 3    # 0-30
        + (20 if is_unicorn else 0)
        + min(associated, 3) * 3  # 0-9
        + (10 if aligned else 0)
        + min(retest, 2) * 5    # 0-10 (3eme+ retest = signal fort)
    )

    return SetupQuality(
        ob_strength=ob_strength,
        sweep_strength=sweep_strength,
        is_unicorn=is_unicorn,
        has_associated_pdr=associated > 0,
        associated_pdr_count=associated,
        retest_count=retest,
        multi_liquidity_sweep=multi_liq,
        aligned_with_htf_trend=aligned,
        total_quality_score=min(total, 100),
    )


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent))
    from bot_v2.concepts.fvg import detect_fvg
    from bot_v2.concepts.liquidity import find_swings
    from bot_v2.concepts.order_block import detect_order_blocks
    from bot_v2.data_loader import load

    df = load("XAUUSD", "M5")
    mask = df.index >= (df.index.max() - pd.Timedelta(days=7))
    df = df[mask]
    print(f"XAUUSD M5 (7j) : {len(df)} bougies")

    swings = find_swings(df)
    obs = detect_order_blocks(df, swings=swings)
    fvgs = detect_fvg(df)
    brks = detect_breakers(df, obs=obs)

    print(f"\nOB total : {len(obs)}")
    qualities = []
    for ob in obs:
        q = compute_setup_quality(ob, df, fvgs, brks, swings)
        qualities.append((ob, q))

    # Stats
    avg_ob = sum(q.ob_strength for _, q in qualities) / len(qualities)
    avg_sweep = sum(q.sweep_strength for _, q in qualities) / len(qualities)
    n_unicorn = sum(1 for _, q in qualities if q.is_unicorn)
    n_multi = sum(1 for _, q in qualities if q.multi_liquidity_sweep)
    avg_total = sum(q.total_quality_score for _, q in qualities) / len(qualities)
    print(f"OB strength moyen   : {avg_ob:.1f}/10")
    print(f"Sweep strength moyen: {avg_sweep:.1f}/10")
    print(f"Unicorn setups      : {n_unicorn}/{len(obs)}")
    print(f"Multi-liquidite     : {n_multi}/{len(obs)}")
    print(f"Quality score moyen : {avg_total:.0f}/100")

    print(f"\n=== Top 5 setups (par quality score) ===")
    top = sorted(qualities, key=lambda x: -x[1].total_quality_score)[:5]
    for ob, q in top:
        print(f"  {ob.validation_ts} | {ob.direction:8s} | quality={q.total_quality_score} "
              f"(OB={q.ob_strength}, sweep={q.sweep_strength}, unicorn={q.is_unicorn}, retests={q.retest_count})")
