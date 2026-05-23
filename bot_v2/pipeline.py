"""Pipeline Vizion — CASCADE de decision (bible §12).

C'est le coeur du bot : on enchaine les verifications dans l'ORDRE Vizion.
Si une etape ELIMINATOIRE echoue, le trade est REJETE (pas negocie).
Si une etape BONUS echoue, le score baisse mais le trade reste valide.

Ordre exact (bible §0 + §12) :
1. **Bias Daily** (HTF) — bullish/bearish/neutral. Si neutral => REJETE.
2. **Killzone active** — hors KZ => REJETE.
3. **Imbrication TF** — OB LTF doit etre dans un OB HTF de meme direction
   => si pas d'OB HTF compatible, REJETE.
4. **OB strict Vizion** — bougies consecutives + sweep + validation cloture.
5. **Discount / Premium** — OB long en Discount, OB short en Premium => sinon REJETE.
6. **PO3 / AMD** (bonus) — phase distribution = bonus de score.
7. **SMT divergence** (bonus, decision user 2026-05-15) — bonus de score.
8. **Breaker / setup Unicorn** (bonus) — confluence supplementaire.
9. **TradeSetup** — Entry/SL/TP/RR. RR < 1.5 => REJETE.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from bot_v2.concepts.breaker import detect_breakers
from bot_v2.concepts.daily_bias import DailyBias, build_d1_from_h1, compute_daily_bias
from bot_v2.concepts.feu_vert import FeuVert, check_feu_vert_h1
from bot_v2.concepts.time_levels import check_above_open_midnight
from bot_v2.concepts.discount_premium import (
    FibRange,
    latest_fib_range,
    ob_zone_check,
)
from bot_v2.concepts.fvg import detect_fvg
from bot_v2.concepts.htf_swings import HTFSwing, collect_htf_swings
from bot_v2.concepts.killzones import killzone_at
from bot_v2.concepts.liquidity import find_swings, find_sweeps
from bot_v2.concepts.market_phase import (
    analyze_phase,
    has_displacement_at_validation,
)
from bot_v2.concepts.order_block import OrderBlock, detect_order_blocks
from bot_v2.concepts.po3_amd import analyze_po3
from bot_v2.concepts.session_bias import SessionContext, get_session_context, session_play_score
from bot_v2.concepts.setup_quality import SetupQuality, compute_setup_quality
from bot_v2.concepts.smt import detect_smt
from bot_v2.concepts.structure import detect_structure_breaks, detect_trend
from bot_v2.config import SMT_BONUS_POINTS, SMT_PAIRS, TF_PARENT, get_param
from bot_v2.data_loader import load
from bot_v2.trade_setup import TradeSetup, build_setup_from_ob


Verdict = Literal["TRADE", "REJECTED"]


@dataclass
class PipelineResult:
    """Resultat de la pipeline pour 1 OB candidat."""
    ob: OrderBlock
    verdict: Verdict
    # Etapes (True = passee, False = echec, None = non evaluee)
    daily_bias_ok: bool | None = None
    killzone_ok: bool | None = None
    htf_alignment_ok: bool | None = None
    discount_premium_ok: bool | None = None
    rr_ok: bool | None = None
    # Bonus (n'eliminent pas)
    po3_ok: bool | None = None
    smt_ok: bool | None = None
    breaker_present: bool | None = None
    # Resultats / details
    daily_bias: DailyBias | None = None
    feu_vert: FeuVert | None = None       # bible V2 §2 : feu vert H1 si daily neutre
    killzone_name: str | None = None
    parent_ob: OrderBlock | None = None
    fib_range: FibRange | None = None
    quality: SetupQuality | None = None    # qualite avancee (Unicorn, force OB, etc.)
    score: int = 0                  # 0-100
    rejection_reason: str = ""
    trade_setup: TradeSetup | None = None
    # Confluences textuelles
    confluences: list[str] = field(default_factory=list)
    # V10 : valeurs continues riches exposees au ML (avant : perdues dans confluences)
    disp_ratio: float = 0.0          # force du displacement de validation en xATR
    vol_ratio: float = 1.0           # ATR14 / ATR100 (volatilite courte/longue)
    po3_body_pct: float = 0.5        # corps / range de la bougie PO3 HTF
    po3_upper_wick: float = 0.0      # meche haute PO3 (ratio range)
    po3_lower_wick: float = 0.0      # meche basse PO3 (ratio range)
    po3_aligned: int = 0             # PO3 sense == direction OB
    po3_htf2_aligned: int = 0        # PO3 grand-parent aligne
    fib_level: float = 0.5           # position de l'OB dans le range Fibo (0-1)


def evaluate_ob(
    ob: OrderBlock,
    df_ltf: pd.DataFrame,
    df_htf: pd.DataFrame,
    df_d1: pd.DataFrame,
    instrument: str,
    ltf_name: str = "M1",
    htf_name: str = "M15",
    df_htf2: pd.DataFrame | None = None,    # grand-parent (H1 si chaine M1->M15->H1)
    htf2_name: str | None = "H1",
    correlated_dfs: dict[str, tuple[pd.DataFrame, str]] | None = None,
    htf_swings: list[HTFSwing] | None = None,
    df_h1: pd.DataFrame | None = None,     # bible V2 §2 : pour Feu Vert H1
    min_score: int = 0,
    min_quality: int = 0,
    cache: dict | None = None,             # OPTIM 1 : pre-calculs partages (FVG/swings/breakers/etc)
    use_dynamic_sl_tp: bool = False,       # mode ICT pur : SL/TP dynamique sur liquidites
) -> PipelineResult:
    """Evalue UN OB candidat avec toute la cascade Vizion.

    Args:
        ob: l'Order Block detecte sur le LTF.
        df_ltf: dataframe LTF (ex. M5).
        df_htf: dataframe HTF parent (ex. H1).
        df_d1: dataframe daily pour le daily bias.
        instrument: nom de l'actif.
        ltf_name, htf_name: noms des timeframes (pour info).
        correlated_dfs: {asset_name: (df, correlation_type)} pour SMT.
    """
    res = PipelineResult(ob=ob, verdict="REJECTED")

    # ========== 1. DAILY BIAS ==========
    # Bible §14.1 ligne 910 : "Trader contre le bias daily = trash" (BKGoNf6vhRY).
    # Analyse trades 2026-05-15 : 3/5 LOSS EURUSD avaient bias contraire.
    # On remet le bias CONTRAIRE en ELIMINATOIRE. Neutral = on trade (pas de bonus).
    target_date = ob.validation_ts.normalize()
    bias = compute_daily_bias(df_d1, target_date)
    res.daily_bias = bias
    if bias is None:
        # Pas de daily bias calculable -> on continue (manque de data)
        res.daily_bias_ok = None
        res.confluences.append("daily_bias=unknown")
    elif bias.bias == ob.direction:
        res.daily_bias_ok = True
        res.score += 25
        res.confluences.append(f"daily_bias={bias.bias}")
    elif bias.bias == "neutral":
        # Bias neutre : bible V2 §2 - on cherche le "Feu Vert H1" comme BOOST.
        # Si feu vert detecte -> score boost (15-20 pts).
        # Sinon -> on laisse passer (comme avant), juste pas de bonus.
        # Decision user 2026-05-15 : on rejette PAS pour eviter de bloquer en intraday.
        if df_h1 is not None:
            fv = check_feu_vert_h1(df_h1, df_d1, ob.validation_ts, ob.direction)
            res.feu_vert = fv
            if fv.is_green:
                res.daily_bias_ok = True
                res.score += 15
                label = f"feu_vert_H1({fv.swept_level})"
                if fv.in_killzone:
                    res.score += 5
                    label += "+KZ"
                res.confluences.append(label)
            else:
                res.daily_bias_ok = None
                res.confluences.append("daily_bias=neutral_sans_feu_vert")
        else:
            res.daily_bias_ok = None
            res.confluences.append("daily_bias=neutral")
    else:
        # ABLATION v7 (user 2026-05-16) : daily_bias contraire devient penalite -15 pts.
        # Avant : REJET eliminatoire (58% des OB rejetes).
        # Maintenant : feature ML 'daily_bias_aligned=0' + penalite score.
        # Le ML decidera si le setup compense malgre le bias contraire.
        res.daily_bias_ok = False
        res.score -= 15
        res.confluences.append(f"daily_bias_contraire_{bias.bias}_toléré")

    # ========== 2. KILLZONE (decision user 2026-05-16 : KZ devient SCORE, pas FILTRE) ==========
    # Exception FOREX (user 2026-05-17) : blocage 21h-02h NY (overnight US, volatilite plate).
    # Exception USDJPY (user 2026-05-21) : paire japonaise, tres active pendant la
    # session Asia (Tokyo). On la laisse tradable 24/24.
    kz = killzone_at(ob.validation_ts)
    res.killzone_name = kz
    is_forex = get_param(instrument, "is_forex", False)

    # Forex : blocage horaire NY 21h-02h (volatilite morte).
    # USDJPY exclu : la paire JPY est active en Asia.
    # V12 test (2026-05-23) : filtre DESACTIVE pour voir si forex/or trade enfin.
    # Si oui -> filtre etait trop strict, on l'ajuste. Si non -> ML severe.
    # if is_forex and instrument != "USDJPY":
    #     from bot_v2.concepts.killzones import to_ny_time
    #     ny_ts = to_ny_time(ob.validation_ts)
    #     ny_hour = ny_ts.hour
    #     if ny_hour >= 21 or ny_hour < 2:
    #         res.killzone_ok = False
    #         res.rejection_reason = f"Forex bloque overnight US ({ny_hour}h NY)"
    #         return res

    if kz is None:
        res.killzone_ok = False
        res.confluences.append("hors_KZ")
    elif kz == "NY_Lunch":
        res.killzone_ok = False
        res.score -= 5
        res.confluences.append(f"killzone={kz}_malus")
    else:
        res.killzone_ok = True
        res.score += 15
        res.confluences.append(f"killzone={kz}")
        if kz in ("London", "NY_AM"):
            res.score += 8
            res.confluences.append("KZ_premium")

    # 2-bis. SESSION PLAY (continuation ou reversal selon mvt deja fait)
    # V3.5 (2026-05-19) : session_score == 0 n'est plus un REJET mais un malus -8.
    # Raison : trop de bons OB rejetes en debut de session (avant qu'une direction soit claire).
    session_ctx = get_session_context(df_ltf, ob.validation_ts)
    session_score = session_play_score(session_ctx, ob.direction)
    if session_score == 0:
        res.score -= 8
        res.confluences.append("session_sans_direction_malus")
    else:
        res.score += session_score
    if session_ctx.session_direction == ob.direction:
        res.confluences.append(f"session_continuation_{session_ctx.session_name}")
    else:
        res.confluences.append(f"session_reversal_{session_ctx.session_name}")

    # ========== 2-ter. PHASE DE MARCHE (bible §11.1 Pm29OIifOns) ==========
    # V5 (user 2026-05-20) : phase manipulation passe en malus (volatilite forte
    # = retournement violent ICT, pattern cible). Accumulation reste REJET
    # (marche vraiment mort, pas de direction).
    phase = analyze_phase(df_ltf, ob.validation_index, lookback=20)
    if phase.phase == "accumulation":
        # Bible §11.1 : marche mort, on ne trade pas
        res.rejection_reason = f"Phase accumulation (range trop serre) — {phase.reason}"
        return res
    if phase.phase == "expansion":
        res.score += 15
        res.confluences.append("phase_expansion")
    elif phase.phase == "reversal":
        res.score += 18
        res.confluences.append("phase_reversal")
    elif phase.phase == "manipulation":
        # V5 : autorise (retournement violent = pattern cible user)
        res.score -= 8
        res.confluences.append("phase_manipulation_malus")

    # ========== 2-quater. DISPLACEMENT - V4 : feature ML uniquement (user 2026-05-20) ==========
    # V3.5 : filtre dur min_displacement_atr -> rejetait setups en volatilite
    # V4   : on calcule le ratio mais on N'ELIMINE PAS. Le ML apprend si un displacement
    #        faible est bon ou mauvais SELON le contexte (vol_ratio, ATR, etc.).
    _, disp_ratio = has_displacement_at_validation(
        df_ltf, ob.validation_index, ob.direction,
        min_displacement_ratio=0.0,  # pas de seuil, juste calcul du ratio
    )
    # Calcule volatilite courte/longue pour feature ML
    vol_ratio = 1.0
    if ob.validation_index >= 100:
        atr_14 = float((df_ltf.iloc[ob.validation_index-14:ob.validation_index]["high"]
                        - df_ltf.iloc[ob.validation_index-14:ob.validation_index]["low"]).mean())
        atr_100 = float((df_ltf.iloc[ob.validation_index-100:ob.validation_index]["high"]
                        - df_ltf.iloc[ob.validation_index-100:ob.validation_index]["low"]).mean())
        if atr_100 > 0:
            vol_ratio = atr_14 / atr_100
    res.confluences.append(f"displacement={disp_ratio:.2f}xATR")
    res.confluences.append(f"vol_ratio={vol_ratio:.2f}")
    # V10 : expose les valeurs continues au ML
    res.disp_ratio = float(disp_ratio)
    res.vol_ratio = float(vol_ratio)
    # Bonus de score si displacement fort, malus si faible (mais pas REJET)
    if disp_ratio >= 1.0:
        res.score += 12
    elif disp_ratio >= 0.5:
        res.score += 6
    else:
        res.score -= 5

    # ========== 3. IMBRICATION TF (contexte HTF Vizion §2.5 rdeAjnVdfRM) ==========
    # Methode "entonnoir" Vizion : on exige un FEU VERT HTF (= un OB HTF recent de
    # meme direction) AVANT de descendre en LTF. Pas un chevauchement strict — c'est
    # plutot un "contexte HTF qui pointe dans cette direction".
    tol_pct = get_param(instrument, "parent_ob_tolerance_pct", 0.003)
    obs_htf_cache = cache.get("obs_htf") if cache else None
    parent_ob = _find_parent_ob(
        ob, df_ltf, df_htf, ltf_name,
        tolerance_pct=tol_pct, obs_htf_cache=obs_htf_cache,
    )
    res.parent_ob = parent_ob
    # V3.5 (2026-05-19) : parent OB HTF absent n'est plus REJET mais malus -15.
    # Raison : tous les OB LTF n'ont pas forcement un OB HTF de meme direction recent.
    # Le ML apprendra que les setups sans parent OB ont moins de chances.
    if parent_ob is None:
        res.htf_alignment_ok = False
        res.score -= 15
        res.confluences.append(f"no_parent_ob_{htf_name}_malus")
    else:
        res.htf_alignment_ok = True
        res.score += 20
        res.confluences.append(f"parent_ob_{htf_name}")

    # V3.5 (2026-05-19) : grand-parent OB n'est plus REJET, juste bonus si present.
    # Raison : exiger une chaine TF complete M1->M15->H1 perd ~30% des setups valides.
    if df_htf2 is not None and parent_ob is not None:
        gp_tol = tol_pct * 2
        obs_htf2_cache = cache.get("obs_htf2") if cache else None
        grandparent_ob = _find_parent_ob(
            parent_ob, df_htf, df_htf2, htf_name, tolerance_pct=gp_tol,
            obs_htf_cache=obs_htf2_cache,
        )
        if grandparent_ob is not None:
            res.score += 15
            res.confluences.append(f"grandparent_ob_{htf2_name}")
        else:
            res.score -= 5
            res.confluences.append(f"no_grandparent_{htf2_name}_malus")

    # ========== 4. DISCOUNT / PREMIUM ==========
    # Bible §6.2 : zone calculee sur le DERNIER MOUVEMENT DIRECTIONNEL PROPRE.
    # Correction 2026-05-15 : on calcule le range Fibo AU MOMENT de la validation
    # de l'OB (pas au present), sinon le range obsolete invalide les OB anciens.
    df_htf_at_ob = df_htf[df_htf.index <= ob.validation_ts]
    fib = latest_fib_range(df_htf_at_ob, swing_strength=2)
    res.fib_range = fib
    if fib is None:
        res.discount_premium_ok = False
        res.rejection_reason = "Range Fibo non calculable"
        return res
    # Iter 14 : test D/P encore plus strict (0.35/0.65).
    ob_mid = (ob.ob_high + ob.ob_low) / 2
    if fib.range_size > 0:
        fib_level = (ob_mid - fib.low_price) / fib.range_size  # 0 a 1
        in_good_zone = (
            (ob.direction == "bullish" and fib_level <= 0.35)
            or (ob.direction == "bearish" and fib_level >= 0.65)
        )
    else:
        fib_level = 0.5
        in_good_zone = True  # range nul -> on laisse passer
    # V10 : expose la position Fibo continue au ML (avant : seulement has_good_zone 0/1)
    res.fib_level = float(fib_level)
    # V3.5 (2026-05-19) : mauvaise zone P/D n'est plus REJET mais malus -10.
    # Raison : 30-50% des OB tombent en mauvaise zone Fibo, mais certains
    # peuvent quand meme fonctionner (le ML decidera).
    if not in_good_zone:
        res.discount_premium_ok = False
        res.score -= 10
        zone = fib.zone_of(ob_mid)
        res.confluences.append(f"mauvaise_zone_{zone}_malus")
    else:
        res.discount_premium_ok = True
        res.score += 15
        res.confluences.append(f"zone={fib.zone_of(ob_mid)}")

    # ========== BONUS : PO3 (HTF + grand-parent) ==========
    htf_bar = _get_htf_bar_containing(df_htf, ob.validation_ts)
    if htf_bar is not None:
        po3 = analyze_po3(htf_bar)
        if po3.phase == "distribution" and po3.sense == ob.direction:
            res.po3_ok = True
            res.score += 10
            res.confluences.append(f"po3_distribution_{po3.sense}_{htf_name}")
        elif po3.phase == "manipulation" and po3.sense == ob.direction:
            res.po3_ok = True
            res.score += 5
            res.confluences.append(f"po3_manipulation_{htf_name}")
        else:
            res.po3_ok = False
        # V10 : expose les valeurs continues PO3 au ML (avant : seulement has_po3_dist 0/1)
        res.po3_body_pct = float(getattr(po3, "body_pct", 0.5))
        res.po3_upper_wick = float(getattr(po3, "upper_wick", 0.0))
        res.po3_lower_wick = float(getattr(po3, "lower_wick", 0.0))
        res.po3_aligned = int(po3.sense == ob.direction)

    # PO3 sur le grand-parent (D1 / H4 / H1 selon chaine)
    if df_htf2 is not None:
        htf2_bar = _get_htf_bar_containing(df_htf2, ob.validation_ts)
        if htf2_bar is not None:
            po3_2 = analyze_po3(htf2_bar)
            res.po3_htf2_aligned = int(po3_2.sense == ob.direction)
            if po3_2.sense == ob.direction:
                res.score += 8
                res.confluences.append(f"po3_{po3_2.phase}_{htf2_name}")

    # ========== BONUS : SMT ==========
    # OPTIMISATION (user 2026-05-16) : slice les df a +/- 2h autour de l'OB
    # avant detect_smt. Le filtre delta < 3600s qui suit jetait deja tout
    # SMT > 1h, donc scanner 3 mois de bougies etait du gaspillage.
    # Gain ~1000x sur cette etape (260k bougies -> 240 bougies par appel).
    if correlated_dfs:
        smt_found = False
        # FIX V9 (2026-05-22) : fenetre stricte fin a validation_ts (avant : +5min = leak)
        smt_window = pd.Timedelta(hours=2)
        ob_ts = ob.validation_ts
        win_start = ob_ts - smt_window
        win_end = ob_ts
        df_ltf_win = df_ltf.loc[win_start:win_end]
        if len(df_ltf_win) >= 5:  # Sinon pas assez de bougies pour des swings
            for corr_name, (df_c, corr_type) in correlated_dfs.items():
                df_c_win = df_c.loc[win_start:win_end]
                if len(df_c_win) < 5:
                    continue
                smts = detect_smt(
                    df_ltf_win, df_c_win, corr_type, instrument, corr_name,
                    swing_strength=2,
                )
                # On cherche un SMT dans la meme direction que l'OB, recent
                for s in smts:
                    if s.direction == ob.direction:
                        delta = abs((s.primary_extreme_ts - ob_ts).total_seconds())
                        if delta < 3600:  # SMT dans la derniere heure de l'OB
                            smt_found = True
                            res.confluences.append(f"smt_{corr_name}")
                            break
                if smt_found:
                    break
        res.smt_ok = smt_found
        if smt_found:
            res.score += SMT_BONUS_POINTS
        # ABLATION user 2026-05-16 : SMT NAS100 retiree comme filtre obligatoire.
        # Avant : SMT obligatoire pour NAS100 (forensic iter 15 - stat petite).
        # Maintenant : SMT reste feature ML (has_smt=0/1), le ML decide.
        # Coherent avec ablations daily_bias/KZ qui ont marche sur XAUUSD v7.

    # ========== BONUS : Breaker (bible V2 §7 - KZ OBLIGATOIRE) ==========
    # Video 01 YAhGt8tmfCY : un BB hors killzone est INVALIDE.
    # On ne compte le breaker comme confluence QUE s'il s'est forme en killzone.
    # FIX V9 (2026-05-22) : avant abs() = leak +/- 20 bougies. Maintenant : breaker
    # forme AVANT ou A validation_index uniquement.
    brks = cache["breakers_ltf"] if cache and "breakers_ltf" in cache else detect_breakers(df_ltf)
    for br in brks:
        if br.direction == ob.direction and 0 <= (ob.validation_index - br.inverse_index) < 20:
            # Bible V2 §7 : verifier que le BB s'est forme en killzone
            br_ts = df_ltf.index[br.inverse_index]
            br_in_kz = killzone_at(br_ts) is not None
            if br_in_kz:
                res.breaker_present = True
                res.score += 10
                res.confluences.append("breaker_in_KZ")
            else:
                # BB hors KZ = pas de bonus, on note quand meme
                res.confluences.append("breaker_hors_KZ_ignore")
            break
    if res.breaker_present is None:
        res.breaker_present = False

    # ========== FILTRE : cible algo deja prise (bible V2 §13) - LIVE SEULEMENT ==========
    # Video 06 : si la cible (SSL/BSL) du leg parent est deja prise, ignorer setups.
    # NOTE : ce filtre necessite des swings HTF anterieurs a la validation OB.
    # On verifie via les HTF swings : si entre le swing parent et la validation OB,
    # un swing oppose HTF a ete pris, alors le leg est consume.
    if htf_swings:
        leg_consumed = False
        # Pour un OB bullish : leg parent = depuis dernier swing low jusqu'a maintenant.
        # Si entre le swing low parent et la validation OB, un swing high HTF a ete pris,
        # le leg algo est termine.
        for hs in htf_swings:
            if hs.swing.timestamp >= ob.validation_ts:
                continue
            if ob.direction == "bullish" and hs.swing.kind == "high":
                # Le swing high a-t-il deja ete pris (price > swing.price) avant validation OB ?
                sub = df_ltf[(df_ltf.index > hs.swing.timestamp) & (df_ltf.index < ob.validation_ts)]
                if len(sub) > 0 and (sub["high"] > hs.swing.price).any():
                    leg_consumed = True
                    break
            elif ob.direction == "bearish" and hs.swing.kind == "low":
                sub = df_ltf[(df_ltf.index > hs.swing.timestamp) & (df_ltf.index < ob.validation_ts)]
                if len(sub) > 0 and (sub["low"] < hs.swing.price).any():
                    leg_consumed = True
                    break
        if leg_consumed:
            res.confluences.append("leg_target_already_taken")
            # Penalise sans rejeter (conservation backtest)
            res.score -= 8

    # ========== BONUS : Open Midnight NY (bible V2 §6) ==========
    # Video 05 : pivot intraday distinct du Daily Open NY 18h.
    # Pour un long, l'OB ne devrait PAS etre sous Open Midnight NY.
    ob_mid_price = (ob.ob_high + ob.ob_low) / 2
    omn_ok, omn_price = check_above_open_midnight(
        df_ltf, ob_mid_price, ob.validation_ts, ob.direction,
    )
    if omn_price is not None:
        if omn_ok:
            res.score += 5
            res.confluences.append("respecte_OpenMidnightNY")
        else:
            res.confluences.append("sous_OpenMidnightNY_biais_affaibli")

    # ========== BONUS : MSS avec FVG (bible V2 §10) ==========
    # Video 07 : "S'il n'y a pas de FVG, il n'y a pas de MSS." On cherche un MSS
    # recent dans la meme direction que l'OB et on verifie qu'il a un FVG dans
    # son displacement.
    try:
        breaks = cache["structure_breaks"] if cache and "structure_breaks" in cache else detect_structure_breaks(df_ltf)
        mss_aligned = [
            b for b in breaks
            if b.kind == "MSS"
            and b.direction == ob.direction
            and 0 <= ob.validation_index - b.break_index <= 15
        ]
        if mss_aligned:
            last_mss = mss_aligned[-1]
            if last_mss.has_displacement_fvg:
                res.score += 10
                res.confluences.append("MSS_with_FVG")
            else:
                # MSS sans FVG = invalide selon bible V2 §10
                res.score -= 5
                res.confluences.append("MSS_sans_FVG_penalise")
    except Exception:
        pass

    # ========== Synchronisation OB + FVG (bible V2 §3 - iter 9 STRICT) ==========
    # Video 06 tjkJoBmT2Gs : "OB valide AVEC creation d'un FVG lors de la
    # validation = high probability". La bougie qui valide l'OB doit etre la
    # bougie centrale d'un FVG meme sens.
    # Iter 9 : sync OBLIGATOIRE (avant : juste bonus +12).
    # V3.5 (2026-05-19) : FVG sync n'est plus OBLIGATOIRE mais BONUS.
    # Raison : exiger une FVG dans ±1 bougie de validation OB rejette 60% des setups.
    # Beaucoup d'OB valides n'ont pas de FVG sync mais marchent (le ML decidera).
    # FIX V9 (2026-05-22) : avant abs() = leak +/- 1 bougie. Maintenant : FVG forme
    # AVANT ou A validation_index uniquement.
    fvgs_for_sync = cache["fvgs_ltf"] if cache and "fvgs_ltf" in cache else detect_fvg(df_ltf)
    sync_found = False
    for fvg in fvgs_for_sync:
        if fvg.direction != ob.direction:
            continue
        if 0 <= ob.validation_index - fvg.center_index <= 1:
            sync_found = True
            res.score += 12
            res.confluences.append("OB_FVG_sync_high_proba")
            break
    if not sync_found:
        res.score -= 5
        res.confluences.append("no_FVG_sync_malus")

    # ========== 5. QUALITE AVANCEE (Unicorn, force OB, retests...) ==========
    swing_strength = get_param(instrument, "swing_strength_m1", 2)
    if cache and "swings_ltf" in cache:
        swings_ltf = cache["swings_ltf"]
        fvgs_ltf = cache["fvgs_ltf"]
        breakers_ltf = cache["breakers_ltf"]
        htf_trend = cache["htf_trend"]
        sweeps_ltf = cache.get("sweeps_ltf")  # peut etre None
    else:
        swings_ltf = find_swings(df_ltf, strength=swing_strength)
        fvgs_ltf = detect_fvg(df_ltf)
        breakers_ltf = detect_breakers(df_ltf)
        htf_trend = detect_trend(swings_ltf, lookback=6)
        sweeps_ltf = None

    # Cache miss pour sweeps -> les calculer (necessaires pour SL dynamique)
    if use_dynamic_sl_tp and sweeps_ltf is None:
        sweeps_ltf = find_sweeps(df_ltf, swings_ltf)

    quality = compute_setup_quality(
        ob, df_ltf, fvgs_ltf, breakers_ltf, swings_ltf,
        htf_trend=htf_trend,
    )
    res.quality = quality
    # Le score qualite (0-100) est integre au score global, ponderé 50%
    res.score += int(quality.total_quality_score * 0.5)
    if quality.is_unicorn:
        res.confluences.append("UNICORN")
    if quality.multi_liquidity_sweep:
        res.confluences.append("multi_liq_sweep")
    if quality.retest_count >= 3:
        res.confluences.append(f"retests={quality.retest_count}")
    res.confluences.append(f"OB_str={quality.ob_strength}/10")
    res.confluences.append(f"sweep_str={quality.sweep_strength}/10")

    if quality.total_quality_score < min_quality:
        res.verdict = "REJECTED"
        res.rejection_reason = f"Quality {quality.total_quality_score} < min {min_quality}"
        return res

    # ========== 6. RR / TradeSetup ==========
    setup = build_setup_from_ob(
        df_ltf, ob, swings_ltf, instrument,
        htf_swings=htf_swings,
        sweeps=sweeps_ltf,
        use_dynamic_sl_tp=use_dynamic_sl_tp,
    )
    if setup is None:
        res.rr_ok = False
        res.rejection_reason = "RR insuffisant ou setup invalide"
        return res
    res.rr_ok = True
    res.trade_setup = setup

    # ========== 7. FILTRE SCORE MIN GLOBAL ==========
    if res.score < min_score:
        res.verdict = "REJECTED"
        res.rejection_reason = f"Score {res.score} < min {min_score} (setup pas assez premium)"
        return res

    res.verdict = "TRADE"
    return res


def _find_parent_ob(
    ob_ltf: OrderBlock,
    df_ltf: pd.DataFrame,
    df_htf: pd.DataFrame,
    ltf_name: str,
    max_bars_lookback: int = 50,
    tolerance_pct: float = 0.003,
    obs_htf_cache: list[OrderBlock] | None = None,
) -> OrderBlock | None:
    """Cherche un OB HTF "contexte" pour l'OB LTF (Vizion §2.5 rdeAjnVdfRM).

    Methode "entonnoir" Vizion : on exige un FEU VERT HTF, pas une boite qui contient.
    "Pas de feu vert HTF, pas de descente en LTF."

    Strategie en 3 passes (du plus strict au plus permissif) :
    1. Chevauchement strict des zones (cas ideal : OB LTF physiquement dans OB HTF).
    2. OB HTF recent de meme direction, proche en prix (tolerance_pct adaptee a l'actif).
    3. OB HTF recent de meme direction, pas encore invalide, dans la fenetre de lookback
       (= "il y a un contexte HTF actif dans cette direction").

    Args:
        tolerance_pct: tolerance prix par actif (config.INSTRUMENT_PARAMS).
                       Forex : 0.0005-0.001. Or : 0.002. Indices : 0.005.
    """
    obs_htf = obs_htf_cache if obs_htf_cache is not None else detect_order_blocks(df_htf)
    ob_ltf_ts = ob_ltf.validation_ts

    # Pre-filtre : OB HTF de meme direction, valide AVANT le LTF
    candidates = [
        ob_h for ob_h in obs_htf
        if ob_h.direction == ob_ltf.direction and ob_h.validation_ts <= ob_ltf_ts
    ]
    if not candidates:
        return None

    # Pre-filtre temporel : dans la fenetre de lookback.
    # OPTIM (2026-05-16) : searchsorted O(log N) au lieu de filtres pandas O(N).
    # Profilage a montre que cette fonction representait 88% du temps de
    # evaluate_ob avec un nombre d'appels de ~925 par OB LTF. Speedup ~100x.
    htf_idx = df_htf.index
    # Borne haute (commune a tous les candidats) :
    # nombre de bougies <= ob_ltf_ts
    n_le_ltf = htf_idx.searchsorted(ob_ltf_ts, side="right")

    def bars_apart(ob_h: OrderBlock) -> int:
        # nombre de bougies > ob_h.validation_ts
        n_gt_h = htf_idx.searchsorted(ob_h.validation_ts, side="right")
        return max(0, n_le_ltf - n_gt_h)

    candidates = [ob_h for ob_h in candidates if bars_apart(ob_h) <= max_bars_lookback]
    if not candidates:
        return None

    # PASSE 1 : chevauchement strict des zones
    for ob_h in candidates:
        overlap = not (ob_h.ob_high < ob_ltf.ob_low or ob_h.ob_low > ob_ltf.ob_high)
        if overlap:
            return ob_h

    # PASSE 2 : OB HTF proche en prix selon tolerance par actif
    mid_ltf = (ob_ltf.ob_high + ob_ltf.ob_low) / 2
    closest: OrderBlock | None = None
    closest_dist = float("inf")
    for ob_h in candidates:
        mid_htf = (ob_h.ob_high + ob_h.ob_low) / 2
        if mid_ltf == 0:
            continue
        dist_pct = abs(mid_htf - mid_ltf) / mid_ltf
        if dist_pct <= tolerance_pct and dist_pct < closest_dist:
            closest_dist = dist_pct
            closest = ob_h
    if closest is not None:
        return closest

    # PASSE 3 : il existe un OB HTF de meme direction NON INVALIDE dans la fenetre
    # (= contexte HTF actif). On verifie qu'il n'a pas ete invalide en cloture.
    # Vizion : "Time Frame Alignement = il faut TOUJOURS un feu vert en HTF" — un OB
    # H1 dans la meme direction qui n'a pas ete invalide = feu vert.
    most_recent: OrderBlock | None = None
    closes = df_htf["close"].values
    for ob_h in candidates:
        # Verifie que l'OB n'a pas ete invalide en cloture entre sa validation et ob_ltf_ts
        invalidated = False
        mask = (df_htf.index > ob_h.validation_ts) & (df_htf.index <= ob_ltf_ts)
        sub = df_htf[mask]
        if len(sub) > 0:
            if ob_h.direction == "bullish":
                invalidated = bool((sub["close"] < ob_h.ob_low).any())
            else:
                invalidated = bool((sub["close"] > ob_h.ob_high).any())
        if invalidated:
            continue
        if most_recent is None or ob_h.validation_ts > most_recent.validation_ts:
            most_recent = ob_h
    return most_recent


def _get_htf_bar_containing(df_htf: pd.DataFrame, ts: pd.Timestamp) -> pd.Series | None:
    """Retourne la bougie HTF qui contient le timestamp ts."""
    matches = df_htf[df_htf.index <= ts]
    if len(matches) == 0:
        return None
    return matches.iloc[-1]


def run_pipeline(
    instrument: str,
    ltf_name: str = "M1",
    htf_name: str = "M15",
    htf2_name: str | None = "H1",
    days: int = 7,
    min_score: int | None = None,
    min_quality: int | None = None,
) -> list[PipelineResult]:
    """Lance le pipeline avec la CHAINE TF VIZION : D1 -> H1 -> M15 -> M1.

    Filtres pour atteindre 1-2 trades premium/actif/jour :
    - Si min_score/min_quality est None, on utilise les seuils PAR ACTIF
      definis dans INSTRUMENT_PARAMS (decision user 2026-05-15).
    """
    # Resolution des seuils par actif si non fournis explicitement
    if min_score is None:
        min_score = get_param(instrument, "min_score", 145)
    if min_quality is None:
        min_quality = get_param(instrument, "min_quality", 55)
    df_ltf = load(instrument, ltf_name)
    df_htf = load(instrument, htf_name)
    df_htf2 = None
    if htf2_name:
        try:
            df_htf2 = load(instrument, htf2_name)
        except Exception:
            df_htf2 = None

    # D1 : cache ou resample
    try:
        df_d1 = load(instrument, "D1")
        if len(df_d1) < 10:
            raise FileNotFoundError
    except (FileNotFoundError, Exception):
        df_h1 = load(instrument, "H1")
        df_d1 = build_d1_from_h1(df_h1)

    # HTF swings (pour TP base sur swing H1/H4/D1)
    htf_dfs_swings = {}
    for tf in ["H1", "H4", "D1"]:
        try:
            df_t = load(instrument, tf)
            htf_dfs_swings[tf] = df_t
        except Exception:
            pass
    htf_swings = collect_htf_swings(htf_dfs_swings, swing_strength=3)

    # Filtre fenetre temporelle
    mask = df_ltf.index >= (df_ltf.index.max() - pd.Timedelta(days=days))
    df_ltf = df_ltf[mask]

    # Correles SMT
    correlated_dfs: dict[str, tuple[pd.DataFrame, str]] = {}
    for corr_name, corr_type in SMT_PAIRS.get(instrument, []):
        for tf_try in [ltf_name, "M1"]:
            try:
                df_c = load(corr_name, tf_try)
                if len(df_c) > 0:
                    mask_c = df_c.index >= (df_c.index.max() - pd.Timedelta(days=days))
                    correlated_dfs[corr_name] = (df_c[mask_c], corr_type)
                    break
            except Exception:
                continue

    swing_strength_ltf = get_param(instrument, "swing_strength_m1", 2)
    obs = detect_order_blocks(df_ltf, swing_strength=swing_strength_ltf)

    # Bible V2 §2 : on charge H1 pour le mecanisme Feu Vert si daily neutre
    df_h1_for_feu_vert: pd.DataFrame | None = None
    if htf2_name == "H1" and df_htf2 is not None:
        df_h1_for_feu_vert = df_htf2
    elif htf_name == "H1":
        df_h1_for_feu_vert = df_htf
    else:
        try:
            df_h1_for_feu_vert = load(instrument, "H1")
        except Exception:
            df_h1_for_feu_vert = None

    # OPTIM 1 : pre-calcul des computations partagees entre tous les OB
    # Avant : 50 OB -> 50x les memes detect_fvg/find_swings/detect_breakers/detect_structure_breaks/detect_order_blocks(df_htf)
    # Apres : calcul UNE seule fois et reuse via cache
    cache: dict = {}
    cache["swings_ltf"] = find_swings(df_ltf, strength=swing_strength_ltf)
    cache["fvgs_ltf"] = detect_fvg(df_ltf)
    cache["breakers_ltf"] = detect_breakers(df_ltf)
    cache["structure_breaks"] = detect_structure_breaks(df_ltf, swings=cache["swings_ltf"])
    cache["htf_trend"] = detect_trend(cache["swings_ltf"], lookback=6)
    cache["obs_htf"] = detect_order_blocks(df_htf)
    if df_htf2 is not None:
        cache["obs_htf2"] = detect_order_blocks(df_htf2)

    results: list[PipelineResult] = []
    for ob in obs:
        r = evaluate_ob(
            ob, df_ltf, df_htf, df_d1, instrument,
            ltf_name=ltf_name, htf_name=htf_name,
            df_htf2=df_htf2, htf2_name=htf2_name,
            correlated_dfs=correlated_dfs,
            htf_swings=htf_swings,
            df_h1=df_h1_for_feu_vert,
            min_score=min_score,
            min_quality=min_quality,
            cache=cache,
        )
        results.append(r)

    return results


if __name__ == "__main__":
    instrument = "XAUUSD"
    print(f"=== Pipeline Vizion {instrument} M5/H1 ===\n")

    results = run_pipeline(instrument, ltf_name="M5", htf_name="H1", days=7)

    total = len(results)
    trades = [r for r in results if r.verdict == "TRADE"]
    rejected = [r for r in results if r.verdict == "REJECTED"]
    print(f"OB candidats : {total}")
    print(f"TRADES valides : {len(trades)}")
    print(f"REJETES : {len(rejected)}")

    # Repartition des raisons de rejet
    from collections import Counter
    reasons = Counter(r.rejection_reason for r in rejected)
    print("\n=== Raisons de rejet ===")
    for reason, count in reasons.most_common():
        print(f"  {count:3d}x : {reason}")

    print("\n=== Trades valides (jusqu'a 10) ===")
    for r in trades[:10]:
        s = r.trade_setup
        confluences = ", ".join(r.confluences)
        print(
            f"  {r.ob.validation_ts} | {r.ob.direction:8s} | "
            f"score={r.score:3d} | entry={s.entry_price:.3f} sl={s.stop_loss:.3f} tp={s.take_profit:.3f} "
            f"RR={s.rr:.2f} | {confluences}"
        )
