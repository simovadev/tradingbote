"""Filtre un cache complet (OB/FVG/breakers/swings/structure_breaks) pour un
cut_iloc donne, en reproduisant le resultat qu'aurait donne le calcul direct
sur df[:cut_iloc].

Utilise par backtest_v12_tick_vast.py : on calcule le cache UNE SEULE FOIS sur
le DataFrame complet (88k bougies + bougies futures jusqu'a end), puis pour
chaque cycle on filtre par cut_iloc -> 288 cycles partagent le meme cache au
lieu de recalculer 288 fois.

Equivalence STRICTE requise : filter(cache_full, cut) == compute(df[:cut]).
Le test test_cache_filter_equivalence.py le verifie sur 1000+ cycles.

Regles de filtrage (basees sur la structure des dataclasses) :
- OrderBlock : .validation_index < cut_iloc
- Swing      : .index < cut_iloc (le swing est confirme par "strength" bougies
               apres son centre, mais ceci est ENCAPSULE dans is_live qu'on ne
               touche pas — voir notes en bas)
- FVG        : .center_index + 1 < cut_iloc (FVG valide a la 3e bougie center+1)
               + reset rebalanced/inversed si l'event > cut_iloc
- Breaker    : .inverse_index < cut_iloc
- StructureBreak : .break_index < cut_iloc

NOTE swings is_live : un swing peut etre "tue" (is_live=False) si plus tard
un high plus haut/low plus bas survient. Dans le cache full, certains swings
recents sont marques is_live=False parce qu'ils ont ete tues. A cut_iloc plus
ancien, ces swings etaient encore is_live. Donc on doit potentiellement
RECALCULER is_live a partir de cut_iloc. Ce n'est PAS fait pour l'instant —
si le test d'equivalence echoue, c'est probablement la cause.
"""
from __future__ import annotations
from dataclasses import replace
from typing import Any


def filter_obs(obs: list, cut_iloc: int) -> list:
    """Garde les OB valides avant cut_iloc."""
    return [ob for ob in obs if ob.validation_index < cut_iloc]


def filter_swings(swings: list, cut_iloc: int) -> list:
    """Garde les swings confirmes avant cut_iloc.

    Un swing de strength s est confirme APRES la s-eme bougie suivant son
    centre (find_swings utilise sliding_window 2s+1 -> centre confirme a
    indice n-s-1 dans un df de longueur n). Donc swing.index est confirme
    quand cut_iloc > swing.index + s.

    NOTE : is_live n'est PAS recalcule. Si un swing avait is_live=True a
    cut_iloc mais a ete tue avant df.index[-1], le cache complet le marque
    is_live=False. Ceci PEUT casser l'equivalence stricte — a verifier
    empiriquement par le test.
    """
    # cut_iloc exclusif. Swing confirme quand cut_iloc >= index + strength + 1
    return [s for s in swings if s.index + s.strength < cut_iloc]


def filter_fvgs(fvgs: list, cut_iloc: int) -> list:
    """Garde les FVG dont center_index + 1 < cut_iloc (FVG valide quand la 3e
    bougie a ferme = a center+1+1 = center+2 fermes, donc valide a partir de
    center+2... mais .center_index est l'index du centre, donc FVG existe
    quand bougie center+1 a ferme = a iloc center+2. On filtre par center+1
    < cut pour etre coherent avec detect_fvg qui boucle for i in 1..n-2).

    Pour les etats mutables (rebalanced, inversed), on reset si l'event a eu
    lieu a un index >= cut_iloc.
    """
    out = []
    for f in fvgs:
        if f.center_index + 1 >= cut_iloc:
            continue
        # Reset rebalanced/inversed si l'event est >= cut_iloc
        rebalanced = f.rebalanced
        rebalance_index = f.rebalance_index
        inversed = f.inversed
        inverse_index = f.inverse_index
        if rebalance_index is not None and rebalance_index >= cut_iloc:
            rebalanced = False
            rebalance_index = None
            # Si pas rebalance, pas inverse non plus
            inversed = False
            inverse_index = None
        elif inverse_index is not None and inverse_index >= cut_iloc:
            inversed = False
            inverse_index = None
        out.append(replace(
            f,
            rebalanced=rebalanced,
            rebalance_index=rebalance_index,
            inversed=inversed,
            inverse_index=inverse_index,
        ))
    return out


def filter_breakers(breakers: list, cut_iloc: int) -> list:
    """Garde les breakers dont inverse_index < cut_iloc.

    Le breaker depend aussi de origin_ob (.validation_index) qui doit aussi
    etre < cut_iloc mais c'est garanti par la transitivite : si inverse < cut,
    alors validation_index < inverse < cut.
    """
    return [b for b in breakers if b.inverse_index < cut_iloc]


def filter_structure_breaks(sb: list, cut_iloc: int) -> list:
    """Garde les structure_breaks dont break_index < cut_iloc."""
    return [s for s in sb if s.break_index < cut_iloc]


def shift_obs(obs: list, offset: int) -> list:
    """Decale tous les indices iloc d'un OB de -offset. Drop si < 0."""
    out = []
    for ob in obs:
        # group_start, group_end, validation, sweep.swing.index, sweep.sweep_index
        if ob.group_start_index - offset < 0:
            continue
        if ob.sweep.swing.index - offset < 0:
            continue
        new_sweep = replace(
            ob.sweep,
            swing=replace(ob.sweep.swing, index=ob.sweep.swing.index - offset),
            sweep_index=ob.sweep.sweep_index - offset,
        )
        out.append(replace(
            ob,
            group_start_index=ob.group_start_index - offset,
            group_end_index=ob.group_end_index - offset,
            validation_index=ob.validation_index - offset,
            sweep=new_sweep,
        ))
    return out


def shift_swings(swings: list, offset: int) -> list:
    """Decale les indices des swings."""
    return [replace(s, index=s.index - offset) for s in swings if s.index - offset >= 0]


def shift_fvgs(fvgs: list, offset: int) -> list:
    """Decale center_index et les *_index optionnels."""
    out = []
    for f in fvgs:
        new_ci = f.center_index - offset
        if new_ci < 0:
            continue
        new_rebal = (f.rebalance_index - offset) if f.rebalance_index is not None else None
        new_inv = (f.inverse_index - offset) if f.inverse_index is not None else None
        out.append(replace(
            f, center_index=new_ci,
            rebalance_index=new_rebal,
            inverse_index=new_inv,
        ))
    return out


def shift_breakers(breakers: list, offset: int) -> list:
    """Decale inverse_index + origin_ob (recursivement)."""
    out = []
    for b in breakers:
        # On a besoin que origin_ob soit aussi shifte
        ob = b.origin_ob
        if ob.group_start_index - offset < 0:
            continue
        new_sweep = replace(
            ob.sweep,
            swing=replace(ob.sweep.swing, index=ob.sweep.swing.index - offset),
            sweep_index=ob.sweep.sweep_index - offset,
        )
        new_ob = replace(
            ob,
            group_start_index=ob.group_start_index - offset,
            group_end_index=ob.group_end_index - offset,
            validation_index=ob.validation_index - offset,
            sweep=new_sweep,
        )
        out.append(replace(
            b,
            origin_ob=new_ob,
            inverse_index=b.inverse_index - offset,
        ))
    return out


def shift_structure_breaks(sb: list, offset: int) -> list:
    """Decale break_index + le swing reference."""
    out = []
    for s in sb:
        if s.break_index - offset < 0:
            continue
        if s.swing.index - offset < 0:
            continue
        new_swing = replace(s.swing, index=s.swing.index - offset)
        out.append(replace(s, swing=new_swing, break_index=s.break_index - offset))
    return out


def filter_cache_for_cut(cache: dict, cut_iloc_m1: int,
                        cut_iloc_m15: int | None = None,
                        cut_iloc_h1: int | None = None,
                        df_m1_cut=None) -> dict:
    """Filtre un cache complet pour reproduire le resultat a cut_iloc.

    Args:
        cache : dict avec keys obs/swings_ltf/fvgs_ltf/breakers_ltf/obs_htf/
                structure_breaks/htf_trend/obs_htf2
        cut_iloc_m1 : iloc dans df_m1 (exclusive)
        cut_iloc_m15 : iloc dans df_m15 (si None, obs_htf non filtres)
        cut_iloc_h1 : iloc dans df_h1 (si None, obs_htf2 non filtres)
        df_m1_cut : df_m1 jusqu'a cut (necessaire pour recalculer
                    structure_breaks dont has_displacement_fvg depend des
                    FVG filtres, pas du cache complet)

    Returns: nouveau dict avec listes filtrees.
    """
    from bot_v2.concepts.structure import detect_trend, detect_structure_breaks
    out = {
        "obs": filter_obs(cache["obs"], cut_iloc_m1),
        "swings_ltf": filter_swings(cache["swings_ltf"], cut_iloc_m1),
        "fvgs_ltf": filter_fvgs(cache["fvgs_ltf"], cut_iloc_m1),
        "breakers_ltf": filter_breakers(cache["breakers_ltf"], cut_iloc_m1),
    }
    # structure_breaks : recalcul a partir des swings/FVG filtres pour que
    # has_displacement_fvg soit coherent (sinon il vient du cache full qui
    # voit les FVG futurs). Le recalcul est rapide car la fonction est
    # O(N_swings * log) avec swings deja calcules.
    if df_m1_cut is not None:
        out["structure_breaks"] = detect_structure_breaks(
            df_m1_cut, swings=out["swings_ltf"], fvgs=out["fvgs_ltf"]
        )
    else:
        # Fallback : filtre seulement (peut etre subtilement faux sur les MSS)
        out["structure_breaks"] = filter_structure_breaks(cache["structure_breaks"], cut_iloc_m1)
    # htf_trend depend des derniers swings filtres
    out["htf_trend"] = detect_trend(out["swings_ltf"], lookback=6)
    # obs_htf / obs_htf2 : si on a fourni cut_iloc_m15/h1, on filtre. Sinon, on
    # garde tel quel (suffisant pour la plupart des cas car ils changent peu).
    if cut_iloc_m15 is not None:
        out["obs_htf"] = filter_obs(cache["obs_htf"], cut_iloc_m15)
    else:
        out["obs_htf"] = cache["obs_htf"]
    if cut_iloc_h1 is not None:
        out["obs_htf2"] = filter_obs(cache["obs_htf2"], cut_iloc_h1)
    else:
        out["obs_htf2"] = cache["obs_htf2"]
    return out
