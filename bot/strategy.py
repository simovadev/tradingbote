"""Strategy V4 - Vizion Trading France approach.

Principes :
- TP = vraie liquidite (PDH/PDL/Asia/swing M15 majeur), pas un RR fixe
- SL = sous le low/high de l'OB + buffer
- Top-down strict : H4/H1 bias > M15 OB > M1 execution
- Selection serree : on garde au MAX 1-3 meilleurs OB du jour par actif
- FVG confluence puissante (M1 + M15)
- Killzones obligatoires, pas de trades nocturnes
"""
from __future__ import annotations

import secrets

import pandas as pd

from bot.detectors.displacement import analyze_displacement
from bot.detectors.entry import find_entry
from bot.detectors.fvg import detect_fvgs, find_fvg_in_zone
from bot.detectors.liquidity import detect_all_sweeps
from bot.detectors.mitigation import check_mitigation
from bot.detectors.multi_sweep import find_sweep_clusters, is_in_cluster
from bot.detectors.ob_confluence import confluence_for, find_confluences
from bot.detectors.order_blocks import check_breaker_block, detect_ob_from_break
from bot.detectors.ote import analyze_ote
from bot.detectors.po3 import detect_judas
from bot.detectors.premium_discount import analyze_position
from bot.detectors.sessions import compute_daily_levels, find_matched_level
from bot.detectors.structure import find_structure_break
from bot.detectors.swings import detect_swings
from bot.detectors.target_finder import find_best_target
from bot.detectors.timing import analyze_day, check_news_blackout
from bot.killzones import current_killzone
from bot.mtf_analysis import analyze_multi_tf
from bot.portfolio import FictivePortfolio, SimulatedTrade
from bot.scoring import DEFAULT_SCORE_THRESHOLD, SetupAnalysis
from config import INSTRUMENTS

# Vizion : on accepte un RR aussi bas que ce que le marche donne, mais idealement >= 1.5
VIZION_MIN_RR = 1.0
# Max nombre de trades a garder par jour (les meilleurs)
MAX_TRADES_PER_DAY = 3


def scan_all(
    dfs: dict[str, pd.DataFrame],
    portfolio: FictivePortfolio,
    instrument: str = "XAUUSD",
    correlates_m1: dict[str, pd.DataFrame] | None = None,
    score_threshold: int = DEFAULT_SCORE_THRESHOLD,
) -> list[SimulatedTrade]:
    df_m1 = dfs.get("M1")
    if df_m1 is None or df_m1.empty:
        return []

    df_m15 = dfs.get("M15")
    df_h1 = dfs.get("H1")
    df_h4 = dfs.get("H4")

    inst_cfg = INSTRUMENTS.get(instrument, {})
    tick_value = inst_cfg.get("tick_value", 100.0)
    min_sl = inst_cfg.get("min_sl_points", 3.0)
    correlates_m1 = correlates_m1 or {}

    swings_m1 = detect_swings(df_m1, left=3, right=3)
    if not swings_m1:
        return []

    sweeps = detect_all_sweeps(df_m1, swings_m1)
    sweep_clusters = find_sweep_clusters(sweeps)
    fvgs_m1 = detect_fvgs(df_m1)
    fvgs_m15 = detect_fvgs(df_m15) if df_m15 is not None and not df_m15.empty else []

    # Pre-calc OBs candidats
    candidate_obs = []
    sb_by_ob: dict[int, object] = {}
    for sweep in sweeps:
        sb = find_structure_break(df_m1, sweep, swings_m1)
        if sb is None:
            continue
        ob = detect_ob_from_break(df_m1, sb)
        if ob is None:
            continue
        if check_breaker_block(df_m1, ob):
            ob.ob_type = "breaker_block"
        candidate_obs.append(ob)
        sb_by_ob[id(ob)] = sb
    ob_confluences = find_confluences(candidate_obs)

    seen_keys: set[tuple] = set()
    all_trades: list[SimulatedTrade] = []

    for ob in candidate_obs:
        sb = sb_by_ob[id(ob)]
        sweep = sb.sweep

        key = (ob.candle_index, ob.kind)
        if key in seen_keys:
            continue
        seen_keys.add(key)

        # Niveaux journaliers
        ob_time = ob.candle_time
        daily_levels = compute_daily_levels(df_m1, ob_time)

        # Target intelligent (Vizion) : on cherche la prochaine vraie liquidite
        # MAIS avec un plafond raisonnable pour eviter SL touche en chemin.
        target = find_best_target(
            direction=ob.kind,
            entry_price=ob.zone_mid,
            entry_index=sb.break_index,
            df_m1=df_m1,
            df_m15=df_m15,
            daily_levels=daily_levels,
        )

        # Cherche l'entree : 2eme touch d'abord (Vizion principe pur)
        entry = None
        if target is not None:
            entry = find_entry(
                df_m1, ob,
                target_price=target.price,
                target_source=target.source,
                min_rr=2.0,           # Vizion : on vise RR >= 2 sur les vraies liquidites
                max_rr=10.0,
                sl_min_points=min_sl,
                require_second_touch=True,
            )
            # Fallback 1er touch si pas de 2eme
            if entry is None:
                entry = find_entry(
                    df_m1, ob,
                    target_price=target.price,
                    target_source=target.source,
                    min_rr=2.0,
                    max_rr=10.0,
                    sl_min_points=min_sl,
                    require_second_touch=False,
                )
        if entry is None:
            continue

        analysis = SetupAnalysis()

        # ============ FILTRES DURS (Vizion stricts mais pas etouffants) ============

        # 1. OB dans killzone (Vizion strict)
        ob_kz = current_killzone(ob.candle_time)
        analysis.add_hard("ob_in_killzone", "OB forme dans une killzone",
                          ob_kz is not None,
                          f"OB @ {ob.candle_time.strftime('%H:%M UTC')} - {ob_kz.label if ob_kz else 'HORS'}")

        # 2. Entree dans killzone
        entry_kz = current_killzone(entry.entry_time)
        analysis.add_hard("entry_in_killzone", "Entree dans une killzone",
                          entry_kz is not None,
                          f"Entry @ {entry.entry_time.strftime('%H:%M UTC')} - {entry_kz.label if entry_kz else 'HORS'}")

        # 3. Sweep confirme
        analysis.add_hard("sweep_confirmed", "Sweep de liquidite", True,
                          f"Pool @ {sweep.pool.price:.2f}")
        # 4. BOS
        analysis.add_hard("bos_confirmed", "Break of Structure", True,
                          f"Cassure {sb.direction} @ {sb.break_price:.2f}")

        # 5. RR >= 1.0
        analysis.add_hard("rr_min", "RR >= 1.0", entry.risk_reward >= 1.0,
                          f"RR = {entry.risk_reward:.2f}")

        # 6. Displacement valide
        disp = analyze_displacement(df_m1, sb.break_index)
        analysis.add_hard("displacement_ok", "Displacement valide (vrai OB)",
                          disp.strength != "weak", disp.detail)

        # 7. News blackout
        news = check_news_blackout(entry.entry_time)
        analysis.add_hard("news_blackout", "Hors blackout news",
                          not news.is_blackout, news.detail or "OK")

        # 8. Indices US/EU pas en Asia
        if instrument in ("NAS100", "GER40"):
            asia_only = entry_kz is not None and entry_kz.name == "asia"
            analysis.add_hard(
                "indices_not_in_asia",
                f"{instrument} hors session Asia",
                passed=not asia_only,
                detail=f"Killzone : {entry_kz.label if entry_kz else 'n/a'}",
            )

        # 9. SL realiste
        sl_dist = abs(entry.entry_price - entry.stop_loss)
        sl_pct = sl_dist / entry.entry_price * 100
        max_sl_pct = {"NAS100": 0.8, "GER40": 0.8, "XAUUSD": 1.0, "USOIL": 2.0}.get(instrument, 1.0)
        analysis.add_hard("sl_realistic", f"SL realiste (<{max_sl_pct}%)",
                          passed=sl_pct <= max_sl_pct,
                          detail=f"SL = {sl_pct:.2f}%")

        # 10. HTF pas DOUBLEMENT contraires (H4 ET H1 contre = reject)
        # Vizion : on accepte H4 contre H1 ou vice-versa, mais pas les 2 contre.
        mtf = analyze_multi_tf(dfs, sb.break_time, ob.kind)
        h4_contra = "H4" in mtf.snapshots and mtf.snapshots["H4"].bias not in (ob.kind, "range")
        h1_contra = "H1" in mtf.snapshots and mtf.snapshots["H1"].bias not in (ob.kind, "range")
        analysis.add_hard("htf_not_double_contra", "H4+H1 pas tous deux contraires",
                          not (h4_contra and h1_contra),
                          f"H4={'contre' if h4_contra else 'ok'} H1={'contre' if h1_contra else 'ok'}")

        # ============ FACTEURS (scoring qualite) ============

        # Killzone weight
        if entry_kz is not None:
            analysis.add_factor("killzone_strength", f"Killzone : {entry_kz.label}",
                                "temporal", "present", entry_kz.weight, "")

        # Target source (PDH/PDL/Asia > swing local)
        if target.source in ("PDH", "PDL"):
            analysis.add_factor("target_quality", f"Target = {target.source} (premium)",
                                "liquidity", "present", 20,
                                f"@ {target.price:.2f}, +{target.distance_pts:.1f}pts")
        elif "Asia" in target.source:
            analysis.add_factor("target_quality", f"Target = {target.source}",
                                "liquidity", "present", 15,
                                f"@ {target.price:.2f}, +{target.distance_pts:.1f}pts")
        elif "London" in target.source or "NY" in target.source:
            analysis.add_factor("target_quality", f"Target = {target.source}",
                                "liquidity", "present", 12,
                                f"@ {target.price:.2f}, +{target.distance_pts:.1f}pts")
        elif "M15" in target.source:
            analysis.add_factor("target_quality", f"Target = {target.source}",
                                "liquidity", "present", 8,
                                f"@ {target.price:.2f}, +{target.distance_pts:.1f}pts")
        else:
            analysis.add_factor("target_quality", f"Target = swing M1 local",
                                "liquidity", "neutral", 3,
                                f"@ {target.price:.2f}, +{target.distance_pts:.1f}pts")

        # Day of week
        day_an = analyze_day(entry.entry_time)
        if day_an.quality == "best":
            analysis.add_factor("day_quality", f"Jour : {day_an.label}", "temporal",
                                "present", day_an.weight, "Smart money day")
        elif day_an.quality == "avoid":
            analysis.add_factor("day_quality", f"Jour : {day_an.label}", "temporal",
                                "warning", abs(day_an.weight), "Eviter")

        # Daily level swept (qualite du sweep)
        matched = find_matched_level(sweep.pool.price, sweep.side, daily_levels, tolerance_pct=0.15)
        if matched is not None:
            level_name, level_price = matched
            weight = 18 if level_name in ("PDH", "PDL") else 14 if "Asia" in level_name else 8
            analysis.add_factor("daily_level_swept", f"Sweep d'un niveau majeur : {level_name}",
                                "liquidity", "present", weight,
                                f"{level_name} @ {level_price:.2f}")

        # Judas
        judas = detect_judas(df_m1, daily_levels, entry.entry_time)
        if judas.detected:
            trade_against_judas = (
                (judas.judas_type == "asia_high_sweep" and ob.kind == "bearish")
                or (judas.judas_type == "asia_low_sweep" and ob.kind == "bullish")
            )
            if trade_against_judas:
                analysis.add_factor("judas_swing", "Trade sens du retournement Judas",
                                    "structure", "present", 15, judas.detail)

        # OTE
        if ob.kind == "bullish":
            swing_low = sweep.wick_extreme
            swing_high = sb.break_price
        else:
            swing_high = sweep.wick_extreme
            swing_low = sb.break_price
        ote = analyze_ote(entry.entry_price, swing_high, swing_low, ob.kind)
        if ote.quality == "sweet_spot":
            analysis.add_factor("ote", "OTE sweet spot (70%)", "entry", "present", 18, ote.detail)
        elif ote.quality == "premium":
            analysis.add_factor("ote", "Zone OTE", "entry", "present", 12, ote.detail)
        elif ote.quality == "extreme":
            analysis.add_factor("ote", "OTE extreme", "entry", "warning", 5, ote.detail)

        # MTF Alignment
        align_strength = mtf.alignment_strength(ob.kind)
        align_weight = {"full": 18, "strong": 12, "moderate": 6, "weak": 0, "contradictory": 0}[align_strength]
        if align_weight > 0:
            analysis.add_factor("mtf_alignment", f"Alignement MTF : {align_strength}",
                                "htf", "present", align_weight,
                                f"{mtf.aligned_count}/{mtf.total_count} TFs")

        # H4 et H1 individuel (bonus)
        for tf_name in ["H4", "H1"]:
            snap = mtf.snapshots.get(tf_name)
            if snap and snap.bias == ob.kind:
                analysis.add_factor(f"tf_{tf_name.lower()}_aligned",
                                    f"{tf_name} bias aligne ({snap.bias})",
                                    "htf", "present", 8, snap.notes)

        # Multi-sweep
        cluster = is_in_cluster(sweep, sweep_clusters)
        if cluster:
            analysis.add_factor("multi_sweep", f"Sweep multiple {cluster.count}x", "liquidity",
                                "warning", 5, f"{cluster.count} sweeps")
        else:
            analysis.add_factor("multi_sweep", "Sweep unique propre", "liquidity",
                                "present", 5, "")

        # OB Type
        if ob.ob_type == "breaker_block":
            analysis.add_factor("ob_type", "Breaker Block", "ob_quality", "present", 12,
                                "OB casse > OB normal")
        elif ob.ob_type == "wick_ob":
            analysis.add_factor("ob_type", "OB de meche", "ob_quality", "present", 8, "")

        # Mitigation (fresh = bonus fort)
        mit = check_mitigation(df_m1, ob)
        if mit.is_fresh:
            analysis.add_factor("ob_fresh", "OB FRESH (jamais teste)", "ob_quality", "present", 12, mit.detail)
        else:
            analysis.add_factor("ob_fresh", f"OB deja teste ({mit.touches_before_break}x)",
                                "ob_quality", "warning", 8, mit.detail)

        # OB confluence
        confl = confluence_for(ob, ob_confluences)
        if confl:
            analysis.add_factor("ob_confluence", f"Confluence {confl.count} OBs",
                                "ob_quality", "present", 18, "Setup premium")

        # FVG M1 confluence
        fvg_m1 = find_fvg_in_zone(fvgs_m1, ob.zone_high, ob.zone_low, ob.kind, sb.break_index)
        if fvg_m1 is not None:
            if fvg_m1.inverted:
                analysis.add_factor("fvg_m1", "iFVG M1 en confluence", "fvg", "present", 14, "FVG inverse")
            elif not fvg_m1.filled:
                analysis.add_factor("fvg_m1", "FVG M1 actif en confluence", "fvg", "present", 12, "")
            else:
                analysis.add_factor("fvg_m1", "FVG M1 deja rempli", "fvg", "present", 5, "")

        # FVG M15 confluence (HTF, plus puissant)
        if fvgs_m15:
            entry_time = entry.entry_time
            # Filtre FVG M15 formes AVANT l'entree
            m15_before = [f for f in fvgs_m15 if f.candle_time <= entry_time]
            # Chevauchement avec la zone OB
            for fvg in m15_before:
                if fvg.kind != ob.kind:
                    continue
                # Chevauche la zone OB ?
                if fvg.zone_high < ob.zone_low or fvg.zone_low > ob.zone_high:
                    continue
                # Match
                if not fvg.filled:
                    analysis.add_factor("fvg_m15", "FVG M15 actif sur la zone (PREMIUM)",
                                        "fvg", "present", 22, "Confluence HTF")
                elif fvg.inverted:
                    analysis.add_factor("fvg_m15", "iFVG M15 sur la zone",
                                        "fvg", "present", 20, "")
                break

        # Entry touch
        if entry.touch_count >= 2:
            analysis.add_factor("entry_touch", f"Entree au {entry.touch_count}eme retour",
                                "entry", "present", 10, "")
        else:
            analysis.add_factor("entry_touch", "1er retour (Vizion : OK si setup propre)",
                                "entry", "present", 4, "")

        # Volume
        if "volume" in df_m1.columns:
            try:
                sweep_vol = float(df_m1.iloc[sweep.candle_index]["volume"])
                start = max(0, sweep.candle_index - 20)
                avg_vol = float(df_m1.iloc[start:sweep.candle_index]["volume"].mean() or 0)
                if avg_vol > 0:
                    ratio = sweep_vol / avg_vol
                    if ratio >= 1.5:
                        analysis.add_factor("volume_sweep", f"Volume x{ratio:.1f} (TOP)",
                                            "volume", "present", 15, "")
                    elif ratio >= 1.0:
                        analysis.add_factor("volume_sweep", f"Volume x{ratio:.1f}",
                                            "volume", "present", 5, "")
            except Exception:
                pass

        # Build trade
        trade = _build_trade(entry, ob, sb, sweep, analysis, portfolio, mtf,
                             instrument, tick_value, target)
        trade.simulate_outcome(df_m1)
        should_trade = analysis.all_hard_passed and analysis.score >= score_threshold
        trade.would_trade = should_trade
        all_trades.append(trade)

    # ============ DEDUP : trades avec meme entry_time = doublon ============
    seen_entry_times: set = set()
    deduped: list[SimulatedTrade] = []
    for t in all_trades:
        key = (t.entry_time, t.direction, round(t.entry_price, 4))
        if key in seen_entry_times:
            continue
        seen_entry_times.add(key)
        deduped.append(t)
    all_trades = deduped

    # ============ SELECTION VIZION : top N par jour ============
    by_day: dict = {}
    for t in all_trades:
        if not t.would_trade:
            continue
        day_key = t.entry_time.date()
        by_day.setdefault(day_key, []).append(t)

    kept_ids: set = set()
    for day, day_trades in by_day.items():
        day_trades.sort(key=lambda t: (-t.score, -t.risk_reward))
        for t in day_trades[:MAX_TRADES_PER_DAY]:
            kept_ids.add(t.id)

    final_trades = []
    for t in all_trades:
        if t.would_trade and t.id not in kept_ids:
            t.would_trade = False
        else:
            if t.would_trade and t.status in ("win", "loss"):
                portfolio.apply_result(t.pnl, t.status == "win")
        final_trades.append(t)

    return final_trades


def _build_trade(entry, ob, sb, sweep, analysis, portfolio, mtf, instrument, tick_value, target):
    lot = portfolio.position_size(entry.entry_price, entry.stop_loss, tick_value=tick_value)
    overlays = {
        "liquidity": {"price": sweep.pool.price, "side": sweep.side},
        "sweep": {"time": sweep.candle_time.isoformat(), "wick": sweep.wick_extreme},
        "structure_break": {"time": sb.break_time.isoformat(), "price": sb.broken_swing.price},
        "ob_zone": {
            "high": ob.zone_high, "low": ob.zone_low, "mid": ob.zone_mid,
            "from_time": ob.candle_time.isoformat(),
            "type": ob.ob_type,
        },
        "entry": {"time": entry.entry_time.isoformat(), "price": entry.entry_price},
        "sl": entry.stop_loss,
        "tp": entry.take_profit,
        "target_source": target.source,
    }
    direction = "bullish" if ob.kind == "bullish" else "bearish"
    return SimulatedTrade(
        id=secrets.token_hex(6),
        direction=direction,
        instrument=instrument,
        entry_time=entry.entry_time,
        entry_price=entry.entry_price,
        stop_loss=entry.stop_loss,
        take_profit=entry.take_profit,
        lot_size=lot,
        risk_reward=entry.risk_reward,
        htf_strength=mtf.alignment_strength(ob.kind),
        ob_type=ob.ob_type,
        touch_count=entry.touch_count,
        notes=f"Target: {target.source} @ {target.price:.2f}",
        chart_overlays=overlays,
        score=analysis.score,
        verdict=analysis.verdict,
        setup_analysis=analysis.to_dict(),
        tick_value=tick_value,
    )
