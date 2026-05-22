"""Construction du dataset ML : extrait features + outcome pour chaque OB candidat.

Pour CHAQUE OB detecte (acceptes ET rejetes par le pipeline Vizion), on extrait :
- Features Vizion (score, confluences, RR, displacement, killzone, etc.)
- Outcome de la simulation (WIN/LOSS/NO_FILL)

Output : dataset.parquet pour entrainer un LightGBM.

PRINCIPE : on garde TOUTES les regles Vizion eliminatoires (daily bias contraire, etc.)
mais on retire les filtres "qualite" (min_score, min_quality) pour avoir un dataset large.
Le ML apprendra a filtrer la qualite.
"""
from __future__ import annotations

import sys
# Auto-detect Windows vs Linux pour le sys.path
_ROOT = "c:/Users/Shadow/TradingBot" if sys.platform == "win32" else "/workspace/TradingBot"
sys.path.insert(0, _ROOT)

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from bot_v2.backtest import simulate_trade
from bot_v2.config import SMT_PAIRS, get_param, primary_instruments
from bot_v2.concepts.breaker import detect_breakers
from bot_v2.concepts.daily_bias import build_d1_from_h1
from bot_v2.concepts.fvg import detect_fvg
from bot_v2.concepts.htf_swings import collect_htf_swings
from bot_v2.concepts.killzones import killzone_at
from bot_v2.concepts.liquidity import find_swings
from bot_v2.concepts.order_block import detect_order_blocks
from bot_v2.concepts.structure import detect_structure_breaks, detect_trend
from bot_v2.data_loader import load
from bot_v2.pipeline import evaluate_ob
from bot_v2.trade_setup import compute_position_size, TradeSetup


def _compute_atr(df, idx, period=14):
    """ATR True Range moyen sur `period` bougies precedant idx."""
    if idx < period:
        return 0.0
    sl = df.iloc[max(0, idx - period):idx]
    high_low = sl["high"] - sl["low"]
    high_close = (sl["high"] - sl["close"].shift(1)).abs()
    low_close = (sl["low"] - sl["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return float(tr.mean()) if len(tr) > 0 else 0.0


def _get_daily_levels(df_d1, validation_ts):
    """Retourne (pdh, pdl, daily_open, prev_day_open) au moment de validation_ts.

    FIX V8 (2026-05-21) : data leakage corrige. Avant : df_d1.index < validation_ts
    incluait la D1 du jour en cours (complete dans le parquet de training, partielle
    en live) -> training apprenait sur du futur, live ne pouvait pas reproduire.
    Maintenant : on filtre sur validation_ts.normalize() pour exclure le D1 du jour.
    """
    if df_d1 is None or len(df_d1) == 0:
        return None, None, None, None
    today_start = validation_ts.normalize()
    past = df_d1[df_d1.index < today_start]
    if len(past) < 2:
        return None, None, None, None
    yesterday = past.iloc[-1]
    # today_open = close de hier (= open d'aujourd'hui ~ prev close, sans data leakage).
    today_open = yesterday["close"]
    prev_day_open = past.iloc[-2]["open"]
    return float(yesterday["high"]), float(yesterday["low"]), float(today_open), float(prev_day_open)


def _extract_features(r, ob, instrument, df_ltf=None, df_d1=None, df_htf=None, mss_setups=None):
    """Extrait features ML pour le dataset de training.

    REFACTO V10 (2026-05-22) : delegue a ml_filter._features_from_result pour
    garantir que training et live calculent EXACTEMENT les memes features.
    Avant : deux copies separees du code -> risque de divergence (cause de bugs).
    On ajoute juste les 3 champs meta (instrument, ts, direction) par-dessus.
    """
    from bot_v2 import ml_filter
    f = {
        "instrument": instrument,
        "ts": ob.validation_ts,
        "direction": ob.direction,
    }
    f.update(ml_filter._features_from_result(
        r, ob, instrument, df_ltf=df_ltf, df_d1=df_d1, mss_setups=mss_setups,
    ))
    return f


# Chaines de TF Vizion par LTF (user 2026-05-16 : multi-TF)
# Note : M30 pas dispo dans data_loader, on utilise H1 a la place pour M5.
TF_CHAINS = {
    "M1":  {"ltf": "M1",  "htf": "M15", "htf2": "H1"},
    "M5":  {"ltf": "M5",  "htf": "H1",  "htf2": "H4"},
    "M15": {"ltf": "M15", "htf": "H1",  "htf2": "H4"},
}


# ============ CACHE WORKER-LEVEL (refacto 2026-05-19) ============
# Chaque worker process garde en cache les data globales (M1/HTF/SMT/swings/OBs HTF)
# pour eviter de les recharger/recalculer a chaque chunk.
# Gain estime : x3-x5 sur la vitesse de build (selon nb chunks).
_WORKER_CACHE = {}


def _get_global_data(inst, ltf_name, htf_name, htf2_name):
    """Charge et cache les data globales (toutes periodes) + indicateurs HTF.

    Appele 1 fois par worker process, reutilise pour tous les chunks du worker.
    """
    cache_key = (inst, ltf_name, htf_name, htf2_name)
    if cache_key in _WORKER_CACHE:
        return _WORKER_CACHE[cache_key]

    # === Load data globales ===
    df_ltf = load(inst, ltf_name)
    df_htf = load(inst, htf_name)
    try:
        df_htf2 = load(inst, htf2_name)
    except Exception:
        df_htf2 = None
    try:
        df_d1 = load(inst, "D1")
        if len(df_d1) < 10:
            raise FileNotFoundError
    except Exception:
        df_h1 = load(inst, "H1")
        df_d1 = build_d1_from_h1(df_h1)

    # === Precompute HTF indicateurs (1 fois pour toutes !) ===
    htf_dfs_swings = {}
    for tf in ["H1", "H4", "D1"]:
        try:
            htf_dfs_swings[tf] = load(inst, tf)
        except Exception:
            pass
    htf_swings = collect_htf_swings(htf_dfs_swings, swing_strength=3)

    # === Precompute OBs HTF (1 fois pour toutes !) ===
    obs_htf = detect_order_blocks(df_htf)
    obs_htf2 = detect_order_blocks(df_htf2) if df_htf2 is not None else None

    # === Load SMT correlated data (XAGUSD, DXY pour XAUUSD) ===
    correlated_dfs_full = {}
    for corr_name, corr_type in SMT_PAIRS.get(inst, []):
        try:
            df_c = load(corr_name, ltf_name)
            if len(df_c) > 0:
                correlated_dfs_full[corr_name] = (df_c, corr_type)
        except Exception:
            continue

    data = {
        "df_ltf": df_ltf,
        "df_htf": df_htf,
        "df_htf2": df_htf2,
        "df_d1": df_d1,
        "htf_swings": htf_swings,
        "obs_htf": obs_htf,
        "obs_htf2": obs_htf2,
        "correlated_dfs_full": correlated_dfs_full,
    }
    _WORKER_CACHE[cache_key] = data
    return data


def _process_instrument(args):
    """Worker process : extrait dataset complet pour un instrument.

    Args tuple : (inst, start_ts, end_ts) ou (inst, start_ts, end_ts, ltf).
    LTF par defaut = M1. Si M5/M15/M30 fournis, on adapte les chaines HTF.

    REFACTO 2026-05-19 : utilise _WORKER_CACHE pour charger data 1 fois par worker.
    """
    if len(args) == 4:
        inst, start_ts, end_ts, ltf = args
    else:
        inst, start_ts, end_ts = args
        ltf = "M1"

    chain = TF_CHAINS.get(ltf, TF_CHAINS["M1"])
    ltf_name = chain["ltf"]
    htf_name = chain["htf"]
    htf2_name = chain["htf2"]

    try:
        # OPTIM V5 (2026-05-20) : on charge UNIQUEMENT la fenetre du chunk + buffer.
        # Buffer = 30j avant pour avoir les swings/OB HTF anterieurs au chunk.
        # Avant : chaque worker chargeait 2.6M bougies M1, ~25s/chunk d'I/O.
        # Maintenant : ~130k bougies M1 (3 mois + buffer), ~3s/chunk.
        buffer_before = pd.Timedelta(days=30)
        load_start = start_ts - buffer_before
        load_end = end_ts

        df_ltf = load(inst, ltf_name, start=load_start, end=load_end)
        df_htf = load(inst, htf_name, start=load_start, end=load_end)
        try:
            df_htf2 = load(inst, htf2_name, start=load_start, end=load_end)
        except Exception:
            df_htf2 = None
        try:
            df_d1 = load(inst, "D1", start=load_start, end=load_end)
            if len(df_d1) < 10:
                raise FileNotFoundError
        except Exception:
            df_h1 = load(inst, "H1", start=load_start, end=load_end)
            df_d1 = build_d1_from_h1(df_h1)

        htf_dfs_swings = {}
        for tf in ["H1", "H4", "D1"]:
            try:
                htf_dfs_swings[tf] = load(inst, tf, start=load_start, end=load_end)
            except Exception:
                pass
        htf_swings = collect_htf_swings(htf_dfs_swings, swing_strength=3)

        mask = (df_ltf.index >= start_ts) & (df_ltf.index <= end_ts)
        df_ltf_w = df_ltf[mask]
        if len(df_ltf_w) < 100:
            return [], inst, "Pas assez de donnees"

        correlated_dfs = {}
        for corr_name, corr_type in SMT_PAIRS.get(inst, []):
            try:
                df_c = load(corr_name, ltf_name, start=load_start, end=load_end)
                if len(df_c) > 0:
                    mask_c = (df_c.index >= start_ts) & (df_c.index <= end_ts)
                    correlated_dfs[corr_name] = (df_c[mask_c], corr_type)
            except Exception:
                continue

        swing_strength_ltf = get_param(inst, "swing_strength_m1", 2)
        obs = detect_order_blocks(df_ltf_w, swing_strength=swing_strength_ltf)

        # Cache pour acceleration (chunk-local : swings/fvg/breakers sur la fenetre)
        cache = {
            "swings_ltf": find_swings(df_ltf_w, strength=swing_strength_ltf),
            "fvgs_ltf": detect_fvg(df_ltf_w),
            "breakers_ltf": detect_breakers(df_ltf_w),
            "obs_htf": detect_order_blocks(df_htf),
        }
        # V5.5 (2026-05-21) : passe fvgs_ltf pre-calcule -> evite 2 detect_fvg
        # redondants. Resultats identiques (verifie), train reste aligne live.
        cache["structure_breaks"] = detect_structure_breaks(
            df_ltf_w, swings=cache["swings_ltf"], fvgs=cache["fvgs_ltf"]
        )
        cache["htf_trend"] = detect_trend(cache["swings_ltf"], lookback=6)
        if df_htf2 is not None:
            cache["obs_htf2"] = detect_order_blocks(df_htf2)

        df_h1_for_feu_vert = df_htf2 if df_htf2 is not None else None

        # V5 (user 2026-05-20) : virer confirm_ob_with_mss (filtre trop strict).
        # Le ML decide via feature has_mss_nearby.
        from bot_v2.concepts.mss_setup import detect_mss_setups
        mss_setups = detect_mss_setups(
            df_ltf_w,
            structure_breaks=cache["structure_breaks"],
            swings=cache["swings_ltf"],
            fvgs=cache["fvgs_ltf"],
        )
        cache["mss_setups"] = mss_setups
        print(f"    OB total: {len(obs)}, MSS: {len(mss_setups)}", flush=True)
        prefiltered_obs = obs
        prefiltered_mss = []

        rows = []
        for ob in prefiltered_obs:
            r = evaluate_ob(
                ob, df_ltf_w, df_htf, df_d1, inst,
                ltf_name=ltf_name, htf_name=htf_name,
                df_htf2=df_htf2, htf2_name=htf2_name,
                correlated_dfs=correlated_dfs,
                htf_swings=htf_swings,
                df_h1=df_h1_for_feu_vert,
                min_score=0, min_quality=0,
                cache=cache,
            )
            # On garde uniquement les OB qui passent les regles Vizion ELIMINATOIRES
            # (bias daily contraire, hors KZ, phase A/M, displacement faible, etc.)
            # Ces "vrais candidats" sont ce que le ML va apprendre a filtrer
            if r.verdict != "TRADE":
                continue
            if r.trade_setup is None:
                continue

            f = _extract_features(r, ob, inst, df_ltf=df_ltf_w, df_d1=df_d1, df_htf=df_htf, mss_setups=mss_setups)

            # Simulate trade pour avoir l'outcome
            try:
                setup = r.trade_setup
                lots, risk_usd = compute_position_size(
                    setup.entry_price, setup.stop_loss, inst,
                    balance=60.0, risk_pct=0.10,
                )
                if lots <= 0:
                    continue
                sim_setup = TradeSetup(
                    instrument=inst, direction=setup.direction,
                    entry_price=setup.entry_price, stop_loss=setup.stop_loss,
                    take_profit=setup.take_profit, rr=setup.rr,
                    risk_points=setup.risk_points, reward_points=setup.reward_points,
                    risk_usd=risk_usd, reward_usd=risk_usd * setup.rr,
                    position_size_lots=lots,
                    ob_validation_ts=setup.ob_validation_ts,
                    tp_source=setup.tp_source,
                )
                tr = simulate_trade(sim_setup, df_ltf_w, ob.validation_index + 1)
                f["outcome"] = tr.outcome  # "WIN" | "LOSS" | "NO_FILL" | "PENDING"
                f["pnl_usd"] = tr.pnl_usd
                f["bars_to_exit"] = (tr.exit_index - tr.fill_index) if (tr.exit_index and tr.fill_index) else None
            except Exception as e:
                f["outcome"] = "ERROR"
                f["pnl_usd"] = 0.0
                f["bars_to_exit"] = None

            rows.append(f)

        # =============== BOUCLE MSS SETUPS ===============
        # Les MSS sont evalues SANS filtre daily_bias (user 2026-05-16).
        # Raison : un MSS = changement de structure CONTRE la tendance par definition.
        # Filtrer par daily_bias annulerait l'essence du setup. Le ML decidera.
        # On garde le daily_bias comme FEATURE pour que le ML l'apprenne.
        from bot_v2.concepts.daily_bias import compute_daily_bias
        from bot_v2.concepts.killzones import killzone_at
        for mss in prefiltered_mss:
            if mss.retest_index is None:
                continue

            # On calcule le bias pour la feature, mais on ne filtre PAS
            target_date = mss.retest_ts.normalize()
            bias = compute_daily_bias(df_d1, target_date)
            bias_label = bias.bias if bias is not None else "neutral"

            # Setup : entry/SL deja calcules dans MSSSetup, TP = RR=2 fixed
            risk = abs(mss.entry_price - mss.stop_loss)
            if risk == 0:
                continue
            if mss.direction == "bullish":
                tp = mss.entry_price + risk * 2.0
            else:
                tp = mss.entry_price - risk * 2.0

            # Features MSS - on remplit autant de features que possible
            kz = killzone_at(mss.retest_ts)
            mss_feat = {
                "instrument": inst,
                "ts": mss.retest_ts,
                "direction": mss.direction,
                # Scores generiques (a defaut, valeurs neutres)
                "score": 50,
                "quality": 50,
                "ob_strength": 5,
                "sweep_strength": 0,
                "retest_count": 1,
                "is_unicorn": 0,
                "multi_liq_sweep": 0,
                # Trade
                "rr": 2.0,
                "risk_points": float(risk),
                "tp_source_htf": 0,
                "tp_source_capped": 1,
                # Daily bias (feature, plus filtre)
                "daily_bias_aligned": int(bias_label == mss.direction),
                "daily_bias_neutral": int(bias_label == "neutral"),
                # KZ
                "kz_london": int(kz == "London"),
                "kz_ny_am": int(kz == "NY_AM"),
                "kz_ny_pm": int(kz == "NY_PM"),
                "kz_asia": int(kz == "Asia"),
                "kz_ny_lunch": int(kz == "NY_Lunch"),
                "kz_none": int(kz is None),
                # Confluences
                "has_sync_fvg": 1,  # MSS+FVG par definition
                "has_smt": 0,
                "has_feu_vert": 0,
                "has_breaker_kz": 0,
                "has_mss_fvg": 1,  # MSS_with_FVG explicite
                "has_grandparent": 0,
                "has_po3_dist": 0,
                "has_phase_expansion": 0,
                "has_phase_reversal": 0,
                "has_open_midnight_respect": 0,
                # Structure
                "ob_group_size": 2,  # convention pour MSS
                "bars_sweep_to_validation": mss.retest_index - mss.mss.swing.index,
                "bars_group_to_validation": mss.retest_index - mss.mss.break_index,
                "is_bullish": int(mss.direction == "bullish"),
                "is_mss_setup": 1,  # flag MSS
            }

            # Simulate trade pour outcome
            try:
                lots, risk_usd = compute_position_size(
                    mss.entry_price, mss.stop_loss, inst, balance=60.0, risk_pct=0.10,
                )
                if lots <= 0:
                    continue
                sim_setup = TradeSetup(
                    instrument=inst, direction=mss.direction,
                    entry_price=mss.entry_price, stop_loss=mss.stop_loss,
                    take_profit=tp, rr=2.0,
                    risk_points=float(risk), reward_points=float(abs(tp - mss.entry_price)),
                    risk_usd=risk_usd, reward_usd=risk_usd * 2.0,
                    position_size_lots=lots,
                    ob_validation_ts=mss.retest_ts,
                    tp_source="mss_rr2",
                )
                tr = simulate_trade(sim_setup, df_ltf_w, mss.retest_index + 1)
                mss_feat["outcome"] = tr.outcome
                mss_feat["pnl_usd"] = tr.pnl_usd
                mss_feat["bars_to_exit"] = (tr.exit_index - tr.fill_index) if (tr.exit_index and tr.fill_index) else None
            except Exception:
                mss_feat["outcome"] = "ERROR"
                mss_feat["pnl_usd"] = 0.0
                mss_feat["bars_to_exit"] = None

            rows.append(mss_feat)

        return rows, inst, None
    except Exception as e:
        import traceback
        return [], inst, f"{e}\n{traceback.format_exc()}"


def _chunk_worker(args):
    """Worker au niveau module (picklable par ProcessPoolExecutor sous Windows).

    Args : (inst, cs, ce, ppath, ltf) - ltf par defaut M1.
    """
    if len(args) == 5:
        inst, cs, ce, ppath, ltf = args
    else:
        inst, cs, ce, ppath = args
        ltf = "M1"
    rows, _, error = _process_instrument((inst, cs, ce, ltf))
    if error:
        return inst, cs.date(), None, error
    if rows:
        import pandas as _pd
        _pd.DataFrame(rows).to_parquet(ppath)
    return inst, cs.date(), len(rows), None


def build_dataset(start_ts, end_ts, instruments=None, output_path=None, chunk_months=6, ltf="M1", version_suffix=""):
    """Construit le dataset ML pour la fenetre [start_ts, end_ts].

    Decoupage en CHUNKS de chunk_months mois pour eviter OOM
    (5 ans M1 = 1.7M bougies, le pipeline n'arrive pas a le traiter d'un coup).

    Args:
        version_suffix: ajoute au nom du dossier de chunks (ex: "_V5" -> ml_partial_M1_V5/).
                        Evite de melanger les chunks V4/V5 (features differentes).
    """
    if instruments is None:
        instruments = primary_instruments()

    # Decoupe en chunks de chunk_months mois
    chunks = []
    cur = start_ts
    while cur < end_ts:
        nxt = min(cur + pd.Timedelta(days=30 * chunk_months), end_ts)
        chunks.append((cur, nxt))
        cur = nxt

    print(f"Construction dataset : {start_ts.date()} -> {end_ts.date()}", flush=True)
    print(f"Actifs : {instruments}", flush=True)
    print(f"Decoupage : {len(chunks)} chunks de ~{chunk_months} mois", flush=True)
    print(f"Total taches : {len(instruments)} actifs x {len(chunks)} chunks = {len(instruments)*len(chunks)}", flush=True)

    all_rows = []
    total_tasks = len(instruments) * len(chunks)
    # Dossier separe par TF pour eviter melanges
    # FIX 2026-05-19 : auto-detect Windows vs Linux path
    import sys as _sys
    _root = "c:/Users/Shadow/TradingBot" if _sys.platform == "win32" else "/workspace/TradingBot"
    partial_dir = Path(f"{_root}/data/ml_partial_{ltf}{version_suffix}")
    partial_dir.mkdir(parents=True, exist_ok=True)

    # Construit la liste des taches a faire (skip celles deja sauvegardees)
    tasks_to_do = []
    skipped = 0
    for inst in instruments:
        for chunk_start, chunk_end in chunks:
            chunk_id = f"{inst}_{ltf}_{chunk_start.strftime('%Y%m%d')}"
            partial_path = partial_dir / f"{chunk_id}.parquet"
            if partial_path.exists():
                df_existing = pd.read_parquet(partial_path)
                all_rows.extend(df_existing.to_dict("records"))
                skipped += 1
            else:
                tasks_to_do.append((inst, chunk_start, chunk_end, partial_path, ltf))

    print(f"Chunks deja sauvegardes (skip) : {skipped}/{total_tasks}", flush=True)
    print(f"Chunks a traiter : {len(tasks_to_do)}", flush=True)

    if not tasks_to_do:
        print("Tout est deja fait.", flush=True)
    else:
        # Workers auto-detect.
        # V5 (2026-05-20) : cap monte a 64 (RAM serveur 251GB suffit pour 64 workers).
        # Avant : 32 max (limite arbitraire deadlock pandas+multiprocessing).
        # Override via env var N_WORKERS si besoin.
        import os
        env_workers = os.environ.get("N_WORKERS")
        if env_workers:
            n_workers = min(int(env_workers), len(tasks_to_do))
        else:
            cpu_count = os.cpu_count() or 4
            if cpu_count <= 8:
                n_workers = min(6, len(tasks_to_do))
            else:
                # Serveur : 64 max (gain 2x sur EPYC 96+ cores)
                n_workers = min(64, cpu_count, len(tasks_to_do))
        print(f"CPU cores detected : {os.cpu_count()}, workers utilises : {n_workers}", flush=True)

        done_count = 0
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_chunk_worker, t): t for t in tasks_to_do}
            for fut in as_completed(futures):
                inst, date, n_rows, error = fut.result()
                done_count += 1
                if error:
                    print(f"  [{done_count}/{len(tasks_to_do)}] {inst} {date}: ERREUR {error[:150]}", flush=True)
                    continue
                print(f"  [{done_count}/{len(tasks_to_do)}] {inst} {date}: {n_rows} candidats", flush=True)
                # Charge le partial sauve par le worker dans all_rows.
                # Cohenrent avec write (ligne 403) : nom = {inst}_{ltf}_{date}.
                chunk_id = f"{inst}_{ltf}_{pd.Timestamp(date).strftime('%Y%m%d')}"
                ppath = partial_dir / f"{chunk_id}.parquet"
                if ppath.exists():
                    df_partial = pd.read_parquet(ppath)
                    all_rows.extend(df_partial.to_dict("records"))

    df = pd.DataFrame(all_rows)
    print(f"\nDataset total : {len(df)} lignes")
    if len(df) == 0:
        print("AUCUN candidat trouve, rien a sauvegarder")
        return df

    # Stats outcomes
    print("\nDistribution outcomes :")
    print(df["outcome"].value_counts().to_string())

    closed = df[df["outcome"].isin(["WIN", "LOSS"])]
    if len(closed) > 0:
        wr = (closed["outcome"] == "WIN").mean() * 100
        print(f"\nWR brut (sans ML) : {wr:.1f}% sur {len(closed)} trades fermes")

    # Save
    if output_path is None:
        # Nom inclut TF si different de M1 (M1 garde compat avec main historique)
        if ltf == "M1":
            output_path = Path(f"{_root}/data/ml_dataset.parquet")
        else:
            output_path = Path(f"{_root}/data/ml_dataset_{ltf}.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path)
    print(f"\nDataset sauve : {output_path}")
    return df


def main():
    # PHASE 1 v2 : XAUUSD ONLY, 4 ans (2022-05 -> 2026-05).
    # v1 sur 2 ans (583 trades train) avait overfit : train WR 91% vs test 51%.
    # 4 ans = ~1750 trades train -> meilleur generalisation attendue.
    # Phase 2 (user 2026-05-16) : NAS100, meme methodologie que XAUUSD v7
    instruments = ["NAS100"]
    df = load("NAS100", "M1")
    earliest_end = df.index[-1]
    latest_start = earliest_end - pd.Timedelta(days=365 * 2)  # 2 ans
    print(f"Plage utilisee : {latest_start.date()} -> {earliest_end.date()} (2 ans)")
    print(f"Actif : NAS100 (Phase 2)")
    build_dataset(latest_start, earliest_end, instruments, chunk_months=3)


if __name__ == "__main__":
    main()
