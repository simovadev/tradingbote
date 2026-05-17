"""Retrain : analyse les validations user et ajuste les parametres du detecteur.

Workflow :
1. Recupere les 30 dernieres validations de la generation courante
2. Calcule stats YES vs NO sur les features mesurables
3. Ajuste min_push_atr, killzones preferes, heures, etc.
4. Stocke en BotTuning + incremente generation
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from sqlalchemy import select

from db.models import BotTuning, SessionLocal, ValidatedSetup


# Fichier qui stocke les parametres actifs du detecteur (lu par strategy/algo)
TUNING_FILE = Path(__file__).parent.parent / "data" / "current_tuning.json"
TUNING_FILE.parent.mkdir(exist_ok=True)


# Parametres par defaut
DEFAULT_PARAMS = {
    "min_push_atr_mult": 2.0,
    "min_push_points_pct": 0.0010,
    "min_push_candles": 2,
    "max_push_candles": 8,
    "max_return_lookforward": 30,
    "dedup_window_candles": 20,
    # Filtres temporels (preferences user)
    "allowed_killzones": ["asia", "london", "ny", "london_close"],
    "allowed_hours": list(range(24)),   # toutes par defaut
    "killzone_required": True,    # OBLIGATOIRE : pas de trade hors killzone (user demande)
    # Filtres direction
    "long_only_hours": None,
    "short_only_hours": None,
}


def load_current_params() -> dict:
    """Charge les parametres actifs (ou les defauts si pas encore tune)."""
    if not TUNING_FILE.exists():
        return dict(DEFAULT_PARAMS)
    try:
        return json.loads(TUNING_FILE.read_text())
    except Exception:
        return dict(DEFAULT_PARAMS)


def save_params(params: dict) -> None:
    TUNING_FILE.write_text(json.dumps(params, indent=2))


def current_generation() -> int:
    """Generation courante = max(gen des validated_setups) ou 1."""
    with SessionLocal() as s:
        gens = s.execute(select(ValidatedSetup.generation)).scalars().all()
        return max(gens) if gens else 1


def n_validations_in_generation(gen: int) -> int:
    with SessionLocal() as s:
        rows = s.execute(
            select(ValidatedSetup).where(ValidatedSetup.generation == gen)
        ).scalars().all()
        return len(rows)


def retrain_from_validations(batch_size: int = 30) -> dict:
    """Analyse les N dernieres validations + ajuste les parametres.

    Returns:
        Dict avec : new_generation, params_changes, stats
    """
    with SessionLocal() as s:
        gen = current_generation()
        validations = s.execute(
            select(ValidatedSetup)
            .where(ValidatedSetup.generation == gen)
        ).scalars().all()

    if len(validations) < batch_size:
        return {
            "ok": False,
            "reason": f"Pas assez de validations ({len(validations)}/{batch_size}) pour generation {gen}",
        }

    yes = [v for v in validations if v.verdict in ("yes", "edited")]
    no = [v for v in validations if v.verdict == "no"]
    skip = [v for v in validations if v.verdict == "skip"]

    if len(yes) < 5 or len(no) < 5:
        return {
            "ok": False,
            "reason": f"Trop peu de YES ({len(yes)}) ou NO ({len(no)}) pour analyser",
        }

    # ============ Analyse des features ============
    current_params = load_current_params()
    new_params = dict(current_params)
    changes = []

    # 1. push_atr - on prend la moyenne des YES comme nouveau seuil mini
    yes_push_atr = [v.features.get("push_atr", 0) for v in yes if v.features]
    no_push_atr = [v.features.get("push_atr", 0) for v in no if v.features]
    if yes_push_atr and no_push_atr:
        yes_med = statistics.median(yes_push_atr)
        no_med = statistics.median(no_push_atr)
        # Si YES significativement plus impulsif, on serre le seuil
        if yes_med > no_med * 1.2:
            new_min = max(1.5, yes_med * 0.8)   # 80% du median YES
            if abs(new_min - current_params["min_push_atr_mult"]) > 0.2:
                changes.append(f"min_push_atr_mult: {current_params['min_push_atr_mult']:.2f} -> {new_min:.2f}")
                new_params["min_push_atr_mult"] = new_min

    # 2. Killzones - identifier celles ou tu YES le plus
    from collections import Counter
    yes_kz = Counter(v.features.get("killzone", "none") for v in yes if v.features)
    no_kz = Counter(v.features.get("killzone", "none") for v in no if v.features)
    # Killzones acceptables = celles ou YES rate >= 40%
    yes_rates = {}
    for kz in set(list(yes_kz.keys()) + list(no_kz.keys())):
        total = yes_kz.get(kz, 0) + no_kz.get(kz, 0)
        if total < 3:
            continue
        yes_rates[kz] = yes_kz.get(kz, 0) / total
    # Killzones a garder
    allowed = [kz for kz, rate in yes_rates.items() if rate >= 0.40 and kz != "none"]
    if allowed and len(allowed) < len(current_params["allowed_killzones"]):
        changes.append(f"allowed_killzones: {current_params['allowed_killzones']} -> {allowed}")
        new_params["allowed_killzones"] = allowed
        # Si "none" (= HORS killzone) a un yes_rate < 30% -> on rend killzone obligatoire
        none_rate = yes_rates.get("none", 0.0)
        if none_rate < 0.30:
            new_params["killzone_required"] = True
            changes.append("killzone_required: True (HORS killzone yes_rate trop bas)")

    # 3. Heures - on garde les heures ou yes_rate >= 40%
    yes_hours = Counter(v.features.get("hour_utc", -1) for v in yes if v.features)
    no_hours = Counter(v.features.get("hour_utc", -1) for v in no if v.features)
    allowed_hours = []
    for h in range(24):
        total = yes_hours.get(h, 0) + no_hours.get(h, 0)
        if total < 2:
            allowed_hours.append(h)   # pas de data, on laisse
            continue
        rate = yes_hours.get(h, 0) / total
        if rate >= 0.35:
            allowed_hours.append(h)
    if len(allowed_hours) < 24 and len(allowed_hours) >= 6:
        changes.append(f"allowed_hours: {len(current_params['allowed_hours'])}h -> {len(allowed_hours)}h actives")
        new_params["allowed_hours"] = allowed_hours

    # ============ Stats de la generation ============
    yes_wins = sum(1 for v in yes if v.outcome == "win")
    yes_losses = sum(1 for v in yes if v.outcome == "loss")
    yes_wr = (yes_wins / (yes_wins + yes_losses) * 100) if (yes_wins + yes_losses) else 0
    yes_expectancy = sum(v.outcome_pnl_r for v in yes) / len(yes) if yes else 0
    no_wins = sum(1 for v in no if v.outcome == "win")
    no_losses = sum(1 for v in no if v.outcome == "loss")
    no_wr = (no_wins / (no_wins + no_losses) * 100) if (no_wins + no_losses) else 0

    stats = {
        "generation": gen,
        "n_validations": len(validations),
        "yes": len(yes),
        "no": len(no),
        "skip": len(skip),
        "yes_winrate": round(yes_wr, 1),
        "yes_expectancy_r": round(yes_expectancy, 2),
        "no_winrate": round(no_wr, 1),
        "yes_rate": round(len(yes) / (len(yes) + len(no)) * 100, 1) if (len(yes) + len(no)) else 0,
    }

    # ============ Sauvegarde ============
    new_gen = gen + 1
    save_params(new_params)

    with SessionLocal() as s:
        s.add(BotTuning(
            generation=new_gen,
            params=new_params,
            stats=stats,
        ))
        s.commit()

    return {
        "ok": True,
        "old_generation": gen,
        "new_generation": new_gen,
        "changes": changes,
        "stats": stats,
        "params": new_params,
    }


def get_all_generations_stats() -> list[dict]:
    """Historique des generations + leurs stats (pour afficher progres)."""
    with SessionLocal() as s:
        tunings = s.execute(select(BotTuning).order_by(BotTuning.generation)).scalars().all()
        return [{
            "generation": t.generation,
            "stats": t.stats,
            "created_at": t.created_at.isoformat(),
        } for t in tunings]
