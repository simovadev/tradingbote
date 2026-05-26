"""Optim génétique V2 — chaque génération sur une fenêtre 3 mois ALEATOIRE.

Améliorations vs V1 :
1. Chaque génération teste sur une PERIODE DIFFERENTE (3 mois choisis au hasard sur 2019-2025)
   → la config gagnante doit etre ROBUSTE sur plusieurs périodes (pas overfit 1 créneau)
2. Early stopping : si le best ne progresse plus pendant 3 gen, on s'arrête
3. Min 10 gen, max 30 gen (sécurité)
4. Fitness = profit en R-multiples × N_trades (récompense volume + qualité)
5. Track : pour chaque config TOP, on stocke ses scores sur TOUTES les périodes vues
   → la "champion config" = celle avec le meilleur score MOYEN sur toutes les fenêtres
"""
from __future__ import annotations
import os
import sys
import json
import random
import time
import math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd


# ==== HYPERPARAM SPACE ====
PARAM_SPACE = {
    "V18_SL_ATR_MULT": [0.0, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
    "V18_MIN_GROUP": [1, 2, 3],
    "V18_SWING_STRENGTH": [1, 2, 3],
    "V18_STRICT_HTF": [0, 1],
    "V18_MAX_BARS": [30, 100, 500, 10000],
}

ASSET = "BTCUSD"

# Fenêtre globale disponible (BTCUSD : 2018-2026, prend marge à droite pour OOS futur)
# BUG #5 FIX (2026-05-26) : avant = 2025-12-31, qui CHEVAUCHE l'OOS (commence 2025-11-22).
# Les fenetres aleatoires 90j tombaient parfois dans l'OOS -> hyperparams OB selectionnes
# sur l'OOS = contamination. On force la limite a TRAIN_END strict (2025-05-22).
WINDOW_GLOBAL_START = pd.Timestamp("2019-01-01", tz="UTC")
WINDOW_GLOBAL_END   = pd.Timestamp("2025-05-22", tz="UTC")  # = TRAIN_END strict

# Critère d'arrêt
MIN_GENERATIONS = 10
MAX_GENERATIONS = 30
PLATEAU_GEN     = 3  # si best ne bouge pas pendant 3 gen → stop


def random_window(rng: random.Random) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Tire une fenêtre 3 mois aléatoire dans 2019-2025."""
    total_days = (WINDOW_GLOBAL_END - WINDOW_GLOBAL_START).days - 95  # marge 3 mois
    days_offset = rng.randint(0, total_days)
    start = WINDOW_GLOBAL_START + timedelta(days=days_offset)
    end = start + timedelta(days=90)
    return start, end


def random_config(rng: random.Random) -> dict:
    return {k: rng.choice(vs) for k, vs in PARAM_SPACE.items()}


def mutate_config(cfg: dict, rng: random.Random, mutation_rate=0.3) -> dict:
    new = dict(cfg)
    for k, vs in PARAM_SPACE.items():
        if rng.random() < mutation_rate:
            new[k] = rng.choice(vs)
    return new


def config_key(cfg: dict) -> str:
    """Clé unique pour identifier une config (pour tracking multi-périodes)."""
    return "_".join(f"{k}={v}" for k, v in sorted(cfg.items()))


def evaluate_config(args: tuple) -> dict:
    """Build dataset avec cette config sur cette fenêtre et retourne stats."""
    cfg, start_iso, end_iso = args
    start = pd.Timestamp(start_iso)
    end = pd.Timestamp(end_iso)

    os.environ["BUILD_DATA_DIR"] = "data_vantage"
    os.environ["BUILD_V12_MODE"] = "1"
    os.environ["OB_VALIDATION_MODE"] = "v18"
    os.environ["V18_STRICT_HTF"] = str(cfg["V18_STRICT_HTF"])
    os.environ["V18_SL_ATR_MULT"] = str(cfg["V18_SL_ATR_MULT"])
    os.environ["V18_MIN_GROUP"] = str(cfg["V18_MIN_GROUP"])
    os.environ["V18_SWING_STRENGTH"] = str(cfg["V18_SWING_STRENGTH"])
    os.environ["V18_MAX_BARS"] = str(cfg["V18_MAX_BARS"])
    # V3 BOOST : chaque eval lance 5 chunks workers (au lieu de 1)
    # → 20 configs × 5 workers = 100 cores actifs (au lieu de 10)
    os.environ["N_WORKERS"] = os.environ.get("EVAL_N_WORKERS", "5")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from bot_v2.ml_dataset import build_dataset

        suffix = f"_OPT_{abs(hash((config_key(cfg), start_iso))) % 100000}"
        df = build_dataset(
            start, end, [ASSET],
            output_path=None,
            chunk_months=0.25, ltf="M1",
            version_suffix=suffix,
        )
    except Exception as e:
        return {"config": cfg, "n_trades": 0, "wr": 0, "profit_R": 0, "fitness": 0,
                "window": f"{start.date()}_{end.date()}", "error": str(e)[:200]}

    if len(df) == 0:
        return {"config": cfg, "n_trades": 0, "wr": 0, "profit_R": 0, "fitness": 0,
                "window": f"{start.date()}_{end.date()}", "error": "no trades"}

    df = df[df["outcome"].isin(["WIN", "LOSS"])]
    n = len(df)
    if n == 0:
        return {"config": cfg, "n_trades": 0, "wr": 0, "profit_R": 0, "fitness": 0,
                "window": f"{start.date()}_{end.date()}", "error": "no closed trades"}

    wr = (df["outcome"] == "WIN").sum() / n * 100
    rr_mean = float(df["rr"].mean()) if "rr" in df.columns else 2.0
    wr_dec = wr / 100
    expectancy = wr_dec * rr_mean - (1 - wr_dec)
    profit_R = expectancy * n

    # Pénalisation pour configs catastrophiques (WR très bas) ou trop peu de volume
    if wr < 35 or n < 20:
        fitness = profit_R * 0.2
    else:
        fitness = profit_R

    return {
        "config": cfg,
        "n_trades": int(n),
        "wr": float(wr),
        "rr_mean": float(rr_mean),
        "expectancy": float(expectancy),
        "profit_R": float(profit_R),
        "fitness": float(fitness),
        "window": f"{start.date()}_{end.date()}",
        "error": None,
    }


def run_generation(configs: list[dict], window: tuple, n_workers: int = 10) -> list[dict]:
    start_iso = window[0].isoformat()
    end_iso = window[1].isoformat()
    print(f"\n  Fenetre : {window[0].date()} -> {window[1].date()}", flush=True)
    print(f"  Eval de {len(configs)} configs avec {n_workers} workers...", flush=True)

    results = []
    args_list = [(cfg, start_iso, end_iso) for cfg in configs]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(evaluate_config, a): i for i, a in enumerate(args_list)}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {"config": configs[i], "n_trades": 0, "wr": 0, "profit_R": 0,
                     "fitness": 0, "window": f"{window[0].date()}_{window[1].date()}",
                     "error": str(e)[:200]}
            results.append(r)
            cfg_str = ' '.join(f'{k.replace("V18_","")}={v}' for k, v in r["config"].items())
            err = r.get("error", None)
            err_str = f" ERR={err[:50]}" if err else ""
            pr = r.get("profit_R", 0)
            wr = r.get("wr", 0)
            n = r.get("n_trades", 0)
            print(f"    -> WR={wr:5.1f}%  N={n:>5}  +{pr:+7.1f}R  fit={r['fitness']:7.1f}  | {cfg_str}{err_str}", flush=True)
    results.sort(key=lambda x: x["fitness"], reverse=True)
    print(f"  ⏱  {time.time()-t0:.0f}s", flush=True)
    return results


def main():
    rng = random.Random(42)
    # V3 BOOST : pop size 20 au lieu de 10 (plus de diversité par gen)
    N_CONFIGS = int(os.environ.get("POP_SIZE", "20"))
    N_WORKERS = int(os.environ.get("OPTIM_WORKERS", "20"))

    print(f"=== GENETIQUE V2 sur {ASSET} ===", flush=True)
    print(f"Min gen: {MIN_GENERATIONS}, Max gen: {MAX_GENERATIONS}, Plateau stop: {PLATEAU_GEN}", flush=True)
    print(f"Pop size: {N_CONFIGS}, Workers: {N_WORKERS}", flush=True)
    print(f"Fenetres : 3 mois ALEATOIRES dans 2019-01 -> 2025-12", flush=True)

    population = [random_config(rng) for _ in range(N_CONFIGS)]

    # Track : pour chaque config (par key), liste de ses scores sur fenetres differentes
    config_scores: dict[str, list[dict]] = {}
    all_results = []  # tous les eval bruts
    gen_best_history = []  # best_fitness par gen
    plateau_count = 0

    for gen in range(MAX_GENERATIONS):
        window = random_window(rng)
        print(f"\n{'='*70}")
        print(f"=== GEN {gen+1}/{MAX_GENERATIONS} ===")
        print(f"{'='*70}")

        results = run_generation(population, window, n_workers=N_WORKERS)
        all_results.extend(results)

        # Track multi-window pour chaque config
        for r in results:
            k = config_key(r["config"])
            if k not in config_scores:
                config_scores[k] = []
            config_scores[k].append({
                "window": r["window"],
                "wr": r["wr"],
                "n_trades": r["n_trades"],
                "profit_R": r["profit_R"],
                "fitness": r["fitness"],
            })

        gen_best = results[0]["fitness"]
        gen_best_history.append(gen_best)
        print(f"\n  ★ GEN {gen+1} best fitness : {gen_best:.1f} ({results[0]['n_trades']} trades, WR {results[0]['wr']:.1f}%, +{results[0]['profit_R']:.0f}R)")

        # Calcul du "champion robuste" : meilleure config par fitness MOYEN sur tous ses essais
        robust_top = []
        for k, scores in config_scores.items():
            if len(scores) >= 2:  # au moins 2 fenêtres testées
                avg_fit = sum(s["fitness"] for s in scores) / len(scores)
                avg_wr  = sum(s["wr"] for s in scores) / len(scores)
                avg_n   = sum(s["n_trades"] for s in scores) / len(scores)
                avg_pr  = sum(s["profit_R"] for s in scores) / len(scores)
                robust_top.append({
                    "key": k, "avg_fitness": avg_fit, "avg_wr": avg_wr,
                    "avg_n": avg_n, "avg_pR": avg_pr, "tested_on": len(scores),
                })
        robust_top.sort(key=lambda x: x["avg_fitness"], reverse=True)
        if robust_top:
            r = robust_top[0]
            print(f"  💎 CHAMPION ROBUSTE (testé sur {r['tested_on']} fenêtres):")
            print(f"     {r['key']}")
            print(f"     avg_WR={r['avg_wr']:.1f}%, avg_N={r['avg_n']:.0f}, avg_profit={r['avg_pR']:+.0f}R, avg_fit={r['avg_fitness']:.1f}")

        # Stopping criterion
        if gen + 1 >= MIN_GENERATIONS:
            # Plateau : si les 3 dernières gen ont best <= max des PLATEAU_GEN précédentes
            recent = gen_best_history[-PLATEAU_GEN:]
            older_max = max(gen_best_history[:-PLATEAU_GEN]) if len(gen_best_history) > PLATEAU_GEN else 0
            if max(recent) <= older_max * 1.05:  # moins de 5% d'amélioration
                plateau_count += 1
                print(f"\n  ⏸ Plateau détecté ({plateau_count}/{PLATEAU_GEN} gen sans progrès > 5%)")
                if plateau_count >= 1:  # 1 fois suffit si on est sur min_gen
                    print(f"\n  🛑 ARRET PRECOCE : plateau confirmé après {gen+1} générations")
                    break
            else:
                plateau_count = 0

        # Reproduction : top 6 + mutations + random
        if gen + 1 < MAX_GENERATIONS:
            top_n = max(3, N_CONFIGS // 4)  # 5 si N_CONFIGS=20
            tops = [r["config"] for r in results[:top_n]]
            new_pop = list(tops)  # élitisme
            # Mute le best (le plus exploité)
            mutations_best = (N_CONFIGS - top_n) // 2
            for _ in range(mutations_best):
                new_pop.append(mutate_config(tops[0], rng, mutation_rate=0.4))
            # Mute le 2eme et 3eme
            for _ in range((N_CONFIGS - top_n) // 4):
                new_pop.append(mutate_config(tops[1] if len(tops) > 1 else tops[0], rng, mutation_rate=0.3))
            # Random pour explorer
            while len(new_pop) < N_CONFIGS:
                new_pop.append(random_config(rng))
            population = new_pop[:N_CONFIGS]

    # === RESUME FINAL ===
    print(f"\n{'='*70}")
    print(f"=== RESUME FINAL ({len(all_results)} evals sur {len(gen_best_history)} générations) ===")
    print(f"{'='*70}")

    # Re-calcul champion final
    robust_final = []
    for k, scores in config_scores.items():
        if len(scores) >= 2:
            avg_fit = sum(s["fitness"] for s in scores) / len(scores)
            std_fit = (sum((s["fitness"] - avg_fit)**2 for s in scores) / len(scores)) ** 0.5
            robust_final.append({
                "key": k,
                "config": next(r["config"] for r in all_results if config_key(r["config"]) == k),
                "scores": scores,
                "avg_fitness": avg_fit,
                "std_fitness": std_fit,
                "avg_wr": sum(s["wr"] for s in scores) / len(scores),
                "avg_n": sum(s["n_trades"] for s in scores) / len(scores),
                "avg_pR": sum(s["profit_R"] for s in scores) / len(scores),
                "tested_on": len(scores),
            })
    robust_final.sort(key=lambda x: x["avg_fitness"], reverse=True)

    print(f"\n💎 TOP 5 CHAMPIONS ROBUSTES (tested >= 2 windows):")
    for i, r in enumerate(robust_final[:5]):
        print(f"\n  #{i+1} (testé sur {r['tested_on']} fenêtres)")
        print(f"    avg WR={r['avg_wr']:.1f}%  avg N={r['avg_n']:.0f}  avg profit=+{r['avg_pR']:.0f}R  avg fit={r['avg_fitness']:.1f} (std={r['std_fitness']:.1f})")
        cfg_str = ' '.join(f'{k.replace("V18_","")}={v}' for k, v in r["config"].items())
        print(f"    Config: {cfg_str}")
        for s in r["scores"][:5]:
            print(f"      - {s['window']}: WR={s['wr']:.1f}% N={s['n_trades']} +{s['profit_R']:+.0f}R")

    # Sauvegarde
    out = {
        "best_robust": robust_final[0] if robust_final else None,
        "top_5": robust_final[:5],
        "all_evals": all_results,
        "gen_best_history": gen_best_history,
    }
    Path("genetic_v2_results.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nResultats sauves dans genetic_v2_results.json")


if __name__ == "__main__":
    main()
