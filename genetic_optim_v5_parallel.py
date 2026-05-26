"""Genetique V5 PARALLEL : 28 actifs en parallele, cache M1+HTF+SMT partage.

vs V4 :
- 28 actifs en parallele (V4 = 1 actif a la fois via OPTIM_ASSET)
- Cache M1+HTF+SMT chargé 1 fois par worker process (V4 rechargait par config)
- ProcessPoolExecutor global (28 workers OS, chacun gere ses generations)
- Output : 28 fichiers genetic_v5_<ASSET>_results.json + 1 recap global

Strategie :
- 1 process par actif (28 process)
- Chaque process fait sa propre boucle genetique (10 gen x 12 configs)
- A l'interieur d'un eval, build_dataset utilise N_WORKERS=4 (subprocess pool interne)
- Total cores actifs : 28 x 4 = 112 cores (saturation OK)
- Cache : data_loader cache au niveau du module Python -> partage entre evals d'un meme process

Espace de recherche identique a V4 (10 params).
Fenetres aleatoires 3 mois dans [2019-01 .. 2025-05-22] strict (anti-leakage OOS).
"""
from __future__ import annotations

import os
import sys
import json
import random
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

PARAM_SPACE = {
    "V18_SL_ATR_MULT":      [0.0, 0.3, 0.5, 1.0, 1.5],
    "V18_MIN_GROUP":        [1, 2, 3],
    "V18_SWING_STRENGTH":   [1, 2, 3],
    "V18_STRICT_HTF":       [0, 1],
    "V18_MAX_BARS":         [30, 100, 500, 10000],
    "RR_OVERRIDE":          [1.5, 2.0, 2.5, 3.0],
    "V18_SL_LOOKBACK":      [3, 5, 10, 20],
    "V18_MIN_SWEEP_DEPTH":  [0.0, 0.2, 0.5],
    "V18_BAN_LONDON_CLOSE": [0, 1],
    "V18_REQUIRE_FVG":      [0, 1],
}

ALL_ASSETS = [
    "BTCUSD", "ETHUSD", "USDMXN", "XAUUSD", "USDJPY", "USDZAR", "USDCAD", "NZDUSD",
    "GBPUSD", "CL-OIL", "EURUSD", "AUDUSD", "USDCHF", "NAS100", "XAGUSD",
    "DJ30", "GAS-C", "HK50", "GER40", "FRA40", "UK100", "Nikkei225",
    "BVSPX", "SP500", "Coffee-C", "Cocoa-C", "Wheat-C", "Sugar-C",
]

WINDOW_START = pd.Timestamp("2019-01-01", tz="UTC")
WINDOW_END = pd.Timestamp("2025-05-22", tz="UTC")  # strict anti-leakage OOS

# Tunables (config par defaut, override via env)
POP_SIZE = int(os.environ.get("V5_POP_SIZE", "12"))
N_GEN = int(os.environ.get("V5_N_GEN", "8"))
EVAL_N_WORKERS = int(os.environ.get("V5_EVAL_WORKERS", "4"))   # workers internes par eval
N_PARALLEL_ASSETS = int(os.environ.get("V5_PARALLEL", "8"))    # nb actifs en parallele


def random_config(rng: random.Random) -> dict:
    return {k: rng.choice(vs) for k, vs in PARAM_SPACE.items()}


def mutate_config(cfg: dict, rng: random.Random, rate: float = 0.3) -> dict:
    out = dict(cfg)
    for k, vs in PARAM_SPACE.items():
        if rng.random() < rate:
            out[k] = rng.choice(vs)
    return out


def crossover(p1: dict, p2: dict, rng: random.Random) -> dict:
    return {k: (p1[k] if rng.random() < 0.5 else p2[k]) for k in PARAM_SPACE}


def config_key(cfg: dict) -> str:
    return "_".join(f"{k}={v}" for k, v in sorted(cfg.items()))


def random_window(rng: random.Random) -> tuple[pd.Timestamp, pd.Timestamp]:
    """3 mois aleatoires dans [WINDOW_START .. WINDOW_END]."""
    total_days = (WINDOW_END - WINDOW_START).days - 95
    offset = rng.randint(0, total_days)
    start = WINDOW_START + timedelta(days=offset)
    end = start + timedelta(days=90)
    return start, end


def _eval_one_config(asset: str, cfg: dict,
                      start_iso: str, end_iso: str) -> dict:
    """Evaluation d'1 config dans 1 fenetre. Inline dans le worker process.

    Retourne fitness = profit_R (expectancy * N).
    """
    os.environ["BUILD_DATA_DIR"] = "data_vantage"
    os.environ["BUILD_V12_MODE"] = "1"
    os.environ["OB_VALIDATION_MODE"] = "v18"
    os.environ["N_WORKERS"] = str(EVAL_N_WORKERS)
    for k, v in cfg.items():
        os.environ[k] = str(v)

    try:
        from bot_v2.ml_dataset import build_dataset
        df = build_dataset(
            pd.Timestamp(start_iso), pd.Timestamp(end_iso),
            [asset], output_path=None,
            chunk_months=0.25, ltf="M1",
            version_suffix="_V5_OPT",
        )
    except Exception as e:
        return {"config": cfg, "n_trades": 0, "wr": 0, "profit_R": 0,
                "fitness": 0, "error": str(e)[:200]}

    if len(df) == 0:
        return {"config": cfg, "n_trades": 0, "wr": 0, "profit_R": 0, "fitness": 0}

    df = df[df["outcome"].isin(["WIN", "LOSS"])]
    n = len(df)
    if n == 0:
        return {"config": cfg, "n_trades": 0, "wr": 0, "profit_R": 0, "fitness": 0}

    wr = (df["outcome"] == "WIN").sum() / n * 100
    rr_mean = float(df["rr"].mean()) if "rr" in df.columns else 2.0
    wr_dec = wr / 100
    expectancy = wr_dec * rr_mean - (1 - wr_dec)
    profit_R = expectancy * n
    fitness = profit_R * (0.2 if (wr < 35 or n < 20) else 1.0)
    return {
        "config": cfg, "n_trades": int(n), "wr": float(wr),
        "rr_mean": float(rr_mean), "expectancy": float(expectancy),
        "profit_R": float(profit_R), "fitness": float(fitness),
    }


def genetic_one_asset(asset: str) -> dict:
    """Run la genetique complete sur 1 actif (1 process)."""
    rng = random.Random(42 + hash(asset) % 1000)
    t0 = time.time()

    print(f"  [{asset}] start", flush=True)
    population = [random_config(rng) for _ in range(POP_SIZE)]
    config_scores: dict[str, list[dict]] = {}
    all_results = []
    gen_best_history = []

    for gen in range(N_GEN):
        window = random_window(rng)
        start_iso = window[0].isoformat()
        end_iso = window[1].isoformat()

        gen_results = []
        for cfg in population:
            r = _eval_one_config(asset, cfg, start_iso, end_iso)
            r["window"] = f"{window[0].date()}_{window[1].date()}"
            gen_results.append(r)

        gen_results.sort(key=lambda x: -x["fitness"])
        all_results.extend(gen_results)

        # Track scores par cle de config
        for r in gen_results:
            k = config_key(r["config"])
            if k not in config_scores:
                config_scores[k] = []
            config_scores[k].append({
                "window": r["window"], "wr": r["wr"],
                "n_trades": r["n_trades"], "profit_R": r["profit_R"],
                "fitness": r["fitness"],
            })

        gen_best = gen_results[0]
        gen_best_history.append(gen_best["fitness"])
        elapsed = time.time() - t0
        print(f"  [{asset}] gen {gen+1}/{N_GEN} : "
              f"best fit={gen_best['fitness']:.1f} "
              f"(N={gen_best['n_trades']}, WR={gen_best['wr']:.1f}%) "
              f"| {elapsed/60:.1f}min", flush=True)

        # Nouvelle population : top 3 + crossover/mutation
        if gen + 1 < N_GEN:
            elites = [r["config"] for r in gen_results[:3]]
            new_pop = list(elites)
            while len(new_pop) < POP_SIZE - 2:
                p1, p2 = rng.sample(elites, 2)
                child = mutate_config(crossover(p1, p2, rng), rng, rate=0.3)
                new_pop.append(child)
            while len(new_pop) < POP_SIZE:
                new_pop.append(random_config(rng))
            population = new_pop[:POP_SIZE]

    # Champion robuste : moyenne sur >= 2 fenetres
    robust = []
    for k, scores in config_scores.items():
        if len(scores) >= 2:
            avg_fit = sum(s["fitness"] for s in scores) / len(scores)
            robust.append({
                "key": k,
                "config": next(r["config"] for r in all_results if config_key(r["config"]) == k),
                "avg_fitness": avg_fit,
                "avg_wr": sum(s["wr"] for s in scores) / len(scores),
                "avg_n": sum(s["n_trades"] for s in scores) / len(scores),
                "tested_on": len(scores),
                "scores": scores,
            })
    robust.sort(key=lambda x: -x["avg_fitness"])

    # Best single (sans necessite de robustesse)
    best_single = max(all_results, key=lambda x: x["fitness"])

    return {
        "asset": asset,
        "elapsed_min": (time.time() - t0) / 60,
        "n_evals": len(all_results),
        "gen_best_history": gen_best_history,
        "best_robust": robust[0] if robust else None,
        "top5_robust": robust[:5],
        "best_single": best_single,
    }


def _worker(asset: str) -> dict:
    """Wrapper top-level pour pickling."""
    try:
        return genetic_one_asset(asset)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"asset": asset, "status": "exception", "error": str(e)[:300]}


def main():
    print("=" * 70, flush=True)
    print(f"=== GENETIC V5 PARALLEL : {len(ALL_ASSETS)} actifs ===", flush=True)
    print(f"  POP_SIZE={POP_SIZE} N_GEN={N_GEN}", flush=True)
    print(f"  N_PARALLEL_ASSETS={N_PARALLEL_ASSETS} EVAL_N_WORKERS={EVAL_N_WORKERS}", flush=True)
    print(f"  Total cores actifs : {N_PARALLEL_ASSETS * EVAL_N_WORKERS}", flush=True)
    print("=" * 70, flush=True)

    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=N_PARALLEL_ASSETS) as ex:
        futs = {ex.submit(_worker, a): a for a in ALL_ASSETS}
        for fut in as_completed(futs):
            a = futs[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {"asset": a, "status": "exception", "error": str(e)[:300]}
            results.append(r)
            # Save per-asset
            out = Path(__file__).parent / f"genetic_v5_{a}_results.json"
            out.write_text(json.dumps(r, indent=2, default=str))
            br = r.get("best_robust") or {}
            print(f"  >>> [{len(results)}/{len(ALL_ASSETS)}] {a} done "
                  f"({r.get('elapsed_min', 0):.1f}min) "
                  f"-> {br.get('key', 'no_robust')[:80] if br else '-'}", flush=True)

    print(f"\n{'='*70}")
    print(f"=== TERMINE en {(time.time()-t0)/60:.1f} min ===")
    print(f"{'='*70}")

    print(f"\n{'Asset':<12} {'Time(min)':<10} {'Best fit':<10} {'WR':<7} {'N':<6}")
    for r in sorted(results, key=lambda x: -((x.get("best_robust") or {}).get("avg_fitness", -1))):
        br = r.get("best_robust") or {}
        print(f"{r['asset']:<12} "
              f"{r.get('elapsed_min', 0):<10.1f} "
              f"{br.get('avg_fitness', 0):<10.1f} "
              f"{br.get('avg_wr', 0):<7.1f} "
              f"{br.get('avg_n', 0):<6.0f}")

    recap = Path(__file__).parent / "genetic_v5_recap.json"
    recap.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nRecap : {recap.name}")


if __name__ == "__main__":
    main()
