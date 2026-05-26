"""GENETIC HYPERPARAMS ML V18.4 — optimisation LightGBM par actif.

Tune les hyperparams du modele ML (max_depth, num_leaves, lr, reg_alpha...)
sur chaque actif, en utilisant le dataset V18.4 DEJA BUILDE.

Pourquoi rapide :
- Pas de rebuild OB/features (dataset deja en parquet)
- Eval = 1 fit LightGBM + 1 predict (~10-30s par config)
- Parallel : 28 actifs simultanes (1 worker = 1 actif x ses 100 configs)
- Mais chaque LightGBM utilise n_jobs=4 -> 28 x 4 = 112 cores actifs

Fitness = AUC sur val_genetic (split intermediaire, distinct de OOS final).
Anti-overfit : on garde train/val_gen/oos separes.

Split temporel :
    TRAIN     : 2018-03 -> 2024-11
    VAL_GEN   : 2024-11 -> 2025-05 (fitness genetic)
    OOS       : 2025-11 -> 2026-05 (test final, jamais vu par genetic)
    (gap 2025-05 -> 2025-11 = embargo + safety margin)

Output : ml_hyperparams_<ASSET>_v18_4.json (config gagnante)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = "c:/Users/Shadow/TradingBot" if sys.platform == "win32" else "/workspace/TradingBot"
sys.path.insert(0, ROOT)

ALL_ASSETS = [
    "BTCUSD", "ETHUSD", "USDMXN", "XAUUSD", "USDJPY", "USDZAR", "USDCAD", "NZDUSD",
    "GBPUSD", "CL-OIL", "EURUSD", "AUDUSD", "USDCHF", "NAS100", "XAGUSD",
    "DJ30", "GAS-C", "HK50", "GER40", "FRA40", "UK100", "Nikkei225",
    "BVSPX", "SP500", "Coffee-C", "Cocoa-C", "Wheat-C", "Sugar-C",
]

# Split temporel
TRAIN_END = pd.Timestamp("2024-11-01", tz="UTC")
VAL_GEN_START = pd.Timestamp("2024-11-15", tz="UTC")   # embargo 2 semaines
VAL_GEN_END = pd.Timestamp("2025-05-22", tz="UTC")
OOS_START = pd.Timestamp("2025-11-24", tz="UTC")
OOS_END = pd.Timestamp("2026-05-22", tz="UTC")

NON_FEATURES = {
    "instrument", "ts", "direction",
    "outcome", "pnl_usd", "bars_to_exit",
    "pnl_R", "pnl_R_clipped",
}

# Espace de recherche : 9 hyperparams
SEARCH_SPACE = {
    "n_estimators":      [500, 1000, 1500, 2000, 3000],
    "learning_rate":     [0.005, 0.01, 0.02, 0.03, 0.05],
    "max_depth":         [3, 4, 5, 6, 7, 8],
    "num_leaves":        [15, 23, 31, 47, 63, 95],
    "min_child_samples": [50, 100, 200, 300, 500, 800],
    "subsample":         [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree":  [0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
    "reg_alpha":         [0.0, 0.1, 0.5, 1.0, 3.0, 10.0],
    "reg_lambda":        [0.0, 1.0, 3.0, 5.0, 10.0, 30.0],
}


def random_config() -> dict:
    return {k: random.choice(v) for k, v in SEARCH_SPACE.items()}


def crossover(p1: dict, p2: dict) -> dict:
    """Uniform crossover : pour chaque gene, 50% chance p1 ou p2."""
    return {k: (p1[k] if random.random() < 0.5 else p2[k]) for k in SEARCH_SPACE}


def mutate(cfg: dict, rate: float = 0.2) -> dict:
    """Mutation : pour chaque gene, rate% chance de tirer une nouvelle valeur."""
    out = dict(cfg)
    for k in SEARCH_SPACE:
        if random.random() < rate:
            out[k] = random.choice(SEARCH_SPACE[k])
    return out


def load_dataset(asset: str) -> pd.DataFrame | None:
    """Charge dataset V18.4 (priorite), fallback V18.3."""
    for suffix in ("v18_4", "v18_3"):
        p = Path(f"{ROOT}/data/ml_dataset_{asset}_vantage_{suffix}.parquet")
        if p.exists():
            df = pd.read_parquet(p)
            df["ts"] = pd.to_datetime(df["ts"], utc=True)
            df = df.sort_values("ts").reset_index(drop=True)
            return df
    return None


def prepare_xy(df: pd.DataFrame):
    df = df[df["outcome"].isin(["WIN", "LOSS"])].copy()
    df["target"] = (df["outcome"] == "WIN").astype(int)
    df = df.sort_values("ts").reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in NON_FEATURES and c != "target"]
    X = df[feature_cols].copy()
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
        elif X[c].dtype == "object" or X[c].dtype.name == "category":
            X[c] = X[c].astype("category")
    y = df["target"]
    return X, y


def eval_config(cfg: dict, X_train, y_train, X_val, y_val) -> float:
    """Train + eval AUC sur val. Retourne -inf si crash."""
    try:
        model = lgb.LGBMClassifier(
            **cfg,
            min_split_gain=0.01,
            random_state=42,
            verbosity=-1,
            n_jobs=2,
        )
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
        proba = model.predict_proba(X_val)[:, 1]
        return float(roc_auc_score(y_val, proba))
    except Exception as e:
        print(f"    [crash] {e}")
        return -1.0


def genetic_one_asset(asset: str, n_gen: int = 8, pop_size: int = 10,
                       elite: int = 3, mut_rate: float = 0.25,
                       seed: int = 42) -> dict:
    """Genetique sur 1 actif. Retourne best config + best auc."""
    random.seed(seed + hash(asset) % 1000)
    np.random.seed(seed + hash(asset) % 1000)

    df = load_dataset(asset)
    if df is None:
        return {"asset": asset, "status": "no_data"}

    df_closed = df[df["outcome"].isin(["WIN", "LOSS"])].copy()
    train_df = df_closed[df_closed["ts"] < TRAIN_END]
    val_df = df_closed[(df_closed["ts"] >= VAL_GEN_START) & (df_closed["ts"] < VAL_GEN_END)]

    if len(train_df) < 200 or len(val_df) < 50:
        return {"asset": asset, "status": "bad_split",
                "n_train": len(train_df), "n_val": len(val_df)}

    X_train, y_train = prepare_xy(train_df)
    X_val, y_val = prepare_xy(val_df)
    # Aligner colonnes (au cas ou)
    common = [c for c in X_train.columns if c in X_val.columns]
    X_train = X_train[common]
    X_val = X_val[common]

    print(f"  [{asset}] train={len(X_train)} val={len(X_val)} features={len(common)}")

    # Population initiale
    population = [random_config() for _ in range(pop_size)]
    # Ajoute la config "baseline V18.2" en seed
    population[0] = {
        "n_estimators": 2000, "learning_rate": 0.01,
        "max_depth": 5, "num_leaves": 31, "min_child_samples": 200,
        "subsample": 0.7, "colsample_bytree": 0.6,
        "reg_alpha": 1.0, "reg_lambda": 5.0,
    }

    history = []
    best_ever = None
    best_auc_ever = -1.0

    for gen in range(n_gen):
        # Eval
        scored = []
        for i, cfg in enumerate(population):
            auc = eval_config(cfg, X_train, y_train, X_val, y_val)
            scored.append((auc, cfg))
        scored.sort(key=lambda x: -x[0])
        best_auc, best_cfg = scored[0]
        worst_auc = scored[-1][0]
        avg_auc = float(np.mean([s[0] for s in scored if s[0] > -1]))
        history.append({"gen": gen, "best": best_auc, "avg": avg_auc, "worst": worst_auc})
        print(f"  [{asset}] gen {gen+1}/{n_gen}: best={best_auc:.4f} avg={avg_auc:.4f}")

        if best_auc > best_auc_ever:
            best_auc_ever = best_auc
            best_ever = best_cfg

        # Selection : elite + crossover/mutation
        elites = [cfg for _, cfg in scored[:elite]]
        children = []
        while len(children) < pop_size - elite:
            p1, p2 = random.sample(elites, 2)
            child = mutate(crossover(p1, p2), rate=mut_rate)
            children.append(child)
        population = elites + children

    return {
        "asset": asset,
        "status": "ok",
        "best_auc_val_gen": best_auc_ever,
        "best_config": best_ever,
        "history": history,
        "n_train": len(X_train),
        "n_val": len(X_val),
    }


def _worker(asset: str, n_gen: int, pop_size: int) -> dict:
    """Wrapper pour ProcessPoolExecutor."""
    try:
        return genetic_one_asset(asset, n_gen=n_gen, pop_size=pop_size)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"asset": asset, "status": "exception", "error": str(e)[:300]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("asset", nargs="?", help="Asset ou --all")
    p.add_argument("--all", action="store_true")
    p.add_argument("--n_gen", type=int, default=8)
    p.add_argument("--pop_size", type=int, default=10)
    p.add_argument("--n_parallel", type=int, default=8,
                   help="Nb actifs traites en parallele")
    args = p.parse_args()

    if args.all or args.asset == "--all":
        assets = ALL_ASSETS
        print(f"=== GENETIC HYPERPARAMS ML V18.4 ===")
        print(f"Assets : {len(assets)}")
        print(f"Pop : {args.pop_size} | Gen : {args.n_gen}")
        print(f"Parallel : {args.n_parallel}")
        print(f"Split val_gen : {VAL_GEN_START.date()} -> {VAL_GEN_END.date()}")
        results = []
        with ProcessPoolExecutor(max_workers=args.n_parallel) as ex:
            futs = {ex.submit(_worker, a, args.n_gen, args.pop_size): a for a in assets}
            for fut in as_completed(futs):
                a = futs[fut]
                try:
                    r = fut.result()
                except Exception as e:
                    r = {"asset": a, "status": "exception", "error": str(e)[:200]}
                results.append(r)
                print(f"  >>> [{len(results)}/{len(assets)}] {a} : "
                      f"{r.get('status', '?')} "
                      f"best_auc={r.get('best_auc_val_gen', 0):.4f}")
                # Save best config par actif
                if r.get("status") == "ok":
                    out = Path(f"{ROOT}/ml_hyperparams_{a}_v18_4.json")
                    out.write_text(json.dumps(r, indent=2, default=str))

        # Recap global
        print("\n" + "=" * 80)
        print("=== RECAP GENETIC HYPERPARAMS V18.4 ===")
        print("=" * 80)
        print(f"{'Asset':<12} {'Status':<10} {'Best AUC val_gen':<18} {'N train':<10}")
        for r in sorted(results, key=lambda x: -x.get("best_auc_val_gen", -1)):
            print(f"{r['asset']:<12} {r.get('status','?'):<10} "
                  f"{r.get('best_auc_val_gen', 0):.4f}              {r.get('n_train', '--')}")
        recap = Path(f"{ROOT}/ml_hyperparams_v18_4_recap.json")
        recap.write_text(json.dumps(results, indent=2, default=str))
        print(f"\nRecap : {recap.name}")

    elif args.asset:
        r = genetic_one_asset(args.asset, n_gen=args.n_gen, pop_size=args.pop_size)
        print(json.dumps(r, indent=2, default=str))
        if r.get("status") == "ok":
            out = Path(f"{ROOT}/ml_hyperparams_{args.asset}_v18_4.json")
            out.write_text(json.dumps(r, indent=2, default=str))
            print(f"Sauve : {out.name}")
    else:
        p.print_help()


if __name__ == "__main__":
    main()
