"""Grid search V12 STYLE BUILD : 20 configs de pipeline filtres, build + train ML
+ verdict WR/AUC OOS pour chacune.

Au lieu de simuler le live cycle par cycle (lent), on fait comme le build V12 :
- Pour chaque config : on appelle build_dataset() avec params modifies via env vars
- On entraine un mini-LightGBM sur 80% des samples, on teste sur 20% (OOS interne)
- On mesure AUC + WR@seuils

Resultat : tableau qui dit quelle config donne le meilleur WR + volume de trades.

C'est ce que tu voulais : "vrai build avec differentes configurations pour
voir le meilleur winrate".

Usage Vast :
    cd /workspace/TradingBot
    nohup python3 -u grid_build_v12.py --asset XAUUSD \
        --start 2026-02-20 --end 2026-05-20 \
        --workers 20 > /workspace/grid_build.log 2>&1 & disown
"""
from __future__ import annotations
import argparse
import os
import sys
import time
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault("SWS_OVERRIDE", "1")
os.environ.setdefault("RR_OVERRIDE", "1.5")
os.environ.setdefault("BUILD_DATA_DIR", "data_vantage")

ROOT = "c:/Users/Shadow/TradingBot" if sys.platform == "win32" else "/workspace/TradingBot"
sys.path.insert(0, ROOT)

import multiprocessing as _mp
_mp.set_start_method("spawn", force=True)


# ============== 20 CONFIGURATIONS ==============
CONFIGS = [
    {"name": "01_baseline_v12",
     "skip_phase": False, "phase_factor": 3.0, "rr": 1.5, "sws": 1, "snapshots": [0]},

    # Variations FILTRE PHASE (impact sur le dataset)
    {"name": "02_no_phase",
     "skip_phase": True, "phase_factor": 3.0, "rr": 1.5, "sws": 1, "snapshots": [0]},
    {"name": "03_phase_15",
     "skip_phase": False, "phase_factor": 1.5, "rr": 1.5, "sws": 1, "snapshots": [0]},
    {"name": "04_phase_20",
     "skip_phase": False, "phase_factor": 2.0, "rr": 1.5, "sws": 1, "snapshots": [0]},
    {"name": "05_phase_25",
     "skip_phase": False, "phase_factor": 2.5, "rr": 1.5, "sws": 1, "snapshots": [0]},

    # Variations RR (impact sur le label WIN/LOSS)
    {"name": "06_rr_10",
     "skip_phase": False, "phase_factor": 3.0, "rr": 1.0, "sws": 1, "snapshots": [0]},
    {"name": "07_rr_20",
     "skip_phase": False, "phase_factor": 3.0, "rr": 2.0, "sws": 1, "snapshots": [0]},
    {"name": "08_rr_30",
     "skip_phase": False, "phase_factor": 3.0, "rr": 3.0, "sws": 1, "snapshots": [0]},

    # Variations SWING_STRENGTH (impact sur la detection OB)
    {"name": "09_sws_2",
     "skip_phase": False, "phase_factor": 3.0, "rr": 1.5, "sws": 2, "snapshots": [0]},
    {"name": "10_sws_3",
     "skip_phase": False, "phase_factor": 3.0, "rr": 1.5, "sws": 3, "snapshots": [0]},

    # COMBINAISONS interessantes
    {"name": "11_nophase_rr20",
     "skip_phase": True, "phase_factor": 3.0, "rr": 2.0, "sws": 1, "snapshots": [0]},
    {"name": "12_nophase_rr10",
     "skip_phase": True, "phase_factor": 3.0, "rr": 1.0, "sws": 1, "snapshots": [0]},
    {"name": "13_phase15_rr20",
     "skip_phase": False, "phase_factor": 1.5, "rr": 2.0, "sws": 1, "snapshots": [0]},
    {"name": "14_phase20_rr10",
     "skip_phase": False, "phase_factor": 2.0, "rr": 1.0, "sws": 1, "snapshots": [0]},
    {"name": "15_sws2_rr20",
     "skip_phase": False, "phase_factor": 3.0, "rr": 2.0, "sws": 2, "snapshots": [0]},

    # === Test "V11 like" (avec snapshots) pour reference ===
    {"name": "16_v11_snapshots",
     "skip_phase": False, "phase_factor": 3.0, "rr": 1.5, "sws": 1, "snapshots": [3, 7, 15]},

    # COMBOS aggressives (max volume)
    {"name": "17_aggro_nophase_sws2",
     "skip_phase": True, "phase_factor": 1.5, "rr": 1.5, "sws": 2, "snapshots": [0]},
    {"name": "18_aggro_nophase_rr10",
     "skip_phase": True, "phase_factor": 1.5, "rr": 1.0, "sws": 1, "snapshots": [0]},

    # COMBOS conservateurs (max WR)
    {"name": "19_conservateur_phase25_rr30",
     "skip_phase": False, "phase_factor": 2.5, "rr": 3.0, "sws": 1, "snapshots": [0]},
    {"name": "20_ultra_phase15_rr20_sws2",
     "skip_phase": False, "phase_factor": 1.5, "rr": 2.0, "sws": 2, "snapshots": [0]},
]


def run_one_build(args_tuple):
    """Worker : build dataset + train ML pour 1 config."""
    config, asset, start_str, end_str = args_tuple
    try:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["BUILD_DATA_DIR"] = "data_vantage"
        os.environ["SWS_OVERRIDE"] = str(config["sws"])
        os.environ["RR_OVERRIDE"] = str(config["rr"])
        os.environ["BUILD_V12_MODE"] = "1" if config["snapshots"] == [0] else "0"

        import sys as _sys
        _sys.path.insert(0, ROOT)

        # Patch market_phase si besoin (avant import du dataset)
        if config["skip_phase"] or config["phase_factor"] != 3.0:
            from bot_v2.concepts import market_phase as _mp_module
            from bot_v2.concepts.market_phase import MarketPhaseAnalysis
            _original = _mp_module.analyze_phase
            _factor = config["phase_factor"]
            _skip = config["skip_phase"]
            def patched(df, at_index, lookback=20):
                r = _original(df, at_index, lookback=lookback)
                if _skip and r.phase == "accumulation":
                    return MarketPhaseAnalysis(
                        phase="expansion", is_tradable=True,
                        range_size=r.range_size, avg_body_size=r.avg_body_size,
                        last_displacement=r.last_displacement,
                        reason="GRID SKIP_PHASE",
                    )
                if _factor != 3.0 and r.phase == "accumulation":
                    window = df.iloc[max(0, at_index - lookback):at_index + 1]
                    rng = float(window["high"].max() - window["low"].min())
                    atr = float((window["high"] - window["low"]).mean())
                    if atr > 0 and rng >= _factor * atr:
                        return MarketPhaseAnalysis(
                            phase="undetermined", is_tradable=True,
                            range_size=rng, avg_body_size=r.avg_body_size,
                            last_displacement=r.last_displacement,
                            reason=f"GRID phase_factor={_factor}",
                        )
                return r
            _mp_module.analyze_phase = patched

        # Patch snapshots si V11-like
        if config["snapshots"] != [0]:
            from bot_v2 import ml_dataset as _ds
            # Hack : on patche la constante SNAPSHOTS_K dans la fonction
            # Plus simple : on garde le mode V11 via BUILD_V12_MODE=0

        import pandas as _pd
        from bot_v2.ml_dataset import build_dataset

        start = _pd.Timestamp(start_str, tz="UTC")
        end = _pd.Timestamp(end_str, tz="UTC")

        t0 = time.time()
        df = build_dataset(
            start, end, [asset],
            output_path=None,
            chunk_months=0.5,  # ~15j par chunk, parallelisable interne
            ltf="M1",
            version_suffix=f"_GRID_{config['name']}",
        )
        elapsed_build = time.time() - t0

        # Filter
        closed = df[df["outcome"].isin(["WIN", "LOSS"])].copy()
        if len(closed) < 100:
            return {
                "ok": False, "config_name": config["name"], "asset": asset,
                "error": f"trop peu de trades ({len(closed)})",
            }

        # Train mini-LightGBM
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score
        closed["ts"] = _pd.to_datetime(closed["ts"], utc=True)
        closed = closed.sort_values("ts").reset_index(drop=True)
        closed["target"] = (closed["outcome"] == "WIN").astype(int)
        NON_FEATS = {"instrument", "ts", "direction", "outcome", "pnl_usd", "bars_to_exit", "target"}
        feat_cols = [c for c in closed.columns if c not in NON_FEATS]
        for c in feat_cols:
            if closed[c].dtype == bool:
                closed[c] = closed[c].astype(int)

        # Split 80/20 temporel
        n = len(closed)
        split = int(n * 0.8)
        train_df = closed.iloc[:split]
        oos_df = closed.iloc[split:]
        X_train = train_df[feat_cols]
        y_train = train_df["target"]
        X_oos = oos_df[feat_cols]
        y_oos = oos_df["target"]

        t_train = time.time()
        model = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=9, num_leaves=63,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbosity=-1,
        )
        model.fit(X_train, y_train)
        elapsed_train = time.time() - t_train

        # Predict OOS
        oos_proba = model.predict_proba(X_oos)[:, 1]
        try:
            auc_oos = float(roc_auc_score(y_oos, oos_proba))
        except Exception:
            auc_oos = float("nan")
        wr_brut = float(y_oos.mean() * 100)

        # WR par seuil
        wr_at = {}
        n_at = {}
        for thr in (0.55, 0.60, 0.65, 0.70, 0.75):
            mask = oos_proba >= thr
            n_at[thr] = int(mask.sum())
            wr_at[thr] = float(y_oos[mask].mean() * 100) if mask.any() else 0

        return {
            "ok": True,
            "config_name": config["name"],
            "config": config,
            "asset": asset,
            "n_total": len(df),
            "n_closed": len(closed),
            "n_train": len(train_df),
            "n_oos": len(oos_df),
            "wr_brut_oos": wr_brut,
            "auc_oos": auc_oos,
            "n_at_065": n_at[0.65],
            "wr_at_065": wr_at[0.65],
            "n_at_070": n_at[0.70],
            "wr_at_070": wr_at[0.70],
            "n_at_055": n_at[0.55],
            "wr_at_055": wr_at[0.55],
            "elapsed_build_s": elapsed_build,
            "elapsed_train_s": elapsed_train,
        }
    except Exception as e:
        import traceback
        return {
            "ok": False, "config_name": config["name"], "asset": asset,
            "error": str(e)[:200],
            "trace": traceback.format_exc()[-500:],
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--asset", default="XAUUSD")
    p.add_argument("--start", default="2026-02-20")
    p.add_argument("--end", default="2026-05-20")
    p.add_argument("--workers", type=int, default=20)
    args = p.parse_args()

    import pandas as pd
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")

    print(f"=== GRID BUILD V12 ===", flush=True)
    print(f"Asset    : {args.asset}", flush=True)
    print(f"Periode  : {args.start} -> {args.end}", flush=True)
    print(f"Configs  : {len(CONFIGS)}", flush=True)
    print(f"Workers  : {args.workers}", flush=True)
    print()

    tasks = [(c, args.asset, args.start, args.end) for c in CONFIGS]

    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_one_build, t): t[0]["name"] for t in tasks}
        done = 0
        for fut in as_completed(futures):
            name = futures[fut]
            done += 1
            try:
                r = fut.result()
                results.append(r)
                if r.get("ok"):
                    print(f"[{done:2d}/{len(tasks)}] {r['config_name']:37s} "
                          f"AUC={r['auc_oos']:.3f} "
                          f"n@0.65={r['n_at_065']:4d} WR@0.65={r['wr_at_065']:5.1f}% "
                          f"n@0.70={r['n_at_070']:4d} WR@0.70={r['wr_at_070']:5.1f}% "
                          f"({r['elapsed_build_s']:.0f}s)",
                          flush=True)
                else:
                    print(f"[{done:2d}/{len(tasks)}] FAIL {name} : {r.get('error','')[:100]}",
                          flush=True)
            except Exception as e:
                print(f"[{done:2d}/{len(tasks)}] EXCEPTION {name}: {e}", flush=True)

    elapsed = time.time() - t0
    print(f"\n=== TERMINE en {elapsed/60:.1f}min ===", flush=True)

    # Tri par PnL implicite : WR * n (= esperance trades gagnants)
    ok = [r for r in results if r.get("ok")]
    ok.sort(key=lambda r: r["wr_at_065"] * r["n_at_065"], reverse=True)

    print(f"\n=== RECAP - tri par (WR * N) a 0.65 ===")
    print(f"{'CONFIG':<37}{'AUC':<7}{'N@0.65':<8}{'WR@0.65':<9}{'N@0.70':<8}{'WR@0.70':<9}")
    print("-" * 80)
    for r in ok:
        print(f"{r['config_name']:<37}{r['auc_oos']:<7.3f}"
              f"{r['n_at_065']:<8}{r['wr_at_065']:<9.1f}"
              f"{r['n_at_070']:<8}{r['wr_at_070']:<9.1f}",
              flush=True)

    # JSON
    out = f"{ROOT}/grid_build_results.json"
    with open(out, "w") as f:
        json.dump([{k: v for k, v in r.items() if k != "trace"} for r in results],
                  f, indent=2, default=str)
    print(f"\nResultats sauves : {out}", flush=True)


if __name__ == "__main__":
    main()
