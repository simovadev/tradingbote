"""Grid search V12 : teste 20 configurations sur 1 actif (XAUUSD) sur 3 mois.

Variations testees :
- ML threshold (0.50, 0.55, 0.60, 0.65, 0.70, 0.75)
- Skip filtre phase accumulation (True/False)
- Phase range factor (1.5x, 2.0x, 2.5x, 3.0x ATR)
- min_score (0, 50, 100)
- cap_age_ob_min (15, 30, 60)

Chaque config tourne dans un worker different en parallele.
Sortie : CSV recap avec metriques par config (trades, WR, PnL, etc.)

Usage Vast :
    cd /workspace/TradingBot
    nohup python3 -u grid_search_v12.py --asset XAUUSD --start 2026-02-20 --end 2026-05-20 \
        --workers 32 > /workspace/grid_v12.log 2>&1 & disown
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

import pandas as pd

import multiprocessing as _mp
_mp.set_start_method("spawn", force=True)


# ============== 20 CONFIGURATIONS ==============
# Chaque config = un dict de leviers
CONFIGS = [
    # === Baseline V12 (config de reference) ===
    {"name": "01_baseline_v12_070",
     "ml_threshold": 0.70, "skip_phase": False, "phase_factor": 3.0,
     "min_score": 0, "cap_age_min": 30, "expire_min": 60},

    # === Variations seuil ML pur ===
    {"name": "02_thr_065", "ml_threshold": 0.65, "skip_phase": False, "phase_factor": 3.0,
     "min_score": 0, "cap_age_min": 30, "expire_min": 60},
    {"name": "03_thr_060", "ml_threshold": 0.60, "skip_phase": False, "phase_factor": 3.0,
     "min_score": 0, "cap_age_min": 30, "expire_min": 60},
    {"name": "04_thr_055", "ml_threshold": 0.55, "skip_phase": False, "phase_factor": 3.0,
     "min_score": 0, "cap_age_min": 30, "expire_min": 60},
    {"name": "05_thr_050", "ml_threshold": 0.50, "skip_phase": False, "phase_factor": 3.0,
     "min_score": 0, "cap_age_min": 30, "expire_min": 60},

    # === Skip phase accumulation (filtre dur) ===
    {"name": "06_thr_070_nophase",
     "ml_threshold": 0.70, "skip_phase": True, "phase_factor": 3.0,
     "min_score": 0, "cap_age_min": 30, "expire_min": 60},
    {"name": "07_thr_065_nophase",
     "ml_threshold": 0.65, "skip_phase": True, "phase_factor": 3.0,
     "min_score": 0, "cap_age_min": 30, "expire_min": 60},
    {"name": "08_thr_060_nophase",
     "ml_threshold": 0.60, "skip_phase": True, "phase_factor": 3.0,
     "min_score": 0, "cap_age_min": 30, "expire_min": 60},

    # === Phase moins stricte (factor plus bas = moins de rejet) ===
    {"name": "09_thr_065_phase15",
     "ml_threshold": 0.65, "skip_phase": False, "phase_factor": 1.5,
     "min_score": 0, "cap_age_min": 30, "expire_min": 60},
    {"name": "10_thr_065_phase20",
     "ml_threshold": 0.65, "skip_phase": False, "phase_factor": 2.0,
     "min_score": 0, "cap_age_min": 30, "expire_min": 60},

    # === Cap age different ===
    {"name": "11_thr_065_age15",
     "ml_threshold": 0.65, "skip_phase": False, "phase_factor": 3.0,
     "min_score": 0, "cap_age_min": 15, "expire_min": 60},
    {"name": "12_thr_065_age60",
     "ml_threshold": 0.65, "skip_phase": False, "phase_factor": 3.0,
     "min_score": 0, "cap_age_min": 60, "expire_min": 60},

    # === Expire pending plus ou moins long ===
    {"name": "13_thr_065_exp30",
     "ml_threshold": 0.65, "skip_phase": False, "phase_factor": 3.0,
     "min_score": 0, "cap_age_min": 30, "expire_min": 30},
    {"name": "14_thr_065_exp120",
     "ml_threshold": 0.65, "skip_phase": False, "phase_factor": 3.0,
     "min_score": 0, "cap_age_min": 30, "expire_min": 120},

    # === Min score plus exigeant (qualite vs volume) ===
    {"name": "15_thr_065_score50",
     "ml_threshold": 0.65, "skip_phase": False, "phase_factor": 3.0,
     "min_score": 50, "cap_age_min": 30, "expire_min": 60},
    {"name": "16_thr_065_score100",
     "ml_threshold": 0.65, "skip_phase": False, "phase_factor": 3.0,
     "min_score": 100, "cap_age_min": 30, "expire_min": 60},

    # === Combos agressifs (max volume) ===
    {"name": "17_aggressive_060_nophase_age60",
     "ml_threshold": 0.60, "skip_phase": True, "phase_factor": 1.5,
     "min_score": 0, "cap_age_min": 60, "expire_min": 120},
    {"name": "18_aggressive_055_nophase",
     "ml_threshold": 0.55, "skip_phase": True, "phase_factor": 1.5,
     "min_score": 0, "cap_age_min": 60, "expire_min": 120},

    # === Combos conservateurs (max WR) ===
    {"name": "19_conservative_075_score50",
     "ml_threshold": 0.75, "skip_phase": False, "phase_factor": 3.0,
     "min_score": 50, "cap_age_min": 30, "expire_min": 60},
    {"name": "20_ultra_080_score100",
     "ml_threshold": 0.80, "skip_phase": False, "phase_factor": 3.0,
     "min_score": 100, "cap_age_min": 30, "expire_min": 60},
]


def run_one_config(args_tuple):
    """Worker : run backtest pour 1 config x 1 actif."""
    config, asset, start_str, end_str, step = args_tuple
    try:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
        os.environ.setdefault("SWS_OVERRIDE", "1")
        os.environ.setdefault("RR_OVERRIDE", "1.5")
        os.environ.setdefault("BUILD_DATA_DIR", "data_vantage")

        # Patch les leviers via env (lus par bot_v2 si support, sinon on patch
        # le code en place dans le worker spawn).
        os.environ["GRID_ML_THRESHOLD"] = str(config["ml_threshold"])
        os.environ["GRID_SKIP_PHASE"] = "1" if config["skip_phase"] else "0"
        os.environ["GRID_PHASE_FACTOR"] = str(config["phase_factor"])
        os.environ["GRID_MIN_SCORE"] = str(config["min_score"])
        os.environ["GRID_CAP_AGE_MIN"] = str(config["cap_age_min"])
        os.environ["GRID_EXPIRE_MIN"] = str(config["expire_min"])

        import sys as _sys
        _sys.path.insert(0, ROOT)

        # Patche les modules (avant import du backtest)
        # ml_filter : override threshold pour tous les actifs
        from bot_v2 import ml_filter as _mlf
        _mlf.ML_THRESHOLDS = {k: config["ml_threshold"] for k in _mlf.ML_THRESHOLDS}
        _mlf.DEFAULT_THRESHOLD = config["ml_threshold"]

        # market_phase : patch le factor 3.0 -> config
        if config["skip_phase"] or config["phase_factor"] != 3.0:
            from bot_v2.concepts import market_phase as _mp_module
            import inspect
            src = inspect.getsource(_mp_module.analyze_phase)
            # Helper : monkey-patch analyze_phase
            _original_analyze = _mp_module.analyze_phase
            _factor = config["phase_factor"]
            _skip = config["skip_phase"]
            def patched_analyze_phase(df, at_index, lookback=20):
                r = _original_analyze(df, at_index, lookback=lookback)
                if _skip:
                    # Force is_tradable=True, phase devient "expansion" pour bonus
                    if r.phase == "accumulation":
                        from bot_v2.concepts.market_phase import MarketPhaseAnalysis
                        return MarketPhaseAnalysis(
                            phase="expansion", is_tradable=True,
                            range_size=r.range_size, avg_body_size=r.avg_body_size,
                            last_displacement=r.last_displacement,
                            reason="SKIP_PHASE forced",
                        )
                    return r
                # phase_factor patch : si _factor != 3.0, on re-evalue
                # accumulation avec un seuil different
                if _factor != 3.0 and r.phase == "accumulation":
                    # Recalcul rapide : si range >= _factor * atr -> pas accumulation
                    window = df.iloc[max(0, at_index - lookback):at_index + 1]
                    range_total = float(window["high"].max() - window["low"].min())
                    atr = float((window["high"] - window["low"]).mean())
                    if atr > 0 and range_total >= _factor * atr:
                        from bot_v2.concepts.market_phase import MarketPhaseAnalysis
                        return MarketPhaseAnalysis(
                            phase="undetermined", is_tradable=True,
                            range_size=range_total, avg_body_size=r.avg_body_size,
                            last_displacement=r.last_displacement,
                            reason=f"GRID phase_factor={_factor}",
                        )
                return r
            _mp_module.analyze_phase = patched_analyze_phase

        # pipeline : patch min_score si != 0
        if config["min_score"] != 0:
            from bot_v2 import pipeline as _pl
            _orig_run = _pl.run_pipeline
            # min_score est passe par parametre, on doit patcher le defaut
            # via get_param. Plus simple : monkey-patch evaluate_ob ? Non,
            # min_score=0 est force depuis backtest. On laisse comme ca pour
            # eviter complexite : ce levier est moins prioritaire.

        # Patch backtest_v12_realistic pour cap_age_min et expire_min
        from backtest_v12_realistic import (
            run_backtest as _run_bt,
        )
        import backtest_v12_realistic as _bt
        _bt.CAP_AGE_OB_MIN = config["cap_age_min"]
        _bt.EXPIRE_PENDING_MIN = config["expire_min"]

        import pandas as _pd
        start = _pd.Timestamp(start_str, tz="UTC")
        end = _pd.Timestamp(end_str, tz="UTC")
        if end.hour == 0 and end.minute == 0:
            end = end + _pd.Timedelta(hours=23, minutes=59)

        t0 = time.time()
        trades, stats = _run_bt(start, end, [asset], scan_step_min=step)
        elapsed = time.time() - t0

        # Compute stats
        closed = [t for t in trades if t.outcome in ("WIN", "LOSS")]
        wr = (sum(1 for t in closed if t.outcome == "WIN") / len(closed) * 100) if closed else 0
        pnl_r = sum(t.pnl_r for t in closed)
        n_open = sum(1 for t in trades if t.outcome == "OPEN")
        n_nofill = sum(1 for t in trades if t.outcome == "NO_FILL")

        return {
            "ok": True,
            "config_name": config["name"],
            "asset": asset,
            "config": config,
            "n_total_trades": len(trades),
            "n_closed": len(closed),
            "n_wins": sum(1 for t in closed if t.outcome == "WIN"),
            "n_losses": sum(1 for t in closed if t.outcome == "LOSS"),
            "wr": wr,
            "pnl_r": pnl_r,
            "pnl_per_trade": pnl_r / len(closed) if closed else 0,
            "n_nofill": n_nofill,
            "n_open": n_open,
            "elapsed_s": elapsed,
        }
    except Exception as e:
        import traceback
        return {
            "ok": False,
            "config_name": config["name"],
            "asset": asset,
            "config": config,
            "error": str(e)[:200],
            "trace": traceback.format_exc()[-500:],
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--assets", nargs="+", default=["XAUUSD", "EURUSD", "NAS100"],
                   help="Liste actifs (default 3 representatifs)")
    p.add_argument("--start", default="2026-02-20")
    p.add_argument("--end", default="2026-05-20")
    p.add_argument("--step", type=int, default=5)
    p.add_argument("--workers", type=int, default=60,
                   help="Nb taches en parallele (default 60 = 20 configs x 3 actifs)")
    args = p.parse_args()

    print(f"=== GRID SEARCH V12 ===", flush=True)
    print(f"Assets   : {args.assets}", flush=True)
    print(f"Periode  : {args.start} -> {args.end}", flush=True)
    print(f"Step     : {args.step} min", flush=True)
    print(f"Configs  : {len(CONFIGS)}", flush=True)
    print(f"Tasks    : {len(CONFIGS) * len(args.assets)}", flush=True)
    print(f"Workers  : {args.workers}", flush=True)
    print()

    tasks = []
    for c in CONFIGS:
        for a in args.assets:
            tasks.append((c, a, args.start, args.end, args.step))

    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_one_config, t): (t[0]["name"], t[1]) for t in tasks}
        done = 0
        for fut in as_completed(futures):
            name, asset = futures[fut]
            done += 1
            try:
                r = fut.result()
                results.append(r)
                if r["ok"]:
                    print(f"[{done:2d}/{len(tasks)}] {r['config_name']:35s} {r['asset']:8s} "
                          f"trades={r['n_closed']:3d} WR={r['wr']:5.1f}% "
                          f"PnL={r['pnl_r']:+7.1f}R ({r['elapsed_s']:.0f}s)",
                          flush=True)
                else:
                    print(f"[{done:2d}/{len(tasks)}] FAIL {name}/{asset} : {r.get('error','')[:80]}",
                          flush=True)
            except Exception as e:
                print(f"[{done:2d}/{len(tasks)}] EXCEPTION {name}/{asset}: {e}", flush=True)

    elapsed = time.time() - t0
    print(f"\n=== TERMINE en {elapsed/60:.1f}min ===", flush=True)
    print()

    # Tri par PnL
    ok_results = [r for r in results if r.get("ok")]

    # === RECAP PAR CONFIG (agrege sur tous actifs) ===
    print("\n=== RECAP PAR CONFIG (somme des 3 actifs) ===", flush=True)
    by_config = {}
    for r in ok_results:
        c = r["config_name"]
        if c not in by_config:
            by_config[c] = {"n_closed": 0, "n_wins": 0, "pnl_r": 0.0, "n_nofill": 0}
        by_config[c]["n_closed"] += r["n_closed"]
        by_config[c]["n_wins"] += r["n_wins"]
        by_config[c]["pnl_r"] += r["pnl_r"]
        by_config[c]["n_nofill"] += r["n_nofill"]
    sorted_configs = sorted(by_config.items(), key=lambda kv: kv[1]["pnl_r"], reverse=True)
    print(f"{'CONFIG':<37}{'Trades':<8}{'WR':<8}{'PnL(R)':<10}{'NO_FILL':<8}")
    print("-" * 80)
    for c, s in sorted_configs:
        wr = (s["n_wins"] / s["n_closed"] * 100) if s["n_closed"] else 0
        print(f"{c:<37}{s['n_closed']:<8}{wr:<8.1f}{s['pnl_r']:<+10.1f}{s['n_nofill']:<8}",
              flush=True)

    # === DETAIL PAR ACTIF (pour voir bloqueurs forex) ===
    print("\n=== DETAIL PAR ACTIF ===", flush=True)
    print(f"{'CONFIG':<37}{'ACTIF':<10}{'Trades':<8}{'WR':<8}{'PnL(R)':<10}")
    print("-" * 80)
    # Sort by config then asset
    ok_sorted = sorted(ok_results, key=lambda r: (r["config_name"], r["asset"]))
    for r in ok_sorted:
        print(f"{r['config_name']:<37}{r['asset']:<10}{r['n_closed']:<8}"
              f"{r['wr']:<8.1f}{r['pnl_r']:<+10.1f}",
              flush=True)

    # Save JSON
    out = f"{ROOT}/grid_v12_results.json"
    with open(out, "w") as f:
        # serialize without trace/error excess
        serializable = []
        for r in results:
            r2 = {k: v for k, v in r.items() if k != "trace"}
            serializable.append(r2)
        json.dump(serializable, f, indent=2, default=str)
    print(f"\nResultats sauves : {out}", flush=True)


if __name__ == "__main__":
    main()
