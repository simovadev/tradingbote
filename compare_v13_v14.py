"""Compare V13 (BOS) vs V14 (mitigation) : detecte les OB sur 1 mois et simule
un trade MARKET au moment de la validation, avec SL/TP RR=2 sur le prix d'entry.

But : verifier si V14 (mitigation) detecte des OB MEILLEURS (plus rapides + WR superieur)
que V13 (BOS) avant de lancer une re-extraction dataset complete.

Usage :
    python compare_v13_v14.py --asset XAUUSD --days 30
    python compare_v13_v14.py --asset XAUUSD --days 90 --all

Output : tableau de stats (WR, RR, latence) pour chaque mode.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from bot_v2.concepts.order_block import detect_order_blocks
from bot_v2.concepts.liquidity import find_swings, find_sweeps


def simulate_trade(
    df: pd.DataFrame,
    ob,
    rr_target: float = 2.0,
    max_bars: int = 240,
) -> dict:
    """Simule un trade MARKET au moment de la validation_index de l'OB.

    Args:
        df: DataFrame M1
        ob: OrderBlock
        rr_target: ratio reward/risk (TP / SL distance)
        max_bars: nb max de bougies pour atteindre TP/SL (sinon TIMEOUT)

    Returns:
        dict avec :
            - outcome : "WIN" / "LOSS" / "TIMEOUT"
            - entry_price : prix d'entry MARKET (close de la bougie validation)
            - sl_price, tp_price
            - bars_to_outcome : nb bougies avant resultat
            - bars_latency : nb bougies entre group_end et validation_index
    """
    val_idx = ob.validation_index
    if val_idx + 1 >= len(df):
        return {"outcome": "NO_FUTURE", "bars_latency": val_idx - ob.group_end_index}

    # Entry MARKET au close de la bougie de validation
    entry = float(df["close"].iloc[val_idx])

    # SL = extreme du group (mecha)
    if ob.direction == "bullish":
        sl = ob.ob_low
        risk = entry - sl
        if risk <= 0:
            return {"outcome": "INVALID_SL", "bars_latency": val_idx - ob.group_end_index}
        tp = entry + risk * rr_target
    else:
        sl = ob.ob_high
        risk = sl - entry
        if risk <= 0:
            return {"outcome": "INVALID_SL", "bars_latency": val_idx - ob.group_end_index}
        tp = entry - risk * rr_target

    # Simule sur les bougies suivantes : 1ere a toucher SL ou TP gagne
    sim_end = min(val_idx + 1 + max_bars, len(df))
    outcome = "TIMEOUT"
    bars_to_outcome = max_bars
    for k in range(val_idx + 1, sim_end):
        bar_high = float(df["high"].iloc[k])
        bar_low = float(df["low"].iloc[k])
        if ob.direction == "bullish":
            # SL touche en premier si low <= sl, TP touche si high >= tp
            sl_hit = bar_low <= sl
            tp_hit = bar_high >= tp
        else:
            sl_hit = bar_high >= sl
            tp_hit = bar_low <= tp
        # En cas de meme bougie, on suppose SL d'abord (conservateur)
        if sl_hit and tp_hit:
            outcome = "LOSS"
            bars_to_outcome = k - val_idx
            break
        if sl_hit:
            outcome = "LOSS"
            bars_to_outcome = k - val_idx
            break
        if tp_hit:
            outcome = "WIN"
            bars_to_outcome = k - val_idx
            break

    return {
        "outcome": outcome,
        "entry_price": entry,
        "sl_price": sl,
        "tp_price": tp,
        "bars_to_outcome": bars_to_outcome,
        "bars_latency": val_idx - ob.group_end_index,
        "direction": ob.direction,
        "ob_ts": ob.group_end_ts,
        "val_ts": ob.validation_ts,
    }


def run_mode(df: pd.DataFrame, swings, sweeps, mode: str) -> pd.DataFrame:
    """Detecte les OB en mode donne + simule chaque trade.

    Returns DataFrame des resultats.
    """
    obs = detect_order_blocks(
        df,
        swings=swings,
        sweeps=sweeps,
        swing_strength=1,         # V11+ aligne live
        max_group_size=5,
        min_group_size=1,
        max_bars_after_sweep=30,
        validation_mode=mode,
    )
    print(f"[{mode}] {len(obs)} OB detectes")

    results = []
    for ob in obs:
        r = simulate_trade(df, ob)
        results.append(r)
    return pd.DataFrame(results)


def stats_summary(df_res: pd.DataFrame, mode: str) -> dict:
    """Resume des stats d'un mode."""
    if len(df_res) == 0:
        return {"mode": mode, "n": 0}
    valid = df_res[df_res["outcome"].isin(["WIN", "LOSS"])]
    wins = (valid["outcome"] == "WIN").sum()
    losses = (valid["outcome"] == "LOSS").sum()
    n = len(valid)
    wr = wins / n * 100 if n > 0 else 0.0
    expectancy = (wins * 2 - losses) / n if n > 0 else 0.0  # RR=2 -> WIN=+2, LOSS=-1
    return {
        "mode": mode,
        "n_total": len(df_res),
        "n_decisive": n,
        "wins": int(wins),
        "losses": int(losses),
        "timeouts": int((df_res["outcome"] == "TIMEOUT").sum()),
        "invalid": int(df_res["outcome"].isin(["NO_FUTURE", "INVALID_SL"]).sum()),
        "wr_pct": round(wr, 1),
        "expectancy_R": round(expectancy, 2),
        "bars_latency_mean": round(df_res["bars_latency"].mean(), 1) if "bars_latency" in df_res.columns else None,
        "bars_latency_median": int(df_res["bars_latency"].median()) if "bars_latency" in df_res.columns else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="XAUUSD")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--all", action="store_true", help="Tester sur tous les 14 actifs")
    args = ap.parse_args()

    assets = ["XAUUSD", "NAS100", "GER40", "BTCUSD", "EURUSD", "GBPUSD",
              "AUDUSD", "USDJPY", "SP500", "DJ30", "UK100", "FRA40",
              "USDCAD", "USDCHF"] if args.all else [args.asset]

    all_stats = []
    for asset in assets:
        parquet = ROOT / "data" / "cache" / f"{asset}_M1.parquet"
        if not parquet.exists():
            print(f"!! {asset} : pas de parquet, skip")
            continue

        df = pd.read_parquet(parquet)
        # Garde les N derniers jours
        cutoff = df.index[-1] - pd.Timedelta(days=args.days)
        df = df[df.index >= cutoff].copy()
        print(f"\n=== {asset} : {len(df)} bougies M1 ({df.index[0]} -> {df.index[-1]}) ===")

        # Pre-calc swings + sweeps (utilise par les 2 modes)
        swings = find_swings(df, strength=1)
        sweeps = find_sweeps(df, swings, min_depth_atr=0.0)
        print(f"  {len(swings)} swings, {len(sweeps)} sweeps")

        for mode in ["bos", "mitigation"]:
            res = run_mode(df, swings, sweeps, mode)
            stats = stats_summary(res, mode)
            stats["asset"] = asset
            all_stats.append(stats)
            print(f"  [{mode:<11}] n={stats['n_decisive']:<4} "
                  f"WR={stats['wr_pct']:.1f}%  "
                  f"expectancy={stats['expectancy_R']:+.2f}R  "
                  f"latency_med={stats['bars_latency_median']}")

    print("\n=== RECAP ===")
    df_stats = pd.DataFrame(all_stats)
    print(df_stats.to_string(index=False))

    # Sauvegarde
    out = ROOT / f"compare_v13_v14_{'all' if args.all else args.asset}_{args.days}d.csv"
    df_stats.to_csv(out, index=False)
    print(f"\nSauve : {out}")


if __name__ == "__main__":
    main()
