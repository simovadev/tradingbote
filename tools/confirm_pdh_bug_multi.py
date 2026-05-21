"""Test du data leakage PDH/PDL sur plusieurs setups + actifs."""
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bot_v2.config import SMT_PAIRS, get_param
from bot_v2.concepts.htf_swings import collect_htf_swings
from bot_v2.concepts.liquidity import find_swings
from bot_v2.concepts.fvg import detect_fvg
from bot_v2.concepts.breaker import detect_breakers
from bot_v2.concepts.order_block import detect_order_blocks
from bot_v2.concepts.structure import detect_structure_breaks, detect_trend
from bot_v2.concepts.mss_setup import detect_mss_setups
from bot_v2.pipeline import evaluate_ob
from bot_v2 import ml_filter
from bot_v2.live_runner_v2 import load_model, N_BARS_M1, N_BARS_M15, N_BARS_H1, N_BARS_D1


def test_asset(asset, parquet_dir):
    df = pd.read_parquet(parquet_dir / f"{asset}_M1.parquet")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df_m1 = df.tail(N_BARS_M1).copy()

    def resample(rule, n):
        out = df_m1.resample(rule, label="left", closed="left").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna()
        return out.tail(n)

    df_m15 = resample("15min", N_BARS_M15)
    df_h1 = resample("1h", N_BARS_H1)
    df_h4 = resample("4h", 500)
    df_d1_full = resample("1D", N_BARS_D1)

    htf_swings = collect_htf_swings({"H1": df_h1, "H4": df_h4, "D1": df_d1_full}, swing_strength=3)
    correlated_dfs = {}
    for corr_name, corr_type in SMT_PAIRS.get(asset, []):
        try:
            df_c = pd.read_parquet(parquet_dir / f"{corr_name}_M1.parquet")
            if df_c.index.tz is None:
                df_c.index = df_c.index.tz_localize("UTC")
            correlated_dfs[corr_name] = (df_c.tail(N_BARS_M1).copy(), corr_type)
        except Exception:
            pass

    sws = get_param(asset, "swing_strength_m1", 2)
    obs = detect_order_blocks(df_m1, swing_strength=sws)
    cache = {
        "swings_ltf": find_swings(df_m1, strength=sws),
        "fvgs_ltf": detect_fvg(df_m1),
        "breakers_ltf": detect_breakers(df_m1),
        "obs_htf": detect_order_blocks(df_m15),
    }
    cache["structure_breaks"] = detect_structure_breaks(
        df_m1, swings=cache["swings_ltf"], fvgs=cache["fvgs_ltf"]
    )
    cache["htf_trend"] = detect_trend(cache["swings_ltf"], lookback=6)
    cache["obs_htf2"] = detect_order_blocks(df_h1)
    mss_setups = detect_mss_setups(
        df_m1, structure_breaks=cache["structure_breaks"], swings=cache["swings_ltf"],
        fvgs=cache["fvgs_ltf"],
    )

    # Setups dernière 6h
    cutoff = df_m1.index[-1] - pd.Timedelta(hours=6)
    obs_recent = [ob for ob in obs if df_m1.index[ob.validation_index] >= cutoff]

    loaded = load_model(asset)
    if loaded is None:
        return []
    model, features = loaded

    results = []
    for ob in obs_recent[-10:]:
        ts_val = df_m1.index[ob.validation_index]
        today_start = ts_val.normalize()
        df_d1_buggy = df_d1_full
        df_d1_live = df_d1_full[df_d1_full.index < today_start].copy()
        today_bars = df_m1[(df_m1.index >= today_start) & (df_m1.index <= ts_val)]
        if len(today_bars) > 0:
            today_row = pd.DataFrame({
                "open": [today_bars.iloc[0]["open"]],
                "high": [today_bars["high"].max()],
                "low":  [today_bars["low"].min()],
                "close": [today_bars.iloc[-1]["close"]],
                "volume": [today_bars["volume"].sum()],
            }, index=[today_start])
            df_d1_live = pd.concat([df_d1_live, today_row])

        probas = {}
        for label, df_d1_used in [("bug", df_d1_buggy), ("live", df_d1_live)]:
            try:
                r = evaluate_ob(ob, df_m1, df_m15, df_d1_used, asset,
                    ltf_name="M1", htf_name="M15",
                    df_htf2=df_h1, htf2_name="H1",
                    correlated_dfs=correlated_dfs, htf_swings=htf_swings,
                    df_h1=df_h1, min_score=0, min_quality=0, cache=cache)
                if r.verdict != "TRADE" or r.trade_setup is None:
                    probas[label] = None
                    continue
                feat = ml_filter._features_from_result(r, ob, asset, df_ltf=df_m1,
                    df_d1=df_d1_used, mss_setups=mss_setups)
                X = pd.DataFrame([[feat.get(f, 0) for f in features]], columns=features)
                probas[label] = float(model.predict_proba(X)[0, 1])
            except Exception:
                probas[label] = None
        if probas.get("bug") is not None and probas.get("live") is not None:
            delta = probas["bug"] - probas["live"]
            results.append({
                "asset": asset, "ts": ts_val, "dir": ob.direction,
                "bug": probas["bug"], "live": probas["live"], "delta": delta
            })
    return results


def main():
    parquet_dir = ROOT / "data_vantage"
    print(f"=== TEST DATA LEAKAGE PDH/PDL — 6 actifs, derniers 6h setups ===\n")
    all_results = []
    for asset in ["DJ30", "NAS100", "SP500", "GER40", "XAUUSD", "BTCUSD"]:
        try:
            results = test_asset(asset, parquet_dir)
            all_results.extend(results)
        except Exception as e:
            print(f"  {asset} : ERROR {e}")

    print(f"{'Asset':<8} {'TS':<20} {'Dir':<8} {'BUG':<8} {'LIVE':<8} {'Delta'}")
    print("-" * 60)
    for r in sorted(all_results, key=lambda r: -abs(r["delta"])):
        print(f"{r['asset']:<8} {str(r['ts'])[:19]:<20} {r['dir']:<8} {r['bug']:.3f}   {r['live']:.3f}   {r['delta']:+.3f}")

    if all_results:
        deltas = [r["delta"] for r in all_results]
        print(f"\nStats : N={len(deltas)}, mean delta={sum(deltas)/len(deltas):+.3f}")
        print(f"  max +delta = {max(deltas):+.3f}")
        print(f"  max -delta = {min(deltas):+.3f}")
        # Bug = MORE probas dans le training => si delta positif global, le bug AUGMENTE artificiellement les probas
        n_pos = sum(1 for d in deltas if d > 0.05)
        n_neg = sum(1 for d in deltas if d < -0.05)
        print(f"  cases ou BUG > LIVE de +0.05 : {n_pos} (training surestime)")
        print(f"  cases ou LIVE > BUG de +0.05 : {n_neg} (training sous-estime)")


if __name__ == "__main__":
    main()
