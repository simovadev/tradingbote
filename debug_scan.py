"""Debug : scanne UNE FOIS chaque actif et affiche ce qui se passe.

Sert a comprendre pourquoi le bot live ne prend pas de trade.
"""
import sys
sys.path.insert(0, 'C:/Users/Administrator/tradingbote')

import pandas as pd
import json
import pickle
from pathlib import Path

from bot_v2 import ml_filter
from bot_v2.concepts.breaker import detect_breakers
from bot_v2.concepts.daily_bias import build_d1_from_h1
from bot_v2.concepts.fvg import detect_fvg
from bot_v2.concepts.htf_swings import collect_htf_swings
from bot_v2.concepts.liquidity import find_swings
from bot_v2.concepts.mss_setup import detect_mss_setups, confirm_ob_with_mss
from bot_v2.concepts.order_block import detect_order_blocks
from bot_v2.concepts.structure import detect_structure_breaks, detect_trend
from bot_v2.config import SMT_PAIRS, get_param
from bot_v2.mt5_executor import MT5Executor
from bot_v2.pipeline import evaluate_ob


ASSETS = ["XAUUSD", "NAS100", "GER40", "BTCUSD", "EURUSD", "GBPUSD", "AUDUSD", "USDJPY"]


def load_model(instrument):
    base = Path("C:/Users/Administrator/tradingbote/bot_v2")
    model = pickle.load(open(base / f"ml_model_{instrument}.pkl", "rb"))
    features = json.load(open(base / f"ml_features_{instrument}.json"))["features"]
    return model, features


def main():
    print("=== DEBUG SCAN — Voir pourquoi 0 setup ===\n")

    mt5e = MT5Executor()
    if not mt5e.initialize():
        print("MT5 init fail")
        return

    for asset in ASSETS:
        print(f"\n--- {asset} ---")
        df_m1 = mt5e.get_bars(asset, "M1", 2000)
        df_m15 = mt5e.get_bars(asset, "M15", 500)
        df_h1 = mt5e.get_bars(asset, "H1", 500)
        try:
            df_d1 = mt5e.get_bars(asset, "D1", 100)
            if df_d1 is None or len(df_d1) < 10:
                raise Exception
        except Exception:
            df_d1 = build_d1_from_h1(df_h1)

        if df_m1 is None or len(df_m1) < 200:
            print(f"  ❌ M1 data insuffisante : {len(df_m1) if df_m1 is not None else 0} bougies")
            continue

        last_ts = df_m1.index[-1]
        print(f"  Derniere bougie M1 : {last_ts}")

        # Detection OB
        sws = get_param(asset, "swing_strength_m1", 2)
        obs = detect_order_blocks(df_m1, swing_strength=sws, max_group_size=2)
        swings = find_swings(df_m1, strength=sws)
        structure_breaks = detect_structure_breaks(df_m1, swings=swings)
        mss_setups = detect_mss_setups(df_m1, structure_breaks=structure_breaks, swings=swings)
        obs_confirmed = confirm_ob_with_mss(obs, mss_setups, window_bars=10)

        print(f"  OB detectes : {len(obs)}, MSS : {len(mss_setups)}, OB+MSS confirmes : {len(obs_confirmed)}")

        # OB des X dernieres minutes (teste 10, 30, 60, 120 min)
        for minutes in [10, 30, 60, 120]:
            cutoff = last_ts - pd.Timedelta(minutes=minutes)
            recent = [ob for ob in obs_confirmed if df_m1.index[ob.validation_index] >= cutoff]
            print(f"  OB+MSS recents (<{minutes}min) : {len(recent)}")

        # Garde la fenetre 60min pour la suite (nouveau cutoff bot)
        cutoff = last_ts - pd.Timedelta(minutes=60)
        recent = [ob for ob in obs_confirmed if df_m1.index[ob.validation_index] >= cutoff]

        # Si on a des recents, voir pourquoi rejetes
        if recent:
            try:
                model, features = load_model(asset)
            except Exception as e:
                print(f"  ❌ Model load fail : {e}")
                continue

            threshold = ml_filter.ML_THRESHOLDS.get(asset, 0.55)

            htf_dfs = {"H1": df_h1, "D1": df_d1}
            try:
                df_h4 = mt5e.get_bars(asset, "H4", 500)
                if df_h4 is not None:
                    htf_dfs["H4"] = df_h4
            except Exception:
                pass
            htf_swings = collect_htf_swings(htf_dfs, swing_strength=3)

            correlated_dfs = {}
            for corr_name, corr_type in SMT_PAIRS.get(asset, []):
                try:
                    df_c = mt5e.get_bars(corr_name, "M1", 2000)
                    if df_c is not None:
                        correlated_dfs[corr_name] = (df_c, corr_type)
                except Exception:
                    continue

            cache = {
                "swings_ltf": swings,
                "fvgs_ltf": detect_fvg(df_m1),
                "breakers_ltf": detect_breakers(df_m1),
                "obs_htf": detect_order_blocks(df_m15),
                "structure_breaks": structure_breaks,
                "htf_trend": detect_trend(swings, lookback=6),
                "obs_htf2": detect_order_blocks(df_h1),
            }

            for ob in recent:
                ts = df_m1.index[ob.validation_index]
                try:
                    r = evaluate_ob(
                        ob, df_m1, df_m15, df_d1, asset,
                        ltf_name="M1", htf_name="M15",
                        df_htf2=df_h1, htf2_name="H1",
                        correlated_dfs=correlated_dfs,
                        htf_swings=htf_swings,
                        df_h1=df_h1, min_score=0, min_quality=0,
                        cache=cache,
                    )
                except Exception as e:
                    print(f"    OB {ts} {ob.direction} : evaluate_ob ERROR {e}")
                    continue

                if r.verdict != "TRADE":
                    print(f"    OB {ts} {ob.direction} : REJET ({r.rejection_reason or 'no_trade'})")
                    continue

                if r.trade_setup is None:
                    print(f"    OB {ts} {ob.direction} : TRADE_SETUP None")
                    continue

                # ML prediction
                feats = ml_filter._features_from_result(r, ob, asset)
                import pandas as pd_local
                X = pd_local.DataFrame([[feats.get(f, 0) for f in features]], columns=features)
                proba = float(model.predict_proba(X)[0, 1])
                status = "✅ TRADE" if proba >= threshold else f"❌ ML {proba:.3f} < {threshold}"
                print(f"    OB {ts} {ob.direction} score={r.score} ML={proba:.3f} | {status}")

    mt5e.shutdown()


if __name__ == "__main__":
    main()
