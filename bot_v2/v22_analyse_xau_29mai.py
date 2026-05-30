"""Analyse pourquoi XAUUSD a 0 trade le 29/05/2026 malgre 63 OBs detectes.

On veut savoir :
- Quel etait le contexte D1 / H1 / PD daily
- Pourquoi chacun des 63 OBs a ete rejete
- Y a-t-il un moment ou ca aurait DU trader ?
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

from bot_v2.concepts.safe import (
    find_obs_simple, daily_bias_safe, htf_value_at_t, atr_safe,
)

DATA = ROOT / "data_vantage"
ASSET = "XAUUSD"
TARGET_DAY = pd.Timestamp("2026-05-29", tz="UTC")


def to_fr(ts): return ts.tz_convert("Europe/Paris").strftime("%H:%M")


def main():
    print("=" * 90)
    print(f"ANALYSE POURQUOI XAUUSD 0 TRADE LE 29/05/2026")
    print("=" * 90)

    df_m5 = pd.read_parquet(DATA / f"{ASSET}_M5.parquet")[["open","high","low","close"]]
    df_h1 = pd.read_parquet(DATA / f"{ASSET}_H1.parquet")[["open","high","low","close"]]
    df_d1 = pd.read_parquet(DATA / f"{ASSET}_D1.parquet")[["open","high","low","close"]]

    ctx_start = TARGET_DAY - pd.Timedelta(days=5)
    day_end = TARGET_DAY + pd.Timedelta(days=1)
    df_m5 = df_m5[(df_m5.index >= ctx_start) & (df_m5.index < day_end)]
    df_h1 = df_h1[(df_h1.index >= ctx_start) & (df_h1.index < day_end)]
    df_d1 = df_d1[(df_d1.index >= TARGET_DAY - pd.Timedelta(days=30)) & (df_d1.index < day_end)]

    # ====== Contexte D1 ======
    print(f"\n=== CONTEXTE DAILY (29/05/2026) ===\n")
    target_day = TARGET_DAY.normalize()
    df_d1_past = df_d1[df_d1.index < target_day]
    if len(df_d1_past) < 2:
        print("Pas assez de data D1"); return
    pdh = float(df_d1_past["high"].iloc[-1])
    pdl = float(df_d1_past["low"].iloc[-1])
    pd_close = float(df_d1_past["close"].iloc[-1])
    pd2_close = float(df_d1_past["close"].iloc[-2])
    mid_daily = (pdh + pdl) / 2
    d1_bias_str = "haussier" if pd_close > pd2_close else "baissier"
    move_d1 = (pd_close - pd2_close) / pd2_close * 100
    print(f"Hier (J-1 = 28/05) :  PDH={pdh:.2f}  PDL={pdl:.2f}  Close={pd_close:.2f}")
    print(f"Avant-hier (J-2 = 27/05) :  Close={pd2_close:.2f}")
    print(f"MID daily : {mid_daily:.2f}")
    print(f"D1 bias : {d1_bias_str} (move {move_d1:+.2f}%)")

    # ====== Trajectoire H1 sur la journee ======
    print(f"\n=== TRAJECTOIRE H1 (29/05/2026) ===\n")
    h1_day = df_h1[(df_h1.index >= TARGET_DAY) & (df_h1.index < day_end)]
    print(f"Nb bougies H1 ce jour : {len(h1_day)}")
    if len(h1_day) > 0:
        print(f"Open H1 du jour : {h1_day['open'].iloc[0]:.2f}")
        print(f"Close H1 dernier : {h1_day['close'].iloc[-1]:.2f}")
        print(f"High du jour : {h1_day['high'].max():.2f}")
        print(f"Low du jour : {h1_day['low'].min():.2f}")
        # Range du jour
        range_pct = (h1_day['high'].max() - h1_day['low'].min()) / h1_day['open'].iloc[0] * 100
        print(f"Range : {range_pct:.2f}% du prix")

    # ====== H1 trend au fil du jour ======
    print(f"\n=== H1 TREND AU COURS DU JOUR (close H1 vs close H1 il y a 10h) ===\n")
    print(f"{'heure_h1':<10} {'close':<10} {'close-10h':<11} {'momentum%':<11} {'trend'}")
    print("-" * 60)
    for ts in h1_day.index[::4]:   # toutes les 4h pour pas spammer
        h1_close = htf_value_at_t(df_h1, ts + pd.Timedelta(seconds=1), "close")
        pos = df_h1.index.searchsorted(ts + pd.Timedelta(seconds=1), side="left") - 1 - 10
        if pos >= 0 and h1_close is not None:
            h1_old = float(df_h1["close"].iloc[pos])
            mom = (h1_close - h1_old) / h1_old * 100
            trend = "haussier" if mom > 0 else "baissier"
            print(f"{to_fr(ts):<10} {h1_close:<10.2f} {h1_old:<11.2f} {mom:<+11.2f} {trend}")

    # ====== Trajectoire M5 ======
    m5_day = df_m5[(df_m5.index >= TARGET_DAY) & (df_m5.index < day_end)]
    open_day = m5_day["open"].iloc[0]
    print(f"\n=== TRAJECTOIRE M5 ===\n")
    print(f"Open jour : {open_day:.2f}")
    print(f"High jour : {m5_day['high'].max():.2f} (mid {mid_daily:.2f})")
    print(f"Low jour : {m5_day['low'].min():.2f} (mid {mid_daily:.2f})")
    print(f"Prix vs mid : {'au-dessus' if open_day > mid_daily else 'en-dessous'} du mid daily")

    # ====== Detect tous les OBs et voir pourquoi rejetes ======
    print(f"\n=== ANALYSE DES 63 OBs (pourquoi rejetes ?) ===\n")
    obs = find_obs_simple(df_m5, as_of_index=len(df_m5) - 1)
    obs_day = [ob for ob in obs if TARGET_DAY <= ob.validation_ts < day_end]
    print(f"Total OBs detectes : {len(obs_day)}")
    bullish_obs = [ob for ob in obs_day if ob.direction == "bullish"]
    bearish_obs = [ob for ob in obs_day if ob.direction == "bearish"]
    print(f"  Bullish : {len(bullish_obs)}")
    print(f"  Bearish : {len(bearish_obs)}")

    # Categories de rejet
    cat = {"no_d1": 0, "no_h1": 0, "no_pd": 0, "no_atr": 0, "passes_all": 0}
    rejet_details = []
    for ob in obs_day:
        val_ts = ob.validation_ts; val_idx = ob.validation_index
        # D1
        d1 = daily_bias_safe(df_d1, val_ts)
        if not d1["ok"]:
            cat["no_d1"] += 1; rejet_details.append((ob, "no_d1_bias"))
            continue
        d1_h = d1["bias"] == "haussier"
        d1_a = (ob.direction == "bullish" and d1_h) or (ob.direction == "bearish" and not d1_h)
        if not d1_a:
            cat["no_d1"] += 1
            rejet_details.append((ob, f"D1={d1['bias']} vs trade={ob.direction}"))
            continue
        # H1
        h1_close = htf_value_at_t(df_h1, val_ts, "close")
        h1_a = False
        h1_trend = None
        if h1_close is not None:
            pos = df_h1.index.searchsorted(val_ts, side="left") - 1 - 10
            if pos >= 0:
                h1_old = float(df_h1["close"].iloc[pos])
                h1_h = h1_close > h1_old
                h1_trend = "haussier" if h1_h else "baissier"
                h1_a = (ob.direction == "bullish" and h1_h) or (ob.direction == "bearish" and not h1_h)
        if not h1_a:
            cat["no_h1"] += 1
            rejet_details.append((ob, f"H1={h1_trend} vs trade={ob.direction}"))
            continue
        # PD daily
        target_day_v = pd.Timestamp(val_ts).normalize()
        df_d1_past_v = df_d1[df_d1.index < target_day_v]
        if len(df_d1_past_v) < 1:
            cat["no_pd"] += 1; continue
        pdh_v = float(df_d1_past_v["high"].iloc[-1])
        pdl_v = float(df_d1_past_v["low"].iloc[-1])
        mid_v = (pdh_v + pdl_v) / 2
        price = float(df_m5["close"].iloc[val_idx])
        pd_zone = "discount" if price < mid_v else "premium"
        pd_a = (ob.direction == "bullish" and price < mid_v) or (ob.direction == "bearish" and price > mid_v)
        if not pd_a:
            cat["no_pd"] += 1
            rejet_details.append((ob, f"PD={pd_zone} vs trade={ob.direction} @ price={price:.2f} vs mid={mid_v:.2f}"))
            continue
        # ATR
        atr = atr_safe(df_m5, ob.validation_index)
        if atr <= 0:
            cat["no_atr"] += 1; continue
        cat["passes_all"] += 1
        rejet_details.append((ob, "PASSE !"))

    print(f"\nCategories de rejet :")
    print(f"  Reject D1 bias  : {cat['no_d1']}")
    print(f"  Reject H1 trend : {cat['no_h1']}")
    print(f"  Reject PD daily : {cat['no_pd']}")
    print(f"  Reject ATR      : {cat['no_atr']}")
    print(f"  PASSE TOUT      : {cat['passes_all']}")

    # Pour comprendre, listons quelques OBs avec leur rejet
    print(f"\n=== ECHANTILLON 15 OBs (premiers + milieu + derniers) ===\n")
    sample = rejet_details[:5] + rejet_details[len(rejet_details)//2-2:len(rejet_details)//2+3] + rejet_details[-5:]
    for ob, reason in sample:
        print(f"  {to_fr(ob.validation_ts)}  {ob.direction:<8} OB[{ob.ob_low:.2f}-{ob.ob_high:.2f}]  -> {reason}")

    # ====== INTERPRETATION ======
    print(f"\n=== INTERPRETATION ===\n")
    if cat['no_d1'] > 30:
        print(f"-> {cat['no_d1']} OBs rejetes parce que D1 bias va dans l'autre sens")
        print(f"   Cad le bot ne veut pas trader contre la tendance daily")
    if cat['no_h1'] > 10:
        print(f"-> {cat['no_h1']} OBs rejetes par H1 trend")
        print(f"   Le H1 a peut-etre fait des changements de direction frequents ce jour")
    if cat['no_pd'] > 10:
        print(f"-> {cat['no_pd']} OBs rejetes par PD daily")
        print(f"   Soit le prix est tout le temps en discount alors que des shorts apparaissent")
        print(f"   Soit en premium alors que des longs apparaissent")


if __name__ == "__main__":
    main()
