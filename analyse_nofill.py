"""Analyse les 13 NO_FILL du backtest tick 19/05.

Pour chaque NO_FILL : on rejoue les ticks APRES le placement (60 min, fenetre
d'expiration du pending) et on regarde ce que le prix a fait SI on etait entre
en MARKET au moment de la validation OB (au lieu d'attendre le LIMIT).

Verdict par trade :
- "RATE_WIN" : en MARKET on aurait touche le TP -> le LIMIT nous a fait rater un WIN
- "EVITE_LOSS" : en MARKET on aurait touche le SL -> le LIMIT nous a protege
- "NEUTRE" : ni TP ni SL -> le prix a stagne
"""
from __future__ import annotations
import sys, datetime
sys.path.insert(0, "c:/Users/Shadow/TradingBot")
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from bot_v2.mt5_executor import MT5Executor, to_broker_symbol

exe = MT5Executor()
exe.initialize()
off = exe.broker_utc_offset_sec

df = pd.read_csv("c:/Users/Shadow/TradingBot/bt_tick_19may_v2.csv")
nofill = df[df["outcome"] == "NO_FILL"].copy()
print(f"Analyse de {len(nofill)} NO_FILL\n")

EXPIRE_MIN = 60
results = {"RATE_WIN": 0, "EVITE_LOSS": 0, "NEUTRE": 0}

for _, row in nofill.iterrows():
    asset = row["instrument"]
    direction = row["direction"]
    placed = pd.Timestamp(row["placed_ts"])
    entry = float(row["entry"]); sl = float(row["sl"]); tp = float(row["tp"])
    broker = to_broker_symbol(asset)

    # Ticks de placed -> placed+60min
    start = placed.to_pydatetime()
    end = (placed + pd.Timedelta(minutes=EXPIRE_MIN)).to_pydatetime()
    raw = mt5.copy_ticks_range(broker, start + datetime.timedelta(seconds=off),
                               end + datetime.timedelta(seconds=off), mt5.COPY_TICKS_ALL)
    if raw is None or len(raw) == 0:
        print(f"  {asset:8s} {direction:8s} : pas de ticks"); continue
    t = pd.DataFrame(raw)
    bid = t["bid"].values.astype(float)
    ask = t["ask"].values.astype(float)

    # Simulation MARKET : on entre au 1er tick (placed) au prix marche
    # bullish -> achat au ask[0], suit le bid pour SL/TP
    # bearish -> vente au bid[0], suit le ask pour SL/TP
    verdict = "NEUTRE"
    if direction == "bullish":
        entry_mkt = ask[0]
        for i in range(1, len(bid)):
            if bid[i] <= sl:
                verdict = "EVITE_LOSS"; break
            if bid[i] >= tp:
                verdict = "RATE_WIN"; break
    else:
        entry_mkt = bid[0]
        for i in range(1, len(ask)):
            if ask[i] >= sl:
                verdict = "EVITE_LOSS"; break
            if ask[i] <= tp:
                verdict = "RATE_WIN"; break

    results[verdict] += 1
    # distance du prix au moment du placement vs entry
    cur_price = ask[0] if direction == "bullish" else bid[0]
    dist = (cur_price - entry) / entry * 100
    print(f"  {asset:8s} {direction:8s} placed={placed.strftime('%H:%M')} "
          f"entry={entry:.5g} prix_actuel={cur_price:.5g} (dist {dist:+.3f}%) -> {verdict}")

exe.shutdown()
print(f"\n=== VERDICT ===")
print(f"  RATE_WIN   (LIMIT a fait rater un WIN)  : {results['RATE_WIN']}")
print(f"  EVITE_LOSS (LIMIT a evite un LOSS)      : {results['EVITE_LOSS']}")
print(f"  NEUTRE     (prix a stagne)              : {results['NEUTRE']}")
print()
if results['RATE_WIN'] > results['EVITE_LOSS']:
    print("=> Le LIMIT fait RATER plus de WIN qu'il n'evite de LOSS -> entree MARKET serait MEILLEURE")
elif results['EVITE_LOSS'] > results['RATE_WIN']:
    print("=> Le LIMIT EVITE plus de LOSS qu'il ne rate de WIN -> LIMIT est PROTECTEUR, on garde")
else:
    print("=> Egalite -> a creuser sur plus de jours")
