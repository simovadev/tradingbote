"""Export les 14 actifs x 6 timeframes depuis MT5 Vantage.

Fetch le MAX de bougies disponibles (Vantage limite a ~3 mois sur M1, plus sur HTF).
Sauvegarde en parquet dans data_vantage/{ASSET}_{TF}.parquet (compatible data_loader).

Usage : python export_vantage_data.py

Apres l'export, on pourra :
- Lancer un backtest V5 OOS sur ces donnees (3 mois Vantage)
- Comparer aux backtests Admiral 8 ans
"""
import os
import sys
import time
from pathlib import Path

import pandas as pd
import MetaTrader5 as mt5

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from bot_v2.config import ALL_TF
from bot_v2.mt5_executor import MT5Executor, to_broker_symbol


# 14 actifs primaires + 3 SMT (XAGUSD/DXY/SPX500 utilises pour XAUUSD/EURUSD/etc.)
ASSETS = [
    "XAUUSD", "NAS100", "GER40", "BTCUSD",
    "EURUSD", "GBPUSD", "AUDUSD", "USDJPY",
    "SP500", "DJ30", "UK100", "FRA40",
    "USDCAD", "USDCHF",
    # SMT (correlations)
    "XAGUSD", "DXY", "SPX500",
]

# Mapping TF -> mt5 constant
TF_MAP = {
    "M1":  mt5.TIMEFRAME_M1,
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1":  mt5.TIMEFRAME_H1,
    "H4":  mt5.TIMEFRAME_H4,
    "D1":  mt5.TIMEFRAME_D1,
}

# Cap par TF (Vantage limite ~3 mois M1)
# Plus le TF est petit, plus on demande de bougies pour avoir ~3 mois minimum
MAX_BARS_PER_TF = {
    "M1":   200_000,   # 200k M1 = ~140 jours = ~4.5 mois (cap Vantage atteint avant)
    "M5":    50_000,   # 50k M5 = ~170 jours
    "M15":   20_000,   # 20k M15 = ~210 jours
    "M30":   10_000,
    "H1":    10_000,   # 10k H1 = ~1.5 an
    "H4":     5_000,
    "D1":     3_000,   # 3k D1 = ~12 ans
}

OUTPUT_DIR = Path(ROOT) / "data_vantage"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_bars(broker_sym, tf_const, n_bars, broker_offset_sec):
    """Fetch les n_bars dernieres bougies. Retourne DataFrame UTC."""
    rates = mt5.copy_rates_from_pos(broker_sym, tf_const, 0, n_bars)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    # Vantage GMT+3 -> UTC : on soustrait l'offset broker
    df["time"] = pd.to_datetime(df["time"] - broker_offset_sec, unit="s", utc=True)
    df = df.set_index("time")
    # Renommer columns pour matcher data_loader format
    df = df.rename(columns={"tick_volume": "volume"})
    df = df[["open", "high", "low", "close", "volume"]]
    return df


def export_asset(mt5_exec, asset):
    """Export les 6 TF d'un asset."""
    broker_sym = to_broker_symbol(asset)
    print(f"\n=== {asset} (broker={broker_sym}) ===", flush=True)

    # Verifie que le symbole existe
    info = mt5.symbol_info(broker_sym)
    if info is None:
        print(f"  KO: symbole {broker_sym} indisponible")
        return {}

    if not info.visible:
        if not mt5.symbol_select(broker_sym, True):
            print(f"  KO: symbol_select fail {broker_sym}")
            return {}

    results = {}
    for tf in ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]:
        tf_const = TF_MAP[tf]
        n_bars = MAX_BARS_PER_TF[tf]
        t0 = time.time()
        df = fetch_bars(broker_sym, tf_const, n_bars, mt5_exec.broker_utc_offset_sec)
        if df is None or len(df) < 10:
            print(f"  {tf:>3}: KO (fetch fail)")
            continue
        out_path = OUTPUT_DIR / f"{asset}_{tf}.parquet"
        df.to_parquet(out_path)
        duration = time.time() - t0
        date_range = f"{df.index[0].date()} -> {df.index[-1].date()}"
        years = (df.index[-1] - df.index[0]).days / 365.25
        print(f"  {tf:>3}: {len(df):>7,} bougies | {date_range} ({years:.2f}y) | {duration:.1f}s | {out_path.name}",
              flush=True)
        results[tf] = len(df)
    return results


def main():
    mt5_exec = MT5Executor()
    mt5_exec.initialize()

    print(f"\nMT5 connecte | balance={mt5_exec.get_balance()} EUR")
    print(f"Broker offset: {mt5_exec.broker_utc_offset_sec}s ({mt5_exec.broker_utc_offset_sec/3600:.0f}h)")
    print(f"Output dir   : {OUTPUT_DIR}")
    print(f"Assets       : {len(ASSETS)} (14 primaires + 3 SMT)")

    t_start = time.time()
    stats = {}
    for asset in ASSETS:
        try:
            stats[asset] = export_asset(mt5_exec, asset)
        except Exception as e:
            print(f"!! FAIL {asset}: {e}")
            continue

    # Recap
    print(f"\n{'='*70}")
    print(f"EXPORT TERMINE en {time.time()-t_start:.0f}s")
    print(f"{'='*70}")
    print(f"\n{'ASSET':<10s} {'M1':>10s} {'M5':>8s} {'M15':>8s} {'M30':>8s} {'H1':>8s} {'H4':>8s} {'D1':>8s}")
    print("-"*70)
    for asset, tfs in stats.items():
        row = f"{asset:<10s} "
        for tf in ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]:
            n = tfs.get(tf, 0)
            row += f"{n:>10,} " if tf == "M1" else f"{n:>8,} "
        print(row)


if __name__ == "__main__":
    main()
