"""Exporte 8 ans d'historique Vantage depuis MT5 -> data_vantage_v7/.

Utilise MT5 Python (deja installe) pour lire les .hcc binaires du
broker Vantage. Plus profond que data_vantage/ actuel (7 mois).

Output :
  data_vantage_v7/<ASSET>_M1.parquet  (8 ans)
  data_vantage_v7/<ASSET>_M15.parquet (8 ans)
  data_vantage_v7/<ASSET>_H1.parquet  (8 ans)
  data_vantage_v7/<ASSET>_H4.parquet  (8 ans)
  data_vantage_v7/<ASSET>_D1.parquet  (8 ans)

Mapping noms : Vantage utilise des suffixes (+) sur certains symboles.
On les enregistre SANS le + dans le nom du fichier pour rester
compatible avec le code du bot (XAUUSD, pas XAUUSD+).

Usage :
    python tools/export_vantage_8ans.py
    python tools/export_vantage_8ans.py XAUUSD       # un seul actif
    python tools/export_vantage_8ans.py --tf M1 H1   # seulement certains TF
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:
    print("ERROR : MetaTrader5 non installe. pip install MetaTrader5")
    sys.exit(1)


# Mapping symbole_bot -> symbole_broker_vantage
# Le bot connait l'actif comme "XAUUSD", Vantage le diffuse sous "XAUUSD+"
SYMBOL_MAP = {
    "XAUUSD":  "XAUUSD+",
    "NAS100":  "NAS100",
    "GER40":   "GER40",
    "BTCUSD":  "BTCUSD",
    "EURUSD":  "EURUSD+",
    "GBPUSD":  "GBPUSD+",
    "AUDUSD":  "AUDUSD+",
    "USDJPY":  "USDJPY+",
    "SP500":   "SP500",
    "DJ30":    "DJ30",
    "UK100":   "UK100",
    "FRA40":   "FRA40",
    "USDCAD":  "USDCAD+",
    "USDCHF":  "USDCHF+",
    # SMT
    "XAGUSD":  "XAGUSD",
    "DXY":     "USDX",     # MT5 Vantage l'appelle USDX
    "SPX500":  "SPX",      # Vantage = "SPX" (cash S&P 500 index)
}

TIMEFRAMES = {
    "M1":  mt5.TIMEFRAME_M1,
    "M15": mt5.TIMEFRAME_M15,
    "H1":  mt5.TIMEFRAME_H1,
    "H4":  mt5.TIMEFRAME_H4,
    "D1":  mt5.TIMEFRAME_D1,
}

# Plage : du 2018-01-01 a aujourd'hui (le broker tronquera ce qu'il n'a pas)
START_DATE = datetime(2018, 1, 1, tzinfo=timezone.utc)
END_DATE = datetime.now(timezone.utc)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data_vantage_v7"


def init_mt5() -> int:
    """Init MT5 + detecte le broker offset UTC.

    Retourne l'offset en secondes (positif = broker en avance sur UTC).
    Vantage = +3h (compense en interne).
    """
    if not mt5.initialize():
        print(f"ERROR mt5.initialize : {mt5.last_error()}")
        sys.exit(1)
    print(f"MT5 OK : version {mt5.version()}")

    # Detect broker UTC offset
    tick = mt5.symbol_info_tick("EURUSD+") or mt5.symbol_info_tick("EURUSD")
    if tick is None:
        print("WARN : pas de tick dispo pour detecter l'offset, on suppose +0h")
        return 0
    utc_now = datetime.now(timezone.utc).timestamp()
    offset_sec = round((tick.time - utc_now) / 3600) * 3600
    print(f"Broker offset UTC : {offset_sec//3600:+d}h ({offset_sec} sec)")
    return offset_sec


def fetch_one(symbol_broker: str, tf_code: int, broker_offset_sec: int,
              start: datetime, end: datetime) -> pd.DataFrame | None:
    """Fetch un actif/TF depuis MT5 via copy_rates_range.

    MT5 utilise des timestamps en broker time. On compense vers UTC reel.
    """
    # MT5 attend datetime sans tz, en broker time. Donc on ajoute l'offset.
    start_broker = start + pd.Timedelta(seconds=broker_offset_sec)
    end_broker = end + pd.Timedelta(seconds=broker_offset_sec)

    rates = mt5.copy_rates_range(
        symbol_broker, tf_code,
        start_broker.replace(tzinfo=None),
        end_broker.replace(tzinfo=None),
    )
    if rates is None or len(rates) == 0:
        err = mt5.last_error()
        print(f"  KO {symbol_broker} : {err}")
        return None

    df = pd.DataFrame(rates)
    # Compenser l'offset : tick.time est en broker time
    df["time"] = pd.to_datetime(df["time"] - broker_offset_sec, unit="s", utc=True)
    df = df.set_index("time").sort_index()
    # Garder les colonnes utiles + standardiser
    df = df.rename(columns={"tick_volume": "volume"})
    keep = ["open", "high", "low", "close", "volume"]
    return df[keep]


def export_one_asset(symbol_bot: str, symbol_broker: str, broker_offset_sec: int,
                      tfs: list[str]) -> None:
    """Exporte tous les TF d'un actif vers data_vantage_v7/."""
    # Verifie que le symbole est dispo
    info = mt5.symbol_info(symbol_broker)
    if info is None:
        # essai avec et sans suffixe
        alt = symbol_broker.rstrip("+") if symbol_broker.endswith("+") else symbol_broker + "+"
        info = mt5.symbol_info(alt)
        if info is None:
            print(f"  SKIP {symbol_bot} : symbole {symbol_broker} introuvable chez Vantage")
            return
        symbol_broker = alt

    # Active dans market watch si non visible
    if not info.visible:
        if not mt5.symbol_select(symbol_broker, True):
            print(f"  KO {symbol_bot} : impossible d'activer {symbol_broker}")
            return

    print(f"\n=== {symbol_bot} (broker: {symbol_broker}) ===")
    for tf_name in tfs:
        tf_code = TIMEFRAMES[tf_name]
        df = fetch_one(symbol_broker, tf_code, broker_offset_sec, START_DATE, END_DATE)
        if df is None or len(df) == 0:
            print(f"  {tf_name} : 0 bougie")
            continue
        out_path = OUTPUT_DIR / f"{symbol_bot}_{tf_name}.parquet"
        df.to_parquet(out_path)
        n_years = (df.index[-1] - df.index[0]).days / 365.25
        print(f"  {tf_name} : {len(df):>9,} bougies ({df.index[0].date()} -> {df.index[-1].date()}, ~{n_years:.1f} ans) -> {out_path.name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("asset", nargs="?", help="Asset specifique (ex: XAUUSD)")
    p.add_argument("--tf", nargs="+", default=["M1", "M15", "H1", "H4", "D1"],
                   help="Timeframes a exporter (defaut: tous)")
    args = p.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output : {OUTPUT_DIR}")

    broker_offset_sec = init_mt5()

    assets_to_do = ([args.asset] if args.asset else list(SYMBOL_MAP.keys()))

    for sym_bot in assets_to_do:
        if sym_bot not in SYMBOL_MAP:
            print(f"!! {sym_bot} pas dans SYMBOL_MAP, skip")
            continue
        sym_broker = SYMBOL_MAP[sym_bot]
        try:
            export_one_asset(sym_bot, sym_broker, broker_offset_sec, args.tf)
        except Exception as e:
            print(f"!! FAIL {sym_bot} : {e}")
            import traceback; traceback.print_exc()

    mt5.shutdown()
    print("\n=== DONE ===")
    print(f"Output : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
