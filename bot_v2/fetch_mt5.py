"""Fetch historique depuis MT5 (Vantage ou autre broker).

Avantages vs Dukascopy :
- Beaucoup plus rapide
- Memes donnees que en live trading -> coherence backtest/live
- Historique illimite (selon le broker)

Prerequis :
- MT5 installe et lance sur le PC
- Compte connecte (demo ou live)
- Symboles affiches dans Market Watch

Usage :
    python -m bot_v2.fetch_mt5 --years 5
    python -m bot_v2.fetch_mt5 --instrument XAUUSD --tf M1 --years 2
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:
    print("ERREUR : pip install MetaTrader5 requis")
    sys.exit(1)

from bot_v2.config import ALL_TF, DATA_DIR, INSTRUMENTS
from bot_v2.data_loader import cache_path


# Mapping interne -> noms Vantage probables (a confirmer apres ouverture MT5).
# Les brokers utilisent souvent des suffixes : .r, +, .x, .raw etc.
# On essaiera plusieurs variantes automatiquement.
MT5_SYMBOL_CANDIDATES = {
    "XAUUSD": ["XAUUSD", "XAUUSD.r", "XAUUSD+", "XAUUSD.x", "GOLD"],
    "NAS100": ["NAS100", "NAS100.r", "USTEC", "US100", "NDX100", "NQ100"],
    "GER40": ["GER40", "GER40.r", "DE40", "DAX40", "DAX", "DE30"],
    "USOUSD": ["USOUSD", "USOIL", "WTI", "OIL.WTI"],
    "BTCUSD": ["BTCUSD", "BTC/USD", "BTCUSD.r", "BITCOIN"],
    "EURUSD": ["EURUSD", "EURUSD.r", "EURUSD+"],
    "GBPUSD": ["GBPUSD", "GBPUSD.r", "GBPUSD+"],
    "USDJPY": ["USDJPY", "USDJPY.r", "USDJPY+"],
    "AUDUSD": ["AUDUSD", "AUDUSD.r", "AUDUSD+"],
    "DXY":    ["DXY", "USDX", "DOLLAR_IDX"],
    "XAGUSD": ["XAGUSD", "XAGUSD.r", "SILVER"],
    "SPX500": ["SPX500", "SP500", "US500", "ES"],
    "UKOIL":  ["UKOIL", "UKOIL.r", "BRENT", "BRO"],
}

MT5_TF = {
    "M1":  mt5.TIMEFRAME_M1,
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1":  mt5.TIMEFRAME_H1,
    "H4":  mt5.TIMEFRAME_H4,
    "D1":  mt5.TIMEFRAME_D1,
}


def find_mt5_symbol(internal_name: str) -> str | None:
    """Cherche le symbole MT5 correspondant a internal_name (essaie plusieurs variantes)."""
    candidates = MT5_SYMBOL_CANDIDATES.get(internal_name, [internal_name])
    all_symbols = {s.name for s in mt5.symbols_get()}
    for candidate in candidates:
        if candidate in all_symbols:
            return candidate
    return None


def fetch_mt5(symbol: str, tf: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Telecharge les bougies MT5 entre start et end."""
    timeframe = MT5_TF[tf]
    # Assure que le symbole est selectionne
    if not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"Impossible de selectionner {symbol}")

    rates = mt5.copy_rates_range(symbol, timeframe, start, end)
    if rates is None or len(rates) == 0:
        err = mt5.last_error()
        raise RuntimeError(f"Pas de donnees pour {symbol} {tf} : {err}")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time").sort_index()
    # Standardise les colonnes pour matcher le format du bot
    df = df.rename(columns={"tick_volume": "volume"})
    df = df[["open", "high", "low", "close", "volume"]].copy()
    return df


def fetch_and_cache(internal_name: str, tf: str, years: float) -> None:
    """Fetch + cache pour 1 instrument / 1 TF.

    Note Vantage : les TF intraday (M1/M5/M15) sont limites a ~2 ans.
    On capse automatiquement pour ces TF.
    """
    end = datetime.now(tz=timezone.utc).replace(minute=0, second=0, microsecond=0)
    # Cap pour TF intraday Vantage
    effective_years = years
    if tf in ("M1", "M5", "M15") and years > 2:
        effective_years = 2.0
    start = end - timedelta(days=int(365 * effective_years))

    mt5_symbol = find_mt5_symbol(internal_name)
    if mt5_symbol is None:
        print(f"  [{internal_name} / {tf}] SYMBOLE INTROUVABLE chez le broker")
        return

    path = cache_path(internal_name, tf)
    print(f"  [{internal_name} ({mt5_symbol}) / {tf}] Fetch {start.date()} -> {end.date()}...", end=" ", flush=True)
    try:
        df = fetch_mt5(mt5_symbol, tf, start, end)
    except Exception as e:
        print(f"ERREUR: {e}")
        return
    print(f"{len(df)} bougies -> {path.name}")
    df.to_parquet(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=2.0, help="Annees d'historique")
    ap.add_argument("--instrument", type=str, default=None)
    ap.add_argument("--tf", type=str, default=None)
    ap.add_argument("--primary-only", action="store_true")
    args = ap.parse_args()

    # Initialise MT5
    print("Initialisation MT5...")
    if not mt5.initialize():
        print(f"ERREUR initialize : {mt5.last_error()}")
        print("Verifie que MT5 est lance et qu'un compte est connecte.")
        sys.exit(1)

    info = mt5.account_info()
    if info is not None:
        print(f"Compte connecte : {info.login} ({info.server}, {info.currency})")
    else:
        print("ATTENTION : aucun compte connecte (mais data peut etre dispo).")

    # Liste les symboles dispos chez le broker
    all_symbols = mt5.symbols_get()
    print(f"Symboles disponibles chez le broker : {len(all_symbols)}")

    if args.instrument:
        instruments = [args.instrument]
    elif args.primary_only:
        instruments = [k for k, v in INSTRUMENTS.items() if v["role"] == "primary"]
    else:
        instruments = list(INSTRUMENTS.keys())

    tfs = [args.tf] if args.tf else ALL_TF
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Affiche le mapping
    print("\n=== Mapping symboles ===")
    for inst in instruments:
        mt5_name = find_mt5_symbol(inst)
        status = mt5_name if mt5_name else "NON TROUVE"
        print(f"  {inst:8s} -> {status}")
    print()

    for inst in instruments:
        if inst not in INSTRUMENTS:
            print(f"INSTRUMENT INCONNU : {inst}")
            continue
        print(f"=== {inst} ({INSTRUMENTS[inst]['label']}) ===")
        for tf in tfs:
            fetch_and_cache(inst, tf, args.years)
        print()

    mt5.shutdown()
    print(f"OK. Cache dans {DATA_DIR}")


if __name__ == "__main__":
    main()
