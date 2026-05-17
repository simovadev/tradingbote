"""Chargement des bougies multi-TF, multi-actif.

Reutilise le cache parquet de `data/fetch.py` (cree par dukascopy-python).
Pas de download ici : on suppose le cache existant. Si manquant -> erreur claire.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from bot_v2.config import DATA_DIR, INSTRUMENTS, ALL_TF


class CacheMissingError(FileNotFoundError):
    """Le cache parquet pour cet (instrument, tf) n'existe pas."""


def cache_path(instrument: str, tf: str) -> Path:
    return DATA_DIR / f"{instrument}_{tf}.parquet"


def load(
    instrument: str,
    tf: str,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Charge un (instrument, tf) depuis le cache parquet.

    Retourne un DataFrame indexe en UTC avec colonnes : open, high, low, close, volume.
    Filtre [start, end] si fournis (inclusifs).
    """
    if instrument not in INSTRUMENTS:
        raise ValueError(f"Instrument inconnu : {instrument}")
    if tf not in ALL_TF:
        raise ValueError(f"Timeframe inconnu : {tf} (attendu : {ALL_TF})")

    path = cache_path(instrument, tf)
    if not path.exists():
        raise CacheMissingError(
            f"Cache absent : {path}\n"
            f"  -> Lancer : python -m data.fetch --instrument {instrument} --tf {tf}"
        )

    df = pd.read_parquet(path)

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    if start is not None:
        df = df[df.index >= start]
    if end is not None:
        df = df[df.index <= end]

    return df


def load_all_tf(
    instrument: str,
    tfs: Optional[list[str]] = None,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> dict[str, pd.DataFrame]:
    """Charge plusieurs timeframes d'un actif d'un coup.

    Retourne {tf: DataFrame}.
    """
    tfs = tfs or ALL_TF
    out: dict[str, pd.DataFrame] = {}
    for tf in tfs:
        try:
            out[tf] = load(instrument, tf, start=start, end=end)
        except CacheMissingError:
            # On laisse passer : appelant decide si critique ou non.
            pass
    return out


def cached_instruments() -> list[str]:
    """Liste les instruments qui ont au moins un fichier parquet en cache."""
    if not DATA_DIR.exists():
        return []
    found: set[str] = set()
    for p in DATA_DIR.glob("*.parquet"):
        # nom = INSTRUMENT_TF.parquet
        inst = p.stem.rsplit("_", 1)[0]
        if inst in INSTRUMENTS:
            found.add(inst)
    return sorted(found)


def cache_status() -> pd.DataFrame:
    """Tableau recap du cache : par (instrument, tf), nombre de bougies + bornes."""
    rows = []
    for inst in INSTRUMENTS:
        for tf in ALL_TF:
            path = cache_path(inst, tf)
            if not path.exists():
                rows.append({"instrument": inst, "tf": tf, "candles": 0,
                             "start": None, "end": None})
                continue
            df = pd.read_parquet(path)
            rows.append({
                "instrument": inst,
                "tf": tf,
                "candles": len(df),
                "start": df.index.min(),
                "end": df.index.max(),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Diagnostic rapide
    df = cache_status()
    print(df.to_string(index=False))
