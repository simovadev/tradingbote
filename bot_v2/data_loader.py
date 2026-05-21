"""Chargement des bougies multi-TF, multi-actif.

Reutilise le cache parquet de `data/fetch.py` (cree par dukascopy-python).
Pas de download ici : on suppose le cache existant. Si manquant -> erreur claire.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd

from bot_v2.config import DATA_DIR, INSTRUMENTS, ALL_TF


class CacheMissingError(FileNotFoundError):
    """Le cache parquet pour cet (instrument, tf) n'existe pas."""


def _resolve_data_dir() -> Path:
    """V6 (2026-05-21) : permet de switcher la source de donnees via env var.

    BUILD_DATA_DIR=data_vantage -> charge depuis le dossier data_vantage/
    (a la racine du repo). Sinon -> DATA_DIR par defaut (data/cache).

    Utilise par le training V6 pour entrainer le ML sur Vantage au lieu
    d'Admiral. En production live, la var est non definie et on retombe
    sur le defaut (compat V5).
    """
    override = os.getenv("BUILD_DATA_DIR")
    if override:
        p = Path(override)
        # Resolution : si relatif, considere relatif a la racine du repo
        if not p.is_absolute():
            repo_root = Path(__file__).resolve().parent.parent
            p = repo_root / p
        return p
    return DATA_DIR


def cache_path(instrument: str, tf: str) -> Path:
    return _resolve_data_dir() / f"{instrument}_{tf}.parquet"


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

    # OPTIM V5 (2026-05-20) : si start/end fournis, on lit le parquet par row groups
    # (pyarrow filters) au lieu de charger les 2.6M bougies puis slicer.
    # Gain : ~10x moins d'I/O sur les chunks 3 mois (130k au lieu 2.6M).
    if start is not None or end is not None:
        try:
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(path)
            # Lecture par row groups, on filtre apres lecture de chaque groupe.
            # Le parquet est trie par timestamp donc on peut skip les groups hors fenetre.
            dfs = []
            for rg_idx in range(pf.num_row_groups):
                rg_meta = pf.metadata.row_group(rg_idx)
                # Tente de recuperer min/max du row group (col index)
                ts_col = None
                for i in range(rg_meta.num_columns):
                    col = rg_meta.column(i)
                    if col.path_in_schema in ("__index_level_0__", "time", "timestamp"):
                        ts_col = col
                        break
                if ts_col is not None and ts_col.statistics is not None:
                    rg_min = pd.Timestamp(ts_col.statistics.min)
                    rg_max = pd.Timestamp(ts_col.statistics.max)
                    if rg_min.tz is None:
                        rg_min = rg_min.tz_localize("UTC")
                    if rg_max.tz is None:
                        rg_max = rg_max.tz_localize("UTC")
                    if end is not None and rg_min > end:
                        continue
                    if start is not None and rg_max < start:
                        continue
                df_rg = pf.read_row_group(rg_idx).to_pandas()
                dfs.append(df_rg)
            if dfs:
                df = pd.concat(dfs)
            else:
                df = pd.read_parquet(path)
        except Exception:
            df = pd.read_parquet(path)
    else:
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
