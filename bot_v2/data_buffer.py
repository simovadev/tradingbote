"""DataBuffer : stockage persistent en memoire des bougies M1 pour le bot live.

Architecture :
- Au demarrage : charge data_vantage/{ASSET}_M1.parquet (7+ mois d'historique)
- Comble le trou via MT5 entre last_ts du parquet et now
- En live : a chaque nouvelle bougie M1, on l'ajoute en memoire (pas de re-fetch)
- Persistance : sauvegarde toutes les 5 min sur disque

Le buffer M1 alimente tous les TF (M15/H1/D1 sont resamples depuis M1).

Usage :
    buffer = DataBuffer(asset="XAUUSD", mt5_exec=mt5_exec)
    buffer.load_and_fill()  # 1 fois au boot
    df_m1 = buffer.get_m1()  # acces direct memoire
    df_m15 = buffer.get_m15()  # resampled depuis M1
    buffer.update()  # appelle a chaque scan (ajoute nouvelle bougie si dispo)
    buffer.save()  # appel periodique (toutes les 5min)
"""
from __future__ import annotations

import os
import time
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data_vantage"

log = logging.getLogger(__name__)


class DataBuffer:
    """Buffer M1 persistent pour 1 actif. Resample HTF a la demande."""

    def __init__(self, asset: str, mt5_exec, root: Optional[Path] = None):
        self.asset = asset
        self.mt5_exec = mt5_exec
        self.root = Path(root) if root else ROOT
        self.data_dir = self.root / "data_vantage"
        self.parquet_path = self.data_dir / f"{asset}_M1.parquet"
        self.df_m1: Optional[pd.DataFrame] = None
        self.last_save_ts: float = 0.0
        self.save_interval_sec: int = 300  # 5 min

        # Cache HTF (invalide a chaque update)
        self._cache_m15: Optional[pd.DataFrame] = None
        self._cache_h1: Optional[pd.DataFrame] = None
        self._cache_h4: Optional[pd.DataFrame] = None
        self._cache_d1: Optional[pd.DataFrame] = None
        self._cache_valid_until: Optional[pd.Timestamp] = None

    # ============ LOAD ============

    def load_and_fill(self) -> bool:
        """Charge parquet local + comble le trou via MT5.

        Returns True si succes.
        """
        # 1. Charge parquet local
        if self.parquet_path.exists():
            try:
                self.df_m1 = pd.read_parquet(self.parquet_path)
                if self.df_m1.index.tz is None:
                    self.df_m1.index = self.df_m1.index.tz_localize("UTC")
                log.info(
                    f"BUFFER {self.asset}: charge {len(self.df_m1):,} M1 de "
                    f"{self.df_m1.index[0]} a {self.df_m1.index[-1]}"
                )
            except Exception as e:
                log.error(f"BUFFER {self.asset}: erreur lecture parquet : {e}")
                self.df_m1 = None
        else:
            log.warning(f"BUFFER {self.asset}: pas de parquet local {self.parquet_path.name}")
            self.df_m1 = None

        # 2. Comble le trou via MT5
        ok = self.fill_gap_from_mt5()
        if not ok and self.df_m1 is None:
            log.error(f"BUFFER {self.asset}: aucune donnee dispo (parquet + MT5 KO)")
            return False
        return True

    def fill_gap_from_mt5(self) -> bool:
        """Recupere les bougies manquantes entre last_ts et now via MT5."""
        now_utc = pd.Timestamp.now(tz="UTC").floor("min")

        if self.df_m1 is not None and len(self.df_m1) > 0:
            last_ts = self.df_m1.index[-1]
            gap_minutes = int((now_utc - last_ts).total_seconds() / 60)
            if gap_minutes <= 1:
                log.info(f"BUFFER {self.asset}: pas de trou (last={last_ts}, now={now_utc})")
                self._invalidate_cache()
                return True
            log.info(f"BUFFER {self.asset}: trou de {gap_minutes} min ({last_ts} -> {now_utc})")
            n_fetch = min(gap_minutes + 200, 80000)  # +marge, max MT5 80k
        else:
            n_fetch = 80000  # fetch max disponible

        # Fetch via MT5
        try:
            df_new = self.mt5_exec.get_bars(self.asset, "M1", n_fetch, force_sync=True)
            if df_new is None or len(df_new) == 0:
                log.warning(f"BUFFER {self.asset}: fetch MT5 KO (None)")
                return False
        except Exception as e:
            log.error(f"BUFFER {self.asset}: exception fetch MT5 : {e}")
            return False

        # Merge avec existant
        if self.df_m1 is None:
            self.df_m1 = df_new
        else:
            # Ne garde que les bougies plus recentes que last_ts
            last_ts = self.df_m1.index[-1]
            df_new_filtered = df_new[df_new.index > last_ts]
            if len(df_new_filtered) > 0:
                self.df_m1 = pd.concat([self.df_m1, df_new_filtered])
                log.info(
                    f"BUFFER {self.asset}: ajoute {len(df_new_filtered)} nouvelles M1 "
                    f"(total {len(self.df_m1):,})"
                )

        # Dedup + sort
        self.df_m1 = self.df_m1[~self.df_m1.index.duplicated(keep="last")].sort_index()
        self._invalidate_cache()
        return True

    # ============ UPDATE ============

    def update(self) -> int:
        """A appeler a chaque scan. Recupere les nouvelles M1 depuis MT5.

        Returns nb de nouvelles bougies ajoutees.
        """
        if self.df_m1 is None:
            return 0

        last_ts = self.df_m1.index[-1]
        now_utc = pd.Timestamp.now(tz="UTC").floor("min")
        gap_min = int((now_utc - last_ts).total_seconds() / 60)
        if gap_min <= 0:
            return 0

        # Fetch les dernieres bougies (avec marge de 10 pour s'assurer)
        n_fetch = max(gap_min + 10, 50)
        try:
            df_new = self.mt5_exec.get_bars(self.asset, "M1", n_fetch, force_sync=True)
            if df_new is None or len(df_new) == 0:
                return 0
        except Exception as e:
            log.debug(f"BUFFER {self.asset} update KO : {e}")
            return 0

        df_new_filtered = df_new[df_new.index > last_ts]
        n_new = len(df_new_filtered)
        if n_new > 0:
            self.df_m1 = pd.concat([self.df_m1, df_new_filtered])
            self.df_m1 = self.df_m1[~self.df_m1.index.duplicated(keep="last")].sort_index()
            self._invalidate_cache()

        # Auto-save si > save_interval_sec depuis derniere save
        if time.time() - self.last_save_ts > self.save_interval_sec:
            self.save()

        return n_new

    # ============ GET ============

    def get_m1(self, n: Optional[int] = None) -> Optional[pd.DataFrame]:
        """Retourne les N dernieres M1 (ou toutes si n=None)."""
        if self.df_m1 is None:
            return None
        if n is None:
            return self.df_m1.copy()
        return self.df_m1.tail(n).copy()

    def get_m15(self, n: int = 5500) -> Optional[pd.DataFrame]:
        if self._cache_m15 is not None:
            return self._cache_m15.tail(n).copy()
        if self.df_m1 is None:
            return None
        self._cache_m15 = self._resample_m1("15min")
        return self._cache_m15.tail(n).copy()

    def get_h1(self, n: int = 1320) -> Optional[pd.DataFrame]:
        if self._cache_h1 is not None:
            return self._cache_h1.tail(n).copy()
        if self.df_m1 is None:
            return None
        self._cache_h1 = self._resample_m1("1h")
        return self._cache_h1.tail(n).copy()

    def get_h4(self, n: int = 500) -> Optional[pd.DataFrame]:
        if self._cache_h4 is not None:
            return self._cache_h4.tail(n).copy()
        if self.df_m1 is None:
            return None
        self._cache_h4 = self._resample_m1("4h")
        return self._cache_h4.tail(n).copy()

    def get_d1(self, n: int = 100) -> Optional[pd.DataFrame]:
        if self._cache_d1 is not None:
            return self._cache_d1.tail(n).copy()
        if self.df_m1 is None:
            return None
        self._cache_d1 = self._resample_m1("1D")
        return self._cache_d1.tail(n).copy()

    def _resample_m1(self, rule: str) -> pd.DataFrame:
        """Resample M1 -> rule (OHLCV)."""
        df = self.df_m1.resample(rule, label="left", closed="left").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()
        return df

    def _invalidate_cache(self):
        self._cache_m15 = None
        self._cache_h1 = None
        self._cache_h4 = None
        self._cache_d1 = None

    # ============ SAVE ============

    def save(self) -> bool:
        """Sauvegarde le buffer M1 sur disque."""
        if self.df_m1 is None or len(self.df_m1) == 0:
            return False
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = self.parquet_path.with_suffix(".parquet.tmp")
            self.df_m1.to_parquet(tmp_path)
            tmp_path.replace(self.parquet_path)
            self.last_save_ts = time.time()
            log.debug(f"BUFFER {self.asset}: saved {len(self.df_m1):,} M1")
            return True
        except Exception as e:
            log.error(f"BUFFER {self.asset}: save KO : {e}")
            return False
