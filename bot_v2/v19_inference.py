"""V19 INFERENCE : wrapper PyTorch pour utiliser v19_best.pt en live.

API compatible avec le pipeline existant :
    predictor = V19Predictor.load(model_path)
    proba = predictor.predict_one(r, ob, instrument, df_ltf, df_d1, df_m15, df_h1, asset_idx)

Le predictor :
- Charge v19_best.pt une seule fois au boot
- Pour chaque OB : extrait sequences M1/M15/H1, features ICT, fait forward pass
- Retourne proba [0, 1] compatible avec le seuil ML existant
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ROOT = dossier parent de bot_v2/ (relatif au fichier, marche partout)
ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, ROOT)

# Lazy torch import (eviter crash si torch indisponible)
_torch = None
_V19ICTNet = None


def _ensure_torch():
    global _torch, _V19ICTNet
    if _torch is None:
        import torch as _t
        from bot_v2.v19_model import V19ICTNet as _M
        _torch = _t
        _V19ICTNet = _M
    return _torch, _V19ICTNet


# Hyperparams identiques V19
M1_LEN = 240
M15_LEN = 60
H1_LEN = 30

# Liste actifs - asset_id = index dans cette liste (doit matcher le train)
ALL_ASSETS = [
    "BTCUSD", "ETHUSD", "USDMXN", "XAUUSD", "USDJPY", "USDZAR", "USDCAD", "NZDUSD",
    "GBPUSD", "CL-OIL", "EURUSD", "AUDUSD", "USDCHF", "NAS100", "XAGUSD",
    "DJ30", "GAS-C", "HK50", "GER40", "FRA40", "UK100", "Nikkei225",
    "BVSPX", "SP500", "Coffee-C", "Cocoa-C", "Wheat-C", "Sugar-C",
]
ASSET_TO_ID = {a: i for i, a in enumerate(ALL_ASSETS)}

# Memes 50 features ICT que v19_prepare_turbo.py (ORDRE CRITIQUE)
ICT_FEATURES = [
    "ob_strength", "sweep_strength", "retest_count", "is_unicorn",
    "rr", "tp_source_htf", "tp_source_capped",
    "daily_bias_aligned", "daily_bias_neutral",
    "kz_london", "kz_ny_am", "kz_ny_pm", "kz_asia", "kz_ny_lunch", "kz_london_close",
    "has_smt", "has_feu_vert", "has_breaker_kz", "has_mss_fvg",
    "has_po3_dist", "has_phase_expansion", "has_open_midnight_respect",
    "has_FVG_sync", "has_parent_ob", "has_grandparent_ob",
    "has_good_zone", "has_session_direction",
    "ob_group_size", "bars_sweep_to_validation", "bars_group_to_validation",
    "is_bullish",
    "hour_sin", "hour_cos", "day_of_week", "minutes_into_killzone",
    "volume_relatif", "vol_ratio_setup",
    "phase_reversal", "phase_manipulation", "has_mss_nearby",
    "po3_body_pct", "po3_upper_wick", "po3_lower_wick",
    "po3_aligned", "po3_htf2_aligned",
    "displacement_ratio", "fib_level",
    "mom_60", "mom_240", "body_ratio_recent", "mom_aligned",
    "atr_ratio_100",
]


def normalize_seq(ohlcv: np.ndarray) -> np.ndarray | None:
    """Identique a v19_prepare_turbo.normalize_chunk."""
    if ohlcv is None or len(ohlcv) == 0:
        return None
    arr = ohlcv.astype(np.float32).copy()
    closes = arr[:, 3]
    ref = closes[-1]
    if ref == 0 or not np.isfinite(ref):
        return None
    out = np.empty_like(arr)
    out[:, 0] = (arr[:, 0] / ref) - 1.0
    out[:, 1] = (arr[:, 1] / ref) - 1.0
    out[:, 2] = (arr[:, 2] / ref) - 1.0
    out[:, 3] = (closes / ref) - 1.0
    vol_mean = arr[:, 4].mean()
    out[:, 4] = arr[:, 4] / vol_mean if vol_mean > 0 else 0.0
    return np.clip(out, -0.20, 0.20)


def extract_seq_from_df(df, ts, n_bars):
    """Extrait n_bars derniers OHLCV STRICTEMENT avant ts."""
    if df is None or df.empty:
        return None
    past = df[df.index < ts]
    if len(past) < n_bars:
        return None
    chunk = past.iloc[-n_bars:][["open", "high", "low", "close", "volume"]].values
    return normalize_seq(chunk)


class V19Predictor:
    """Predicteur V19 ICTNet. Load 1x au boot, predict_one() pour chaque OB."""

    _instance: "V19Predictor | None" = None

    def __init__(self, model_path: Path):
        torch, V19ICTNet = _ensure_torch()

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        ckpt = torch.load(str(model_path), map_location=self.device, weights_only=False)
        n_assets = int(ckpt.get("n_assets", 28))
        n_ict = int(ckpt.get("n_ict", len(ICT_FEATURES)))
        self.ict_means = np.asarray(ckpt["ict_means"], dtype=np.float32)
        self.ict_stds = np.asarray(ckpt["ict_stds"], dtype=np.float32)

        self.model = V19ICTNet(n_assets=n_assets, n_ict_features=n_ict).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.n_assets = n_assets
        self.n_ict = n_ict

    @classmethod
    def get_instance(cls, model_path: Path | None = None) -> "V19Predictor | None":
        """Singleton : 1 seul predicteur charge en RAM."""
        if cls._instance is not None:
            return cls._instance
        if model_path is None:
            model_path = Path(f"{ROOT}/bot_v2/v19_best.pt")
            if not model_path.exists():
                model_path = Path(f"{ROOT}/v19_best.pt")
        if not model_path.exists():
            return None
        try:
            cls._instance = cls(model_path)
            return cls._instance
        except Exception as e:
            import traceback
            print(f"[V19Predictor] Failed to load V19 ({type(e).__name__}): {e}")
            traceback.print_exc()
            return None

    def predict_one(self, ict_features_dict: dict, instrument: str,
                     df_m1: pd.DataFrame, df_m15: pd.DataFrame, df_h1: pd.DataFrame,
                     ts: pd.Timestamp) -> float | None:
        """Retourne proba WIN [0, 1] ou None si donnees insuffisantes."""
        torch = self.torch

        # Sequences
        m1 = extract_seq_from_df(df_m1, ts, M1_LEN)
        if m1 is None:
            return None
        m15 = extract_seq_from_df(df_m15, ts, M15_LEN)
        if m15 is None:
            return None
        h1 = extract_seq_from_df(df_h1, ts, H1_LEN)
        if h1 is None:
            return None

        # ICT features (50, in correct order)
        ict_vec = np.zeros(len(ICT_FEATURES), dtype=np.float32)
        for j, fname in enumerate(ICT_FEATURES):
            v = ict_features_dict.get(fname, 0)
            if isinstance(v, bool):
                v = int(v)
            try:
                ict_vec[j] = float(v)
            except (TypeError, ValueError):
                ict_vec[j] = 0.0

        # Normalisation Z-score (ict_means/stds appris pendant train)
        ict_norm = (ict_vec - self.ict_means) / self.ict_stds

        # Asset id
        asset_id = ASSET_TO_ID.get(instrument, 0)

        # Tensors
        m1_t = torch.tensor(m1, dtype=torch.float32).unsqueeze(0).to(self.device)
        m15_t = torch.tensor(m15, dtype=torch.float32).unsqueeze(0).to(self.device)
        h1_t = torch.tensor(h1, dtype=torch.float32).unsqueeze(0).to(self.device)
        ict_t = torch.tensor(ict_norm, dtype=torch.float32).unsqueeze(0).to(self.device)
        aid_t = torch.tensor([asset_id], dtype=torch.long).to(self.device)

        with torch.no_grad():
            proba = self.model(m1_t, m15_t, h1_t, ict_t, aid_t)
        return float(proba.cpu().numpy()[0])


def predict_proba_v19(r, ob, instrument: str,
                       df_m1: pd.DataFrame = None,
                       df_m15: pd.DataFrame = None,
                       df_h1: pd.DataFrame = None,
                       df_d1: pd.DataFrame = None,
                       mss_setups: list | None = None) -> float | None:
    """API compatible avec predict_proba (live_runner) mais retourne V19.

    Retourne None si V19 indispo ou data insuffisante (caller utilise alors V18.6).
    """
    predictor = V19Predictor.get_instance()
    if predictor is None:
        return None
    if df_m1 is None or df_m15 is None or df_h1 is None:
        return None

    # Compute features ICT via ml_filter (meme code que training)
    try:
        from bot_v2 import ml_filter
        feats = ml_filter._features_from_result(
            r, ob, instrument,
            df_ltf=df_m1, df_d1=df_d1, mss_setups=mss_setups, df_htf=df_m15,
        )
    except Exception:
        return None

    ts = ob.validation_ts
    try:
        return predictor.predict_one(feats, instrument, df_m1, df_m15, df_h1, ts)
    except Exception as e:
        print(f"[V19] predict failed for {instrument}: {e}")
        return None


def is_v19_available() -> bool:
    return V19Predictor.get_instance() is not None


if __name__ == "__main__":
    # Test rapide : charge le modele
    p = V19Predictor.get_instance()
    if p is None:
        print("V19 model NOT FOUND. Place v19_best.pt in bot_v2/ ou racine.")
    else:
        print(f"V19 chargé OK. Device : {p.device}")
        print(f"  n_assets : {p.n_assets}")
        print(f"  n_ict_features : {p.n_ict}")
        # Test forward
        import torch
        with torch.no_grad():
            m1 = torch.randn(1, M1_LEN, 5).to(p.device)
            m15 = torch.randn(1, M15_LEN, 5).to(p.device)
            h1 = torch.randn(1, H1_LEN, 5).to(p.device)
            ict = torch.randn(1, 50).to(p.device)
            aid = torch.tensor([0], dtype=torch.long).to(p.device)
            proba = p.model(m1, m15, h1, ict, aid)
        print(f"  Test forward : proba = {float(proba):.4f}")
