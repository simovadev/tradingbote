"""V20 INFERENCE : wrapper PyTorch pour utiliser v20_best.pt en live.

V20 = ICT universel, sans asset_embedding.
API compatible avec live_runner V19 (asset_id ignore par le model).
"""
from __future__ import annotations

# Torch en premier (Windows DLL load order)
try:
    import torch as _TORCH_EAGER
    _TORCH_LOADED = True
except Exception as _e:
    _TORCH_EAGER = None
    _TORCH_LOADED = False
    _TORCH_ERROR = _e

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, ROOT)

_V20ICTNet = None


def _ensure_torch():
    global _V20ICTNet
    if not _TORCH_LOADED:
        raise RuntimeError(f"Torch indisponible : {_TORCH_ERROR}")
    if _V20ICTNet is None:
        from bot_v2.v20_model import V20ICTNet as _M
        _V20ICTNet = _M
    return _TORCH_EAGER, _V20ICTNet


M1_LEN = 240
M15_LEN = 60
H1_LEN = 30

# 52 features ICT (memes que v20_prepare_turbo, ORDRE CRITIQUE)
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
    if df is None or df.empty:
        return None
    past = df[df.index < ts]
    if len(past) < n_bars:
        return None
    chunk = past.iloc[-n_bars:][["open", "high", "low", "close", "volume"]].values
    return normalize_seq(chunk)


class V20Predictor:
    """Predicteur V20 ICTNet (sans asset_emb). Singleton."""

    _instance: "V20Predictor | None" = None

    def __init__(self, model_path: Path):
        torch, V20ICTNet = _ensure_torch()

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        ckpt = torch.load(str(model_path), map_location=self.device, weights_only=False)
        n_ict = int(ckpt.get("n_ict", len(ICT_FEATURES)))
        self.ict_means = np.asarray(ckpt["ict_means"], dtype=np.float32)
        self.ict_stds = np.asarray(ckpt["ict_stds"], dtype=np.float32)

        self.model = V20ICTNet(n_ict_features=n_ict).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.n_ict = n_ict

    @classmethod
    def get_instance(cls, model_path: Path | None = None) -> "V20Predictor | None":
        if cls._instance is not None:
            return cls._instance
        if model_path is None:
            model_path = Path(f"{ROOT}/bot_v2/v20_best.pt")
            if not model_path.exists():
                model_path = Path(f"{ROOT}/v20_best.pt")
        if not model_path.exists():
            return None
        try:
            cls._instance = cls(model_path)
            return cls._instance
        except Exception as e:
            import traceback
            print(f"[V20Predictor] Failed to load V20 ({type(e).__name__}): {e}")
            traceback.print_exc()
            return None

    def predict_one(self, ict_features_dict: dict, instrument: str,
                     df_m1: pd.DataFrame, df_m15: pd.DataFrame, df_h1: pd.DataFrame,
                     ts: pd.Timestamp) -> float | None:
        torch = self.torch

        m1 = extract_seq_from_df(df_m1, ts, M1_LEN)
        if m1 is None:
            return None
        m15 = extract_seq_from_df(df_m15, ts, M15_LEN)
        if m15 is None:
            return None
        h1 = extract_seq_from_df(df_h1, ts, H1_LEN)
        if h1 is None:
            return None

        ict_vec = np.zeros(len(ICT_FEATURES), dtype=np.float32)
        for j, fname in enumerate(ICT_FEATURES):
            v = ict_features_dict.get(fname, 0)
            if isinstance(v, bool):
                v = int(v)
            try:
                ict_vec[j] = float(v)
            except (TypeError, ValueError):
                ict_vec[j] = 0.0

        ict_norm = (ict_vec - self.ict_means) / self.ict_stds

        m1_t = torch.tensor(m1, dtype=torch.float32).unsqueeze(0).to(self.device)
        m15_t = torch.tensor(m15, dtype=torch.float32).unsqueeze(0).to(self.device)
        h1_t = torch.tensor(h1, dtype=torch.float32).unsqueeze(0).to(self.device)
        ict_t = torch.tensor(ict_norm, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            proba = self.model(m1_t, m15_t, h1_t, ict_t)
        return float(proba.cpu().numpy()[0])


def predict_proba_v20(r, ob, instrument: str,
                       df_m1: pd.DataFrame = None,
                       df_m15: pd.DataFrame = None,
                       df_h1: pd.DataFrame = None,
                       df_d1: pd.DataFrame = None,
                       mss_setups: list | None = None) -> float | None:
    predictor = V20Predictor.get_instance()
    if predictor is None:
        return None
    if df_m1 is None or df_m15 is None or df_h1 is None:
        return None

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
        print(f"[V20] predict failed for {instrument}: {e}")
        return None


def is_v20_available() -> bool:
    return V20Predictor.get_instance() is not None


if __name__ == "__main__":
    p = V20Predictor.get_instance()
    if p is None:
        print("V20 model NOT FOUND.")
    else:
        print(f"V20 charge OK. Device : {p.device}")
        print(f"  n_ict_features : {p.n_ict}")
