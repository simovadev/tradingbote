"""V21 MONSTER inference : wrapper PyTorch pour utiliser v21_monster_best.pt en live.

API compatible avec live_runner_v20 :
- Singleton chargé 1x au boot
- predict_one() : fetch 5 actifs ref + bougies focus + forward V21MonsterNet
"""
from __future__ import annotations

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

_V21MonsterNet = None


def _ensure_torch():
    global _V21MonsterNet
    if not _TORCH_LOADED:
        raise RuntimeError(f"Torch indisponible : {_TORCH_ERROR}")
    if _V21MonsterNet is None:
        from bot_v2.v21_model_monster import V21MonsterNet as _M
        _V21MonsterNet = _M
    return _TORCH_EAGER, _V21MonsterNet


M1_LEN = 240
M15_LEN = 60
H1_LEN = 30
D1_LEN = 15
REF_ASSETS = ["XAUUSD", "USDJPY", "SP500", "BTCUSD", "USDCHF"]
N_REF = 5


def normalize_chunk(ohlcv_chunk):
    """Z-score sur log-returns x100 (meme que v21_prepare_turbo)."""
    if ohlcv_chunk is None or len(ohlcv_chunk) < 2:
        return None
    arr = ohlcv_chunk.astype(np.float32)
    closes = arr[:, 3]
    ref = closes[-1]
    if ref == 0 or not np.isfinite(ref):
        return None
    out = np.empty_like(arr)
    out[:, 0] = np.log(arr[:, 0] / ref) * 100
    out[:, 1] = np.log(arr[:, 1] / ref) * 100
    out[:, 2] = np.log(arr[:, 2] / ref) * 100
    out[:, 3] = np.log(closes / ref) * 100
    price_std = out[:, :4].std()
    if price_std > 1e-6:
        out[:, :4] = out[:, :4] / price_std
    vol_mean = arr[:, 4].mean()
    out[:, 4] = (arr[:, 4] / vol_mean - 1.0) if vol_mean > 0 else 0.0
    return np.clip(out, -10.0, 10.0)


def extract_seq_from_df(df, ts, n_bars):
    if df is None or df.empty:
        return None
    past = df[df.index < ts]
    if len(past) < n_bars:
        return None
    chunk = past.iloc[-n_bars:][["open", "high", "low", "close", "volume"]].values
    return normalize_chunk(chunk)


class V21Predictor:
    """V21 MONSTER predictor. Singleton."""
    _instance: "V21Predictor | None" = None

    def __init__(self, model_path: Path):
        torch, V21MonsterNet = _ensure_torch()
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(str(model_path), map_location=self.device, weights_only=False)
        d_model = ckpt.get("d_model", 192)
        n_heads = ckpt.get("n_heads", 6)
        self.model = V21MonsterNet(d_model=d_model, n_heads=n_heads).to(self.device)
        # Strip torch.compile prefix si present
        sd = ckpt["state_dict"]
        sd_clean = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        self.model.load_state_dict(sd_clean)
        self.model.eval()
        self.d_model = d_model

    @classmethod
    def get_instance(cls, model_path: Path | None = None) -> "V21Predictor | None":
        if cls._instance is not None:
            return cls._instance
        if model_path is None:
            for p in [Path(f"{ROOT}/v21_monster_best.pt"), Path(f"{ROOT}/bot_v2/v21_monster_best.pt")]:
                if p.exists():
                    model_path = p
                    break
        if model_path is None or not model_path.exists():
            return None
        try:
            cls._instance = cls(model_path)
            return cls._instance
        except Exception as e:
            import traceback
            print(f"[V21Predictor] Failed to load V21 ({type(e).__name__}): {e}")
            traceback.print_exc()
            return None

    def predict_one(self, ts: pd.Timestamp,
                     df_m1: pd.DataFrame, df_m15: pd.DataFrame,
                     df_h1: pd.DataFrame, df_d1: pd.DataFrame,
                     ref_m15_dict: dict) -> float | None:
        """Retourne proba [0,1] ou None si data insuffisante.

        ref_m15_dict : {asset_name: df_m15} pour les 5 refs.
        """
        torch = self.torch

        # Focus seqs
        m1 = extract_seq_from_df(df_m1, ts, M1_LEN)
        if m1 is None:
            return None
        m15 = extract_seq_from_df(df_m15, ts, M15_LEN)
        if m15 is None:
            return None
        h1 = extract_seq_from_df(df_h1, ts, H1_LEN)
        if h1 is None:
            return None
        d1 = extract_seq_from_df(df_d1, ts, D1_LEN)
        if d1 is None:
            return None

        # Refs
        refs = []
        for ref in REF_ASSETS:
            df_r = ref_m15_dict.get(ref)
            r_seq = extract_seq_from_df(df_r, ts, M15_LEN)
            if r_seq is None:
                return None
            refs.append(r_seq)
        ref_arr = np.stack(refs)  # (5, 60, 5)

        # Tensors batch 1
        m1_t = torch.tensor(m1, dtype=torch.float32).unsqueeze(0).to(self.device)
        m15_t = torch.tensor(m15, dtype=torch.float32).unsqueeze(0).to(self.device)
        h1_t = torch.tensor(h1, dtype=torch.float32).unsqueeze(0).to(self.device)
        d1_t = torch.tensor(d1, dtype=torch.float32).unsqueeze(0).to(self.device)
        ref_t = torch.tensor(ref_arr, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            p, r, v = self.model(m1_t, m15_t, h1_t, d1_t, ref_t)
        return float(p.cpu().numpy()[0])


def is_v21_available() -> bool:
    return V21Predictor.get_instance() is not None


if __name__ == "__main__":
    p = V21Predictor.get_instance()
    if p is None:
        print("V21 model NOT FOUND.")
    else:
        print(f"V21 MONSTER charge OK. Device : {p.device}")
        print(f"  d_model : {p.d_model}")
