"""v22_dataset_builder.py - Build dataset V22 PROPRE (sans look-ahead).

Pour chaque OB safe detecte en M5 entre 2023 et 2026, on construit :
- Features ICT (tous safe, calcules a validation_index strict)
- Sequence M5 (60 dernieres bougies normalisees, input du DL)
- Label triple-barrier : 1 si TP touche AVANT SL, 0 sinon

Periode : 2023-01-01 -> 2026-04-01 (evite le drift 2018-2020 du V21)
Actif : XAUUSD (le mieux valide deja)
Timeframe : M5 (au lieu de M1 V21, moins de bruit)

Output : .npz avec X_seq, X_feat, y, ts
"""
from __future__ import annotations
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

from bot_v2.concepts.safe import (
    find_obs_safe, daily_bias_safe, htf_value_at_t, atr_safe, extract_seq_safe,
)
from bot_screener.time_utils import killzone_name, current_macro

DATA = ROOT / "data_vantage"

# Parametres
ASSET = "XAUUSD"
START_DATE = "2023-01-01"
END_DATE = "2026-04-01"
SEQ_LEN = 60                 # 60 bougies M5 = 5h de contexte
N_FEAT = 12                  # nb de features tabulaires

# Plan trade pour le label triple-barrier
SL_ATR_MULT = 1.0            # SL = 1 ATR
TP_RR = 2.0                  # TP = 2R
MAX_HOLD_BARS = 60           # 60 bougies M5 = 5h max

# Filtres OB
DISPLACEMENT_MIN = 1.5
SWING_LOOKBACK = 20

# Step d'exploration (pour ne pas analyser chaque bougie M5)
STEP = 1   # scan toutes les bougies (mais on detecte les OBs uniquement quand un OB se forme)


def build_features(df_m5, df_h1, df_d1, ob, asset):
    """Construit le vecteur features tabulaires pour un OB.

    TOUTES les features sont calculees au moment ob.validation_index strict.
    """
    val_idx = ob.validation_index
    val_ts = ob.validation_ts
    feats = []

    # 1. Direction (1 = bullish, -1 = bearish)
    feats.append(1.0 if ob.direction == "bullish" else -1.0)

    # 2. Displacement en ATR
    feats.append(min(ob.displacement_atr, 5.0))   # cap a 5

    # 3. ATR au moment de validation (en % du prix)
    atr = atr_safe(df_m5, val_idx)
    price = float(df_m5["close"].iloc[val_idx])
    feats.append(atr / price if price > 0 else 0.0)

    # 4-5. Killzone (one-hot reduit : NY AM ou London ou other)
    kz = killzone_name(val_ts)
    feats.append(1.0 if kz == "NY AM" else 0.0)
    feats.append(1.0 if kz == "London" else 0.0)

    # 6. Macro (1 si dans une macro window)
    macro = current_macro(val_ts)
    feats.append(1.0 if macro else 0.0)

    # 7-8. Daily bias aligne (D1)
    d1_bias = daily_bias_safe(df_d1, val_ts)
    if d1_bias["ok"]:
        d1_haussier = 1.0 if d1_bias["bias"] == "haussier" else -1.0
        # Aligne avec direction
        feats.append(d1_haussier)
        feats.append(1.0 if (
            (ob.direction == "bullish" and d1_haussier > 0) or
            (ob.direction == "bearish" and d1_haussier < 0)
        ) else -1.0)
    else:
        feats.append(0.0); feats.append(0.0)

    # 9. H1 momentum : close H1 actuelle - close H1 il y a 5 bougies
    h1_close = htf_value_at_t(df_h1, val_ts, "close")
    h1_close_5ago = None
    if h1_close is not None:
        # Trouve la valeur H1 5 bougies plus tot
        pos = df_h1.index.searchsorted(val_ts, side="left") - 1 - 5
        if pos >= 0:
            h1_close_5ago = float(df_h1["close"].iloc[pos])
    if h1_close and h1_close_5ago:
        h1_momentum = (h1_close - h1_close_5ago) / h1_close_5ago
        # Aligne avec direction
        feats.append(h1_momentum * (1.0 if ob.direction == "bullish" else -1.0))
    else:
        feats.append(0.0)

    # 10. Distance vs PDH/PDL en ATR
    if df_d1 is not None and len(df_d1) >= 2:
        target_day = pd.Timestamp(val_ts).normalize()
        df_d1_past = df_d1[df_d1.index < target_day]
        if len(df_d1_past) >= 1 and atr > 0:
            pdh = float(df_d1_past["high"].iloc[-1])
            pdl = float(df_d1_past["low"].iloc[-1])
            if ob.direction == "bullish":
                dist = (pdh - price) / atr   # distance jusqu'a la cible PDH
            else:
                dist = (price - pdl) / atr
            feats.append(min(max(dist, -5.0), 10.0))   # cap
        else:
            feats.append(0.0)
    else:
        feats.append(0.0)

    # 11. Heure dans la journee (sin)
    hour = val_ts.hour + val_ts.minute / 60
    feats.append(np.sin(2 * np.pi * hour / 24))

    # 12. Jour de la semaine (1=lundi, 5=vendredi, -1=autres)
    dow = val_ts.weekday()
    feats.append((dow + 1) / 5.0 if dow < 5 else -1.0)

    assert len(feats) == N_FEAT, f"Expected {N_FEAT} feats, got {len(feats)}"
    return np.array(feats, dtype=np.float32)


def simulate_triple_barrier(df_m5, ob, atr):
    """Simule trade avec SL = 1 ATR, TP = 2 ATR (RR 2), timeout 60 bougies.

    Plan : entry = close de la bougie de validation (entry market).
    Retourne 1 si TP touche AVANT SL, 0 sinon (LOSS ou timeout).
    """
    val_idx = ob.validation_index
    if val_idx + 1 + MAX_HOLD_BARS >= len(df_m5):
        return None   # pas assez de futur, on jette ce sample

    entry = float(df_m5["close"].iloc[val_idx])
    if ob.direction == "bullish":
        sl = entry - SL_ATR_MULT * atr
        tp = entry + TP_RR * SL_ATR_MULT * atr
    else:
        sl = entry + SL_ATR_MULT * atr
        tp = entry - TP_RR * SL_ATR_MULT * atr

    if abs(entry - sl) < 1e-9:
        return None

    # Simule bougie par bougie sur les MAX_HOLD prochaines
    fh = df_m5["high"].values[val_idx + 1:val_idx + 1 + MAX_HOLD_BARS]
    fl = df_m5["low"].values[val_idx + 1:val_idx + 1 + MAX_HOLD_BARS]
    fo = df_m5["open"].values[val_idx + 1:val_idx + 1 + MAX_HOLD_BARS]
    for i in range(len(fh)):
        if ob.direction == "bullish":
            hit_sl = fl[i] <= sl
            hit_tp = fh[i] >= tp
        else:
            hit_sl = fh[i] >= sl
            hit_tp = fl[i] <= tp
        # Si les 2 touchent dans la meme bougie, on prend conservateur (SL)
        if hit_sl and hit_tp:
            # On regarde le sens du body : si open plus pres du SL -> SL touche d'abord
            if abs(fo[i] - sl) < abs(fo[i] - tp):
                return 0
            return 1
        if hit_sl:
            return 0
        if hit_tp:
            return 1
    return 0   # timeout = considere comme LOSS


def main():
    print(f"\n=== BUILD DATASET V22 ===")
    print(f"Actif : {ASSET}")
    print(f"Periode : {START_DATE} -> {END_DATE}")
    print(f"Timeframe : M5")
    print(f"Seq len : {SEQ_LEN} | Features tabulaires : {N_FEAT}")
    print(f"Label : triple-barrier RR={TP_RR}, max_hold={MAX_HOLD_BARS} bougies M5")
    print()

    # Load data
    df_m5_full = pd.read_parquet(DATA / f"{ASSET}_M5.parquet")[["open", "high", "low", "close"]]
    df_h1_full = pd.read_parquet(DATA / f"{ASSET}_H1.parquet")[["open", "high", "low", "close"]]
    df_d1_full = pd.read_parquet(DATA / f"{ASSET}_D1.parquet")[["open", "high", "low", "close"]]

    start = pd.Timestamp(START_DATE, tz="UTC")
    end = pd.Timestamp(END_DATE, tz="UTC")

    # On garde du contexte avant START pour les sequences M5 (60 bougies) et H1 (50 bougies = 50h)
    context_start = start - pd.Timedelta(days=10)
    df_m5 = df_m5_full[(df_m5_full.index >= context_start) & (df_m5_full.index < end)]
    df_h1 = df_h1_full[(df_h1_full.index >= context_start) & (df_h1_full.index < end)]
    df_d1 = df_d1_full[(df_d1_full.index >= start - pd.Timedelta(days=30)) & (df_d1_full.index < end)]

    print(f"M5 : {len(df_m5)} bougies")
    print(f"H1 : {len(df_h1)} bougies")
    print(f"D1 : {len(df_d1)} bougies")

    # Detect TOUS les OBs sur la periode complete (en mode strict as_of_index <= validation)
    # Note : find_obs_safe avec as_of_index=full retourne tous les OBs de la serie
    # mais chaque OB a son propre validation_index qui represente le moment ou il a ete
    # CONFIRME en live -> les features et labels seront calcules a CE moment.
    print("\nDetect OBs safe ...")
    obs = find_obs_safe(
        df_m5, as_of_index=len(df_m5) - 1,
        sweep_lookback=SWING_LOOKBACK,
        displacement_min_atr=DISPLACEMENT_MIN,
    )
    print(f"OBs detectes : {len(obs)}")

    # Filtre : ne garder que les OBs dont validation_ts est dans la periode cible
    obs = [ob for ob in obs if start <= ob.validation_ts < end]
    print(f"OBs dans periode {START_DATE} -> {END_DATE} : {len(obs)}")

    # Build dataset
    X_seq = []
    X_feat = []
    y = []
    ts = []
    skipped_short_future = 0
    skipped_features = 0
    skipped_label = 0

    for i, ob in enumerate(obs):
        if i % 500 == 0 and i > 0:
            print(f"  {i}/{len(obs)} ({i*100/len(obs):.0f}%)")
        try:
            # Sequence M5 jusqu'a validation_index (inclus)
            seq = extract_seq_safe(df_m5, ob.validation_index, n=SEQ_LEN)
            if seq is None:
                skipped_short_future += 1
                continue

            # Features ICT
            feats = build_features(df_m5, df_h1, df_d1, ob, ASSET)
            if np.any(np.isnan(feats)) or np.any(np.isinf(feats)):
                skipped_features += 1
                continue

            # Label triple-barrier
            atr = atr_safe(df_m5, ob.validation_index)
            if atr <= 0:
                skipped_label += 1
                continue
            label = simulate_triple_barrier(df_m5, ob, atr)
            if label is None:
                skipped_label += 1
                continue

            X_seq.append(seq)
            X_feat.append(feats)
            y.append(label)
            ts.append(ob.validation_ts)
        except Exception as e:
            print(f"  ERR ob {i} : {type(e).__name__}: {e}")

    X_seq = np.array(X_seq, dtype=np.float32)
    X_feat = np.array(X_feat, dtype=np.float32)
    y = np.array(y, dtype=np.int8)
    ts = np.array([t.timestamp() for t in ts], dtype=np.int64)

    print(f"\nDataset final : {len(X_seq)} samples")
    print(f"Skipped (short future) : {skipped_short_future}")
    print(f"Skipped (features) : {skipped_features}")
    print(f"Skipped (label) : {skipped_label}")
    print(f"Labels : {y.sum()} WIN ({y.mean()*100:.1f}%) / {(1-y).sum()} LOSS")
    print(f"X_seq shape : {X_seq.shape}")
    print(f"X_feat shape : {X_feat.shape}")

    out = ROOT / f"v22_dataset_{ASSET}_M5_2023_2026.npz"
    np.savez_compressed(out, X_seq=X_seq, X_feat=X_feat, y=y, ts=ts)
    print(f"\nSauvegarde : {out}")
    print(f"Taille : {out.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
