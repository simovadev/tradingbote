"""v22_train.py - Train V22 avec walk-forward strict.

Decoupe le dataset par TIMESTAMP (pas random shuffle) en :
- TRAIN : 2023-01 -> 2024-12 (24 mois)
- VAL   : 2025-01 -> 2025-06 (6 mois)
- OOS   : 2025-07 -> 2026-04 (9 mois) -- jamais touche pendant tuning

Verdict : on regarde AUC OOS reel (pas l'IS).
Si AUC OOS > 0.55 -> edge mesurable
Si AUC OOS [0.50, 0.55] -> edge marginal
Si AUC OOS < 0.50 -> bot mort
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime

from bot_v2.v22_model import V22Model, count_params


DATASET_PATH = ROOT / "v22_dataset_XAUUSD_M5_2023_2026.npz"
BEST_MODEL_PATH = ROOT / "v22_best.pt"
SEED = 42

# Splits par dates UTC
TRAIN_END = pd.Timestamp("2025-01-01", tz="UTC").timestamp()
VAL_END = pd.Timestamp("2025-07-01", tz="UTC").timestamp()

# Hyperparams
BATCH_SIZE = 32
LR = 5e-4
WD = 1e-4
EPOCHS = 50
PATIENCE = 8

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def auc_roc(y_true, y_pred):
    """AUC ROC sans sklearn dep."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(float)
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()
    if n_pos == 0 or n_neg == 0:
        return 0.5
    # rang moyen des positifs
    order = np.argsort(y_pred)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_pred) + 1)
    sum_pos_ranks = ranks[pos_mask].sum()
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"\n=== TRAIN V22 ===")
    print(f"Device : {DEVICE}")
    print(f"Dataset : {DATASET_PATH}")

    data = np.load(DATASET_PATH)
    X_seq = data["X_seq"]      # (N, 60, 4)
    X_feat = data["X_feat"]    # (N, 12)
    y = data["y"]              # (N,)
    ts = data["ts"]            # (N,) seconds since epoch

    print(f"Total samples : {len(y)}")
    print(f"WR baseline (sans modele) : {y.mean()*100:.1f}%")

    # Walk-forward split par ts
    train_mask = ts < TRAIN_END
    val_mask = (ts >= TRAIN_END) & (ts < VAL_END)
    oos_mask = ts >= VAL_END

    print(f"\nSplit walk-forward :")
    print(f"  TRAIN ({pd.Timestamp(ts[train_mask].min(), unit='s')} -> {pd.Timestamp(ts[train_mask].max(), unit='s')}) : {train_mask.sum()} samples, WR {y[train_mask].mean()*100:.1f}%")
    if val_mask.sum() > 0:
        print(f"  VAL   ({pd.Timestamp(ts[val_mask].min(), unit='s')} -> {pd.Timestamp(ts[val_mask].max(), unit='s')}) : {val_mask.sum()} samples, WR {y[val_mask].mean()*100:.1f}%")
    if oos_mask.sum() > 0:
        print(f"  OOS   ({pd.Timestamp(ts[oos_mask].min(), unit='s')} -> {pd.Timestamp(ts[oos_mask].max(), unit='s')}) : {oos_mask.sum()} samples, WR {y[oos_mask].mean()*100:.1f}%")

    if val_mask.sum() < 30 or oos_mask.sum() < 30:
        print("\nWARNING : VAL ou OOS < 30 samples - resultats peu fiables")

    def to_tensors(mask):
        return (
            torch.from_numpy(X_seq[mask]).float(),
            torch.from_numpy(X_feat[mask]).float(),
            torch.from_numpy(y[mask]).float(),
        )

    train_seq, train_feat, train_y = to_tensors(train_mask)
    val_seq, val_feat, val_y = to_tensors(val_mask)
    oos_seq, oos_feat, oos_y = to_tensors(oos_mask)

    train_ds = TensorDataset(train_seq, train_feat, train_y)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # Class imbalance : pos_weight = N_neg / N_pos
    n_pos = train_y.sum().item()
    n_neg = len(train_y) - n_pos
    pos_weight = torch.tensor([n_neg / max(1, n_pos)], device=DEVICE)
    print(f"\npos_weight (class imbalance) : {pos_weight.item():.2f}")

    # Model
    model = V22Model(n_feat_tab=X_feat.shape[1]).to(DEVICE)
    print(f"Model params : {count_params(model):,}")

    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_auc = 0.5
    best_oos_auc = 0.5
    best_epoch = 0
    patience_left = PATIENCE

    for epoch in range(1, EPOCHS + 1):
        # TRAIN
        model.train()
        losses = []
        for sq, ft, yy in train_dl:
            sq = sq.to(DEVICE); ft = ft.to(DEVICE); yy = yy.to(DEVICE).unsqueeze(1)
            optim.zero_grad()
            logit = model(sq, ft)
            loss = criterion(logit, yy)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            losses.append(loss.item())
        sched.step()
        train_loss = np.mean(losses)

        # EVAL
        model.eval()
        with torch.no_grad():
            train_pred = torch.sigmoid(model(train_seq.to(DEVICE), train_feat.to(DEVICE))).cpu().numpy().flatten()
            val_pred = torch.sigmoid(model(val_seq.to(DEVICE), val_feat.to(DEVICE))).cpu().numpy().flatten()
            oos_pred = torch.sigmoid(model(oos_seq.to(DEVICE), oos_feat.to(DEVICE))).cpu().numpy().flatten()

        train_auc = auc_roc(train_y.numpy(), train_pred)
        val_auc = auc_roc(val_y.numpy(), val_pred)
        oos_auc = auc_roc(oos_y.numpy(), oos_pred)

        msg = (f"Epoch {epoch:3d}  loss={train_loss:.4f}  "
               f"AUC train={train_auc:.3f} val={val_auc:.3f} oos={oos_auc:.3f}")
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_oos_auc = oos_auc
            best_epoch = epoch
            patience_left = PATIENCE
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            msg += "  [best]"
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(msg)
                print(f"Early stop a epoch {epoch}")
                break
        print(msg)

    print(f"\n=== FINAL ===")
    print(f"Best epoch : {best_epoch}")
    print(f"Best VAL AUC : {best_val_auc:.4f}")
    print(f"Best OOS AUC : {best_oos_auc:.4f}")

    # Verdict
    if best_oos_auc >= 0.60:
        print("VERDICT : Edge SOLIDE detecte (AUC OOS >= 0.60)")
    elif best_oos_auc >= 0.55:
        print("VERDICT : Edge MARGINAL (AUC OOS [0.55, 0.60[)")
    elif best_oos_auc >= 0.52:
        print("VERDICT : Edge TRES FAIBLE (AUC OOS [0.52, 0.55[)")
    else:
        print("VERDICT : Pas d'edge mesurable (AUC OOS < 0.52). V22 mort comme V21.")

    # WR a differents seuils
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    model.eval()
    with torch.no_grad():
        oos_pred = torch.sigmoid(model(oos_seq.to(DEVICE), oos_feat.to(DEVICE))).cpu().numpy().flatten()
    print(f"\nWR OOS par seuil :")
    for thr in [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70]:
        sel = oos_pred >= thr
        if sel.sum() > 0:
            wr = oos_y.numpy()[sel].mean() * 100
            print(f"  thr {thr:.2f} : {sel.sum()} trades, WR {wr:.1f}%")
        else:
            print(f"  thr {thr:.2f} : 0 trades")


if __name__ == "__main__":
    main()
