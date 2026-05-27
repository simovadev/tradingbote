"""V19 ETAPE 4 : Train V19 ICTNet sur sequences extraites.

Strategy :
- Train : trades anciens (avant 2025-05-22)
- Val   : 2025-05-22 -> 2025-11-22 (6 mois)
- OOS   : 2025-11-24 -> 2026-05-22 (6 mois)

Optimizer : AdamW + cosine schedule + warmup
Early stopping sur val loss
Adversarial validation post-train pour mesurer le drift
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score

ROOT = "c:/Users/Shadow/TradingBot" if sys.platform == "win32" else "/workspace/TradingBot"
sys.path.insert(0, ROOT)

from bot_v2.v19_model import V19ICTNet, ict_aware_loss, count_params


# V19 FIX leakage : embargo 2j entre splits.
# simulate_trade scanne jusqu'a 1440 M1 (24h) pour fermer un trade.
# Sans embargo, des trades de fin TRAIN ont leur outcome calcule sur des bougies VAL
# -> contamination future->passe. 2 jours = couvre 24h scan + marge.
TRAIN_END = pd.Timestamp("2025-05-20").to_datetime64()
VAL_START = pd.Timestamp("2025-05-22").to_datetime64()  # embargo 2j
VAL_END = pd.Timestamp("2025-11-20").to_datetime64()
OOS_START = pd.Timestamp("2025-11-22").to_datetime64()  # embargo 2j


class V19Dataset(Dataset):
    def __init__(self, m1, m15, h1, ict, aid, labels):
        self.m1 = torch.tensor(m1, dtype=torch.float32)
        self.m15 = torch.tensor(m15, dtype=torch.float32)
        self.h1 = torch.tensor(h1, dtype=torch.float32)
        self.ict = torch.tensor(ict, dtype=torch.float32)
        self.aid = torch.tensor(aid, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return (self.m1[i], self.m15[i], self.h1[i],
                self.ict[i], self.aid[i], self.labels[i])


def normalize_ict_features(ict_train, ict_val, ict_oos):
    """Z-score normalisation sur les features ICT (skip binaires)."""
    means = ict_train.mean(axis=0)
    stds = ict_train.std(axis=0) + 1e-8
    # Pour binaires (std petit), ne pas normaliser
    binary_mask = stds < 0.1
    means[binary_mask] = 0.0
    stds[binary_mask] = 1.0
    return ((ict_train - means) / stds,
            (ict_val - means) / stds,
            (ict_oos - means) / stds,
            (means, stds))


def evaluate(model, loader, device, ict_idx):
    model.eval()
    all_proba, all_target = [], []
    with torch.no_grad():
        for m1, m15, h1, ict, aid, lbl in loader:
            m1, m15, h1 = m1.to(device), m15.to(device), h1.to(device)
            ict, aid, lbl = ict.to(device), aid.to(device), lbl.to(device)
            p = model(m1, m15, h1, ict, aid)
            all_proba.append(p.cpu().numpy())
            all_target.append(lbl.cpu().numpy())
    proba = np.concatenate(all_proba)
    target = np.concatenate(all_target)
    try:
        auc = float(roc_auc_score(target, proba))
    except Exception:
        auc = float("nan")
    metrics = {"auc": auc, "n": len(target)}
    # WR @ thresholds
    for thr in (0.55, 0.60, 0.65, 0.70, 0.75):
        mask = proba >= thr
        n = int(mask.sum())
        if n >= 5:
            wr = float(target[mask].mean() * 100)
            metrics[f"n@{thr}"] = n
            metrics[f"wr@{thr}"] = wr
    return metrics


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== V19 TRAIN ===")
    print(f"Device : {device}\n")

    # Load data
    npz = np.load(Path(f"{ROOT}/data/v19_sequences.npz"), allow_pickle=True)
    m1, m15, h1 = npz["m1_seqs"], npz["m15_seqs"], npz["h1_seqs"]
    ict, aid, lbl = npz["ict_feats"], npz["asset_ids"], npz["labels"]
    ts = np.array(npz["timestamps"], dtype="datetime64[ns]")

    print(f"Total samples : {len(lbl)}")
    print(f"  Champions (1) : {(lbl == 1).sum()}")
    print(f"  Clear LOSS (0): {(lbl == 0).sum()}")

    # Splits temporels AVEC EMBARGO 2j (anti-leakage future->passe)
    train_mask = ts < TRAIN_END
    val_mask = (ts >= VAL_START) & (ts < VAL_END)
    oos_mask = ts >= OOS_START

    print(f"\nSplits temporels :")
    print(f"  Train : {train_mask.sum()} samples")
    print(f"  Val   : {val_mask.sum()} samples")
    print(f"  OOS   : {oos_mask.sum()} samples")

    if val_mask.sum() < 50 or oos_mask.sum() < 50:
        print("!! Splits trop petits, abort")
        return

    # Normalize ICT features
    ict_train, ict_val, ict_oos, (means, stds) = normalize_ict_features(
        ict[train_mask], ict[val_mask], ict[oos_mask])

    ds_train = V19Dataset(m1[train_mask], m15[train_mask], h1[train_mask],
                           ict_train, aid[train_mask], lbl[train_mask])
    ds_val = V19Dataset(m1[val_mask], m15[val_mask], h1[val_mask],
                         ict_val, aid[val_mask], lbl[val_mask])
    ds_oos = V19Dataset(m1[oos_mask], m15[oos_mask], h1[oos_mask],
                         ict_oos, aid[oos_mask], lbl[oos_mask])

    BATCH = 64
    train_loader = DataLoader(ds_train, batch_size=BATCH, shuffle=True, num_workers=0)
    val_loader = DataLoader(ds_val, batch_size=BATCH, num_workers=0)
    oos_loader = DataLoader(ds_oos, batch_size=BATCH, num_workers=0)

    # Model
    n_assets = int(aid.max()) + 1
    n_ict = ict.shape[1]
    model = V19ICTNet(n_assets=n_assets, n_ict_features=n_ict,
                       m1_len=240, m15_len=60, h1_len=30).to(device)
    print(f"\nModel : V19ICTNet, params={count_params(model):,}")

    # Index ICT (pour ict_aware_loss)
    ICT_FEAT_NAMES = ["ob_strength", "sweep_strength", "retest_count", "is_unicorn",
                       "rr", "tp_source_htf", "tp_source_capped",
                       "daily_bias_aligned", "daily_bias_neutral"]
    daily_bias_idx = ICT_FEAT_NAMES.index("daily_bias_aligned")
    sweep_idx = ICT_FEAT_NAMES.index("sweep_strength")
    # has_FVG_sync est au-dela des 9 premieres, on prend daily_bias seul pour test

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
    best_val_auc = 0.0
    best_epoch = 0
    patience = 5
    bad_epochs = 0

    print(f"\n=== Training ===")
    for epoch in range(30):
        model.train()
        t0 = time.time()
        losses = []
        for m1_b, m15_b, h1_b, ict_b, aid_b, lbl_b in train_loader:
            m1_b, m15_b, h1_b = m1_b.to(device), m15_b.to(device), h1_b.to(device)
            ict_b, aid_b, lbl_b = ict_b.to(device), aid_b.to(device), lbl_b.to(device)
            p = model(m1_b, m15_b, h1_b, ict_b, aid_b)
            loss, _, _ = ict_aware_loss(p, lbl_b, ict_b,
                                          daily_bias_idx=daily_bias_idx,
                                          sweep_strength_idx=sweep_idx,
                                          has_fvg_idx=0, rr_idx=4, w_ict=0.3)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        sched.step()

        train_loss = np.mean(losses)
        val_m = evaluate(model, val_loader, device, daily_bias_idx)
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1:>2} : train_loss={train_loss:.4f} | "
              f"val AUC={val_m['auc']:.4f} | "
              f"WR@0.65={val_m.get('wr@0.65', 0):.1f}% (N={val_m.get('n@0.65', 0)}) "
              f"| {elapsed:.0f}s", flush=True)

        if val_m["auc"] > best_val_auc:
            best_val_auc = val_m["auc"]
            best_epoch = epoch + 1
            bad_epochs = 0
            # Save best
            torch.save({"state_dict": model.state_dict(),
                         "ict_means": means, "ict_stds": stds,
                         "n_assets": n_assets, "n_ict": n_ict},
                        Path(f"{ROOT}/v19_best.pt"))
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stop at epoch {epoch+1} (best @ {best_epoch})")
                break

    print(f"\nBest val AUC : {best_val_auc:.4f} @ epoch {best_epoch}")

    # Eval finale
    ckpt = torch.load(Path(f"{ROOT}/v19_best.pt"), weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    print(f"\n=== EVAL FINAL ===")
    oos_m = evaluate(model, oos_loader, device, daily_bias_idx)
    val_m = evaluate(model, val_loader, device, daily_bias_idx)
    train_m = evaluate(model, train_loader, device, daily_bias_idx)
    print(f"Train AUC : {train_m['auc']:.4f}")
    print(f"Val AUC   : {val_m['auc']:.4f}")
    print(f"OOS AUC   : {oos_m['auc']:.4f}")
    print(f"Gap train-oos : {train_m['auc'] - oos_m['auc']:+.4f}")
    print()
    print(f"OOS thresholds :")
    for thr in (0.55, 0.60, 0.65, 0.70, 0.75):
        if f"wr@{thr}" in oos_m:
            print(f"  thr={thr} : N={oos_m[f'n@{thr}']}, WR={oos_m[f'wr@{thr}']:.1f}%")

    recap = {
        "best_val_auc": best_val_auc, "best_epoch": best_epoch,
        "train_auc": train_m["auc"], "val_auc": val_m["auc"], "oos_auc": oos_m["auc"],
        "gap_train_oos": train_m["auc"] - oos_m["auc"],
        "oos_metrics": oos_m, "val_metrics": val_m, "train_metrics": train_m,
    }
    Path(f"{ROOT}/v19_train_recap.json").write_text(json.dumps(recap, indent=2, default=str))
    print(f"\nRecap : v19_train_recap.json")


if __name__ == "__main__":
    main()
