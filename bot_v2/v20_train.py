"""V20 ETAPE 4 : Train V20 ICTNet (SANS asset_embedding) sur 78 actifs.

Difference vs V19 :
- Architecture V20ICTNet (pas d'asset_emb) -> generalisation pure ICT
- Dataset v20_dataset.npz (78 actifs, 722k seqs)
- asset_id ignore par le model mais passe pour compat

Splits avec embargo 2j (anti-leakage).
Optim : AdamW + cosine + warmup + ICT-aware loss.
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

ROOT = Path("/workspace/TradingBot") if sys.platform == "linux" else Path("c:/Users/Shadow/TradingBot")
sys.path.insert(0, str(ROOT))

from bot_v2.v20_model import V20ICTNet, ict_aware_loss, count_params


TRAIN_END = pd.Timestamp("2025-05-20").to_datetime64()
VAL_START = pd.Timestamp("2025-05-22").to_datetime64()
VAL_END = pd.Timestamp("2025-11-20").to_datetime64()
OOS_START = pd.Timestamp("2025-11-22").to_datetime64()


class V20Dataset(Dataset):
    def __init__(self, m1, m15, h1, ict, labels):
        self.m1 = torch.tensor(m1, dtype=torch.float32)
        self.m15 = torch.tensor(m15, dtype=torch.float32)
        self.h1 = torch.tensor(h1, dtype=torch.float32)
        self.ict = torch.tensor(ict, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return (self.m1[i], self.m15[i], self.h1[i],
                self.ict[i], self.labels[i])


def normalize_ict_features(ict_train, ict_val, ict_oos):
    means = ict_train.mean(axis=0)
    stds = ict_train.std(axis=0) + 1e-8
    binary_mask = stds < 0.1
    means[binary_mask] = 0.0
    stds[binary_mask] = 1.0
    return ((ict_train - means) / stds,
            (ict_val - means) / stds,
            (ict_oos - means) / stds,
            (means, stds))


def evaluate(model, loader, device):
    model.eval()
    all_proba, all_target = [], []
    with torch.no_grad():
        for m1, m15, h1, ict, lbl in loader:
            m1, m15, h1 = m1.to(device), m15.to(device), h1.to(device)
            ict, lbl = ict.to(device), lbl.to(device)
            p = model(m1, m15, h1, ict)
            all_proba.append(p.cpu().numpy())
            all_target.append(lbl.cpu().numpy())
    proba = np.concatenate(all_proba)
    target = np.concatenate(all_target)
    try:
        auc = float(roc_auc_score(target, proba))
    except Exception:
        auc = float("nan")
    metrics = {"auc": auc, "n": len(target)}
    for thr in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85):
        mask = proba >= thr
        n = int(mask.sum())
        if n >= 5:
            wr = float(target[mask].mean() * 100)
            metrics[f"n@{thr}"] = n
            metrics[f"wr@{thr}"] = wr
    return metrics


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== V20 TRAIN (sans asset_emb) ===")
    print(f"Device : {device}")
    if device == "cuda":
        print(f"  GPU : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # Charge les 78 chunks depuis /dev/shm (RAM) ou disque
    chunks_dir = Path("/dev/shm/v20_chunks") if Path("/dev/shm/v20_chunks").exists() else ROOT / "data" / "v20_chunks"
    chunk_files = sorted(chunks_dir.glob("*.npz"))
    print(f"Loading {len(chunk_files)} chunks from {chunks_dir}...")
    parts = {"m1_seqs": [], "m15_seqs": [], "h1_seqs": [],
             "ict_feats": [], "labels": [], "timestamps": []}
    for c in chunk_files:
        d = np.load(c, allow_pickle=True)
        for k in parts:
            parts[k].append(d[k])
    m1 = np.concatenate(parts["m1_seqs"])
    m15 = np.concatenate(parts["m15_seqs"])
    h1 = np.concatenate(parts["h1_seqs"])
    ict = np.concatenate(parts["ict_feats"])
    lbl = np.concatenate(parts["labels"])
    ts = np.array(np.concatenate(parts["timestamps"]), dtype="datetime64[ns]")
    del parts
    print(f"Loaded : m1={m1.shape}, ict={ict.shape}, lbl={lbl.shape}")

    print(f"Total samples : {len(lbl):,}")
    print(f"  Champions (1) : {(lbl == 1).sum():,}")
    print(f"  Clear LOSS (0): {(lbl == 0).sum():,}")
    print(f"  WR brut       : {lbl.mean()*100:.1f}%")

    train_mask = ts < TRAIN_END
    val_mask = (ts >= VAL_START) & (ts < VAL_END)
    oos_mask = ts >= OOS_START

    print(f"\nSplits temporels (embargo 2j) :")
    print(f"  Train (< 2025-05-20)        : {train_mask.sum():,}")
    print(f"  Val   (2025-05-22 ~ 11-20)  : {val_mask.sum():,}")
    print(f"  OOS   (>= 2025-11-22)       : {oos_mask.sum():,}")

    if val_mask.sum() < 50 or oos_mask.sum() < 50:
        print("!! Splits trop petits, abort")
        return

    ict_train, ict_val, ict_oos, (means, stds) = normalize_ict_features(
        ict[train_mask], ict[val_mask], ict[oos_mask])

    ds_train = V20Dataset(m1[train_mask], m15[train_mask], h1[train_mask],
                            ict_train, lbl[train_mask])
    ds_val = V20Dataset(m1[val_mask], m15[val_mask], h1[val_mask],
                          ict_val, lbl[val_mask])
    ds_oos = V20Dataset(m1[oos_mask], m15[oos_mask], h1[oos_mask],
                          ict_oos, lbl[oos_mask])

    BATCH = 512 if device == "cuda" else 64
    train_loader = DataLoader(ds_train, batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=(device=="cuda"))
    val_loader = DataLoader(ds_val, batch_size=BATCH, num_workers=4, pin_memory=(device=="cuda"))
    oos_loader = DataLoader(ds_oos, batch_size=BATCH, num_workers=4, pin_memory=(device=="cuda"))

    n_ict = ict.shape[1]
    model = V20ICTNet(n_ict_features=n_ict,
                       m1_len=240, m15_len=60, h1_len=30).to(device)
    print(f"\nModel : V20ICTNet, params={count_params(model):,}")
    print(f"Batch size : {BATCH}")

    # ICT indices pour ict_aware_loss
    ICT_NAMES = ["ob_strength", "sweep_strength", "retest_count", "is_unicorn",
                  "rr", "tp_source_htf", "tp_source_capped",
                  "daily_bias_aligned", "daily_bias_neutral"]
    daily_bias_idx = ICT_NAMES.index("daily_bias_aligned")
    sweep_idx = ICT_NAMES.index("sweep_strength")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
    best_val_auc = 0.0
    best_epoch = 0
    patience = 5
    bad_epochs = 0

    print(f"\n=== Training (max 30 epochs, patience={patience}) ===")
    for epoch in range(30):
        model.train()
        t0 = time.time()
        losses = []
        for m1_b, m15_b, h1_b, ict_b, lbl_b in train_loader:
            m1_b, m15_b, h1_b = m1_b.to(device), m15_b.to(device), h1_b.to(device)
            ict_b, lbl_b = ict_b.to(device), lbl_b.to(device)
            p = model(m1_b, m15_b, h1_b, ict_b)
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

        train_loss = float(np.mean(losses))
        val_m = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1:>2} : train_loss={train_loss:.4f} | "
              f"val AUC={val_m['auc']:.4f} | "
              f"WR@0.65={val_m.get('wr@0.65', 0):.1f}% (N={val_m.get('n@0.65', 0)}) "
              f"| {elapsed:.0f}s", flush=True)

        if val_m["auc"] > best_val_auc:
            best_val_auc = val_m["auc"]
            best_epoch = epoch + 1
            bad_epochs = 0
            torch.save({"state_dict": model.state_dict(),
                         "ict_means": means, "ict_stds": stds,
                         "n_ict": n_ict, "version": "v20"},
                        ROOT / "v20_best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stop at epoch {epoch+1} (best @ {best_epoch})")
                break

    print(f"\nBest val AUC : {best_val_auc:.4f} @ epoch {best_epoch}")

    # Eval finale
    ckpt = torch.load(ROOT / "v20_best.pt", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    print(f"\n=== EVAL FINAL ===")
    oos_m = evaluate(model, oos_loader, device)
    val_m = evaluate(model, val_loader, device)
    train_m = evaluate(model, train_loader, device)
    print(f"Train AUC : {train_m['auc']:.4f}")
    print(f"Val AUC   : {val_m['auc']:.4f}")
    print(f"OOS AUC   : {oos_m['auc']:.4f}")
    print(f"Gap train-oos : {train_m['auc'] - oos_m['auc']:+.4f}")
    print()
    print(f"OOS thresholds :")
    for thr in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85):
        if f"wr@{thr}" in oos_m:
            print(f"  thr={thr} : N={oos_m[f'n@{thr}']:>5}, WR={oos_m[f'wr@{thr}']:.1f}%")

    recap = {
        "best_val_auc": best_val_auc, "best_epoch": best_epoch,
        "train_auc": train_m["auc"], "val_auc": val_m["auc"], "oos_auc": oos_m["auc"],
        "gap_train_oos": train_m["auc"] - oos_m["auc"],
        "oos_metrics": oos_m, "val_metrics": val_m, "train_metrics": train_m,
        "version": "v20",
    }
    (ROOT / "v20_train_recap.json").write_text(json.dumps(recap, indent=2, default=str))
    print(f"\nRecap : v20_train_recap.json")
    print(f"Model : v20_best.pt ({(ROOT / 'v20_best.pt').stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
