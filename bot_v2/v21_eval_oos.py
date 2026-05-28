"""V21 MONSTER : eval OOS finale (jamais vu pendant training)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score

ROOT = Path("/workspace/TradingBot") if sys.platform == "linux" else Path("c:/Users/Shadow/TradingBot")
sys.path.insert(0, str(ROOT))

from bot_v2.v21_model_monster import V21MonsterNet


OOS_START = pd.Timestamp("2025-11-22").to_datetime64()


class V21Dataset(Dataset):
    def __init__(self, m1, m15, h1, d1, ref, labels):
        self.m1 = torch.tensor(m1, dtype=torch.float32)
        self.m15 = torch.tensor(m15, dtype=torch.float32)
        self.h1 = torch.tensor(h1, dtype=torch.float32)
        self.d1 = torch.tensor(d1, dtype=torch.float32)
        self.ref = torch.tensor(ref, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return (self.m1[i], self.m15[i], self.h1[i], self.d1[i],
                self.ref[i], self.labels[i])


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== V21 MONSTER OOS EVAL ===")
    print(f"Device : {device}")

    # Load chunks
    chunks_dir = Path("/dev/shm/v21_chunks")
    print(f"Loading chunks from {chunks_dir}")
    chunks = sorted(chunks_dir.glob("*.npz"))
    parts = {"m1_seqs": [], "m15_seqs": [], "h1_seqs": [], "d1_seqs": [],
             "ref_seqs": [], "labels": [], "timestamps": []}
    for c in chunks:
        d = np.load(c, allow_pickle=True)
        for k in parts:
            parts[k].append(d[k])
    m1 = np.concatenate(parts["m1_seqs"])
    m15 = np.concatenate(parts["m15_seqs"])
    h1 = np.concatenate(parts["h1_seqs"])
    d1 = np.concatenate(parts["d1_seqs"])
    ref = np.concatenate(parts["ref_seqs"])
    lbl = np.concatenate(parts["labels"])
    ts = np.array(np.concatenate(parts["timestamps"]), dtype="datetime64[ns]")
    del parts

    oos_mask = ts >= OOS_START
    n_oos = oos_mask.sum()
    print(f"OOS samples : {n_oos:,}")

    ds_oos = V21Dataset(m1[oos_mask], m15[oos_mask], h1[oos_mask], d1[oos_mask],
                          ref[oos_mask], lbl[oos_mask])
    oos_loader = DataLoader(ds_oos, batch_size=512, num_workers=8, pin_memory=True)

    # Load model
    ckpt = torch.load(ROOT / "v21_monster_best.pt", weights_only=False)
    d_model = ckpt.get("d_model", 192)
    n_heads = ckpt.get("n_heads", 6)
    model = V21MonsterNet(d_model=d_model, n_heads=n_heads).to(device)
    # Handle compiled model state dict prefix
    state_dict = ckpt["state_dict"]
    new_state = {}
    for k, v in state_dict.items():
        nk = k.replace("_orig_mod.", "") if k.startswith("_orig_mod.") else k
        new_state[nk] = v
    model.load_state_dict(new_state)
    model.eval()
    print(f"Model loaded: V21MonsterNet d_model={d_model} n_heads={n_heads}")

    # Eval
    all_proba = []
    all_target = []
    with torch.no_grad():
        for m1_b, m15_b, h1_b, d1_b, ref_b, lbl_b in oos_loader:
            m1_b, m15_b = m1_b.to(device, non_blocking=True), m15_b.to(device, non_blocking=True)
            h1_b, d1_b = h1_b.to(device, non_blocking=True), d1_b.to(device, non_blocking=True)
            ref_b = ref_b.to(device, non_blocking=True)
            p, _, _ = model(m1_b, m15_b, h1_b, d1_b, ref_b)
            all_proba.append(p.cpu().numpy())
            all_target.append(lbl_b.numpy())
    proba = np.concatenate(all_proba)
    target = np.concatenate(all_target)
    auc = float(roc_auc_score(target, proba))

    print(f"\n=== OOS RESULTS ===")
    print(f"AUC OOS : {auc:.4f}")
    print(f"Proba stats : min={proba.min():.3f} max={proba.max():.3f} mean={proba.mean():.3f} std={proba.std():.3f}")
    print()
    print(f"Seuil | N trades | WR     | % of OOS")
    print(f"------|----------|--------|----------")
    n_total = len(target)
    for thr in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85):
        mask = proba >= thr
        n = int(mask.sum())
        pct = n / n_total * 100
        if n >= 5:
            wr = float(target[mask].mean() * 100)
            print(f" {thr:.2f} | {n:>8,} | {wr:>5.1f}% | {pct:>5.1f}%")
        else:
            print(f" {thr:.2f} | {n:>8,} | (n<5)  | {pct:>5.1f}%")

    recap = {
        "auc_oos": auc,
        "n_oos": int(n_total),
        "proba_stats": {"min": float(proba.min()), "max": float(proba.max()),
                          "mean": float(proba.mean()), "std": float(proba.std())},
        "thresholds": {},
    }
    for thr in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85):
        mask = proba >= thr
        n = int(mask.sum())
        if n >= 5:
            recap["thresholds"][str(thr)] = {"n": n, "wr": float(target[mask].mean() * 100),
                                                "pct_of_oos": n / n_total * 100}
    (ROOT / "v21_monster_oos_recap.json").write_text(json.dumps(recap, indent=2))
    print(f"\nRecap : v21_monster_oos_recap.json")


if __name__ == "__main__":
    main()
