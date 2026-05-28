"""V21 TRAIN : Transformer profond cross-asset sur GPU Blackwell.

Charge chunks v21 (avec cross-asset), train V21Net, multi-task loss.
Sauve v21_best.pt avec best val AUC.

Splits embargo 2j identiques V20 pour comparaison fair.
"""
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

from bot_v2.v21_model import V21Net, v21_multitask_loss, count_params


TRAIN_END = pd.Timestamp("2025-05-20").to_datetime64()
VAL_START = pd.Timestamp("2025-05-22").to_datetime64()
VAL_END = pd.Timestamp("2025-11-20").to_datetime64()
OOS_START = pd.Timestamp("2025-11-22").to_datetime64()


class V21Dataset(Dataset):
    def __init__(self, m1, m15, h1, d1, ref, labels, rr):
        self.m1 = torch.tensor(m1, dtype=torch.float32)
        self.m15 = torch.tensor(m15, dtype=torch.float32)
        self.h1 = torch.tensor(h1, dtype=torch.float32)
        self.d1 = torch.tensor(d1, dtype=torch.float32)
        self.ref = torch.tensor(ref, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.rr = torch.tensor(rr, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return (self.m1[i], self.m15[i], self.h1[i], self.d1[i],
                self.ref[i], self.labels[i], self.rr[i])


def evaluate(model, loader, device):
    model.eval()
    all_proba, all_target = [], []
    use_amp = (device == "cuda")
    with torch.no_grad():
        for m1, m15, h1, d1, ref, lbl, rr in loader:
            m1, m15, h1, d1 = m1.to(device, non_blocking=True), m15.to(device, non_blocking=True), h1.to(device, non_blocking=True), d1.to(device, non_blocking=True)
            ref, lbl = ref.to(device, non_blocking=True), lbl.to(device, non_blocking=True)
            if use_amp:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    p, _ = model(m1, m15, h1, d1, ref)
            else:
                p, _ = model(m1, m15, h1, d1, ref)
            all_proba.append(p.float().cpu().numpy())
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
    print(f"=== V21 TRAIN (cross-asset + self-discovery) ===")
    print(f"Device : {device}")
    if device == "cuda":
        print(f"  GPU : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # Load chunks
    chunks_dir = Path("/dev/shm/v21_chunks") if Path("/dev/shm/v21_chunks").exists() else ROOT / "data" / "v21_chunks"
    print(f"Loading chunks from {chunks_dir}", flush=True)
    chunks = sorted(chunks_dir.glob("*.npz"))
    if not chunks:
        print("FAIL : aucun chunk V21")
        return
    t0_load = time.time()
    parts = {"m1_seqs": [], "m15_seqs": [], "h1_seqs": [], "d1_seqs": [],
             "ref_seqs": [], "labels": [], "rr_target": [], "timestamps": []}
    for i, c in enumerate(chunks):
        d = np.load(c, allow_pickle=True)
        for k in parts:
            parts[k].append(d[k])
        if (i+1) % 20 == 0:
            print(f"  loaded {i+1}/{len(chunks)} chunks ({time.time()-t0_load:.0f}s)", flush=True)
    print(f"  concatenating arrays...", flush=True)
    t1 = time.time()
    m1 = np.concatenate(parts["m1_seqs"])
    m15 = np.concatenate(parts["m15_seqs"])
    h1 = np.concatenate(parts["h1_seqs"])
    d1 = np.concatenate(parts["d1_seqs"])
    ref = np.concatenate(parts["ref_seqs"])
    lbl = np.concatenate(parts["labels"])
    rr = np.concatenate(parts["rr_target"])
    ts = np.array(np.concatenate(parts["timestamps"]), dtype="datetime64[ns]")
    del parts
    print(f"  concat done in {time.time()-t1:.0f}s | total load {time.time()-t0_load:.0f}s", flush=True)
    print(f"Loaded : m1={m1.shape}, ref={ref.shape}, lbl={lbl.shape}", flush=True)
    print(f"  Total : {len(lbl):,} samples, WR brut : {lbl.mean()*100:.1f}%")

    train_mask = ts < TRAIN_END
    val_mask = (ts >= VAL_START) & (ts < VAL_END)
    oos_mask = ts >= OOS_START
    print(f"\nSplits :")
    print(f"  Train : {train_mask.sum():,}")
    print(f"  Val   : {val_mask.sum():,}")
    print(f"  OOS   : {oos_mask.sum():,}")

    if val_mask.sum() < 50 or oos_mask.sum() < 50:
        print("!! Splits trop petits")
        return

    ds_train = V21Dataset(m1[train_mask], m15[train_mask], h1[train_mask], d1[train_mask],
                            ref[train_mask], lbl[train_mask], rr[train_mask])
    ds_val = V21Dataset(m1[val_mask], m15[val_mask], h1[val_mask], d1[val_mask],
                          ref[val_mask], lbl[val_mask], rr[val_mask])
    ds_oos = V21Dataset(m1[oos_mask], m15[oos_mask], h1[oos_mask], d1[oos_mask],
                          ref[oos_mask], lbl[oos_mask], rr[oos_mask])

    # V5 FIX : retire AMP qui causait freeze (gradients underflow FP16)
    # Garde batch large + num_workers eleve
    BATCH = 512 if device == "cuda" else 32
    train_loader = DataLoader(ds_train, batch_size=BATCH, shuffle=True,
                                num_workers=16, pin_memory=(device=="cuda"),
                                persistent_workers=(device=="cuda"),
                                prefetch_factor=4)
    val_loader = DataLoader(ds_val, batch_size=BATCH, num_workers=8, pin_memory=(device=="cuda"))
    oos_loader = DataLoader(ds_oos, batch_size=BATCH, num_workers=8, pin_memory=(device=="cuda"))

    # V7 : V21Net SIMPLE (V20-like CNN + cross-asset, sans Transformer)
    import os as _os
    hidden = int(_os.environ.get("V21_HIDDEN", "64"))
    embed_dim = int(_os.environ.get("V21_EMBED", "128"))
    dropout = float(_os.environ.get("V21_DROPOUT", "0.3"))
    lr = float(_os.environ.get("V21_LR", "1e-3"))
    n_epochs = int(_os.environ.get("V21_EPOCHS", "30"))
    patience = int(_os.environ.get("V21_PATIENCE", "5"))

    model = V21Net(hidden=hidden, embed_dim=embed_dim, dropout=dropout).to(device)
    print(f"\nModel : V21Net SIMPLE (hidden={hidden}, embed_dim={embed_dim}, dropout={dropout})")
    print(f"  params : {count_params(model):,} (~{count_params(model)/1e6:.1f}M)")
    print(f"  batch  : {BATCH}, lr : {lr}, epochs : {n_epochs}, patience : {patience}")

    # torch.compile SKIP : AMP + batch 1024 donne deja ~x3 speed, compile = compile slow
    # try:
    #     model = torch.compile(model, mode="reduce-overhead")
    # except Exception as e:
    #     print(f"  torch.compile : SKIP ({e})")
    print(f"  torch.compile : SKIP (AMP+batch1024 suffit)", flush=True)

    # AdamW + OneCycleLR (warmup intégré)
    n_steps = len(train_loader) * n_epochs
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr,
                                                  total_steps=n_steps,
                                                  pct_start=0.05, anneal_strategy='cos')

    # V5 FIX : AMP DESACTIVE (FP16 causait gradients underflow -> AUC 0.5 freeze 5 epochs)
    use_amp = False
    scaler = None
    print(f"  AMP : OFF (FP32 stable)", flush=True)

    best_val_auc = 0.0
    best_epoch = 0
    bad_epochs = 0

    print(f"\n=== Training ===", flush=True)
    for epoch in range(n_epochs):
        model.train()
        t0 = time.time()
        losses, bces, rrs = [], [], []
        for m1_b, m15_b, h1_b, d1_b, ref_b, lbl_b, rr_b in train_loader:
            m1_b, m15_b = m1_b.to(device, non_blocking=True), m15_b.to(device, non_blocking=True)
            h1_b, d1_b = h1_b.to(device, non_blocking=True), d1_b.to(device, non_blocking=True)
            ref_b, lbl_b, rr_b = ref_b.to(device, non_blocking=True), lbl_b.to(device, non_blocking=True), rr_b.to(device, non_blocking=True)
            opt.zero_grad()
            if use_amp:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    p, r = model(m1_b, m15_b, h1_b, d1_b, ref_b)
                    loss, bce, rr_mse = v21_multitask_loss(p, r, lbl_b, rr_b, w_rr=0.1)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                p, r = model(m1_b, m15_b, h1_b, d1_b, ref_b)
                loss, bce, rr_mse = v21_multitask_loss(p, r, lbl_b, rr_b, w_rr=0.1)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()  # OneCycleLR step per batch
            losses.append(loss.item())
            bces.append(bce.item())
            rrs.append(rr_mse.item())

        train_loss = float(np.mean(losses))
        train_bce = float(np.mean(bces))
        train_rr_mse = float(np.mean(rrs))
        val_m = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1:>2} : loss={train_loss:.4f} (bce={train_bce:.4f}, rr_mse={train_rr_mse:.4f}) | "
              f"val AUC={val_m['auc']:.4f} | "
              f"WR@0.65={val_m.get('wr@0.65', 0):.1f}% (N={val_m.get('n@0.65', 0)}) "
              f"| {elapsed:.0f}s", flush=True)

        if val_m["auc"] > best_val_auc:
            best_val_auc = val_m["auc"]
            best_epoch = epoch + 1
            bad_epochs = 0
            torch.save({"state_dict": model.state_dict(),
                         "hidden": hidden, "embed_dim": embed_dim,
                         "version": "v21"},
                        ROOT / "v21_best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stop at epoch {epoch+1} (best @ {best_epoch})")
                break

    print(f"\nBest val AUC : {best_val_auc:.4f} @ epoch {best_epoch}")

    # Eval final
    ckpt = torch.load(ROOT / "v21_best.pt", weights_only=False)
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
            print(f"  thr={thr} : N={oos_m[f'n@{thr}']:>6}, WR={oos_m[f'wr@{thr}']:.1f}%")

    # Comparaison V20 vs V21
    v20_recap = ROOT / "v20_train_recap.json"
    if v20_recap.exists():
        v20 = json.loads(v20_recap.read_text())
        print(f"\n=== COMPARAISON V20 vs V21 (OOS) ===")
        print(f"  V20 AUC OOS : {v20['oos_auc']:.4f}")
        print(f"  V21 AUC OOS : {oos_m['auc']:.4f}")
        print(f"  Delta        : {oos_m['auc'] - v20['oos_auc']:+.4f}")

    recap = {
        "best_val_auc": best_val_auc, "best_epoch": best_epoch,
        "train_auc": train_m["auc"], "val_auc": val_m["auc"], "oos_auc": oos_m["auc"],
        "gap_train_oos": train_m["auc"] - oos_m["auc"],
        "oos_metrics": oos_m, "val_metrics": val_m, "train_metrics": train_m,
        "hyperparams": {"hidden": hidden, "embed_dim": embed_dim,
                          "dropout": dropout, "lr": lr},
        "n_params": count_params(model),
        "version": "v21",
    }
    (ROOT / "v21_train_recap.json").write_text(json.dumps(recap, indent=2, default=str))
    print(f"\nRecap : v21_train_recap.json")
    print(f"Model : v21_best.pt ({(ROOT / 'v21_best.pt').stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
