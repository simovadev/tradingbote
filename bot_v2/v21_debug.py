"""V21 DEBUG SERIEUX : trouver pourquoi AUC=0.5 stuck.

Tests systematiques :
1. Forward variety : 1000 samples differents donnent des outputs differents ?
2. Train mini : 50k samples 3 epochs en local -> AUC bouge ?
3. Gradients : non-zero / non-NaN apres backward ?
4. Sans cross-asset : V20-like pur sur cross-asset data -> AUC ?
5. Avec un MLP super simple : juste flatten + 2 Linear -> AUC ?
"""
from __future__ import annotations

import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "/workspace/TradingBot")
ROOT = Path("/workspace/TradingBot")

# Charge XAUUSD + EURUSD pour avoir 30-40k samples
print("=== Loading data ===", flush=True)
chunks = []
for asset in ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY"]:
    p = Path(f"/dev/shm/v21_chunks/{asset}.npz")
    if p.exists():
        d = np.load(p, allow_pickle=True)
        chunks.append({
            "m1": d["m1_seqs"], "m15": d["m15_seqs"],
            "h1": d["h1_seqs"], "d1": d["d1_seqs"],
            "ref": d["ref_seqs"], "lbl": d["labels"],
            "rr": d["rr_target"], "ts": d["timestamps"],
        })
        print(f"  {asset}: {len(d['labels'])} samples", flush=True)

m1 = np.concatenate([c["m1"] for c in chunks])
m15 = np.concatenate([c["m15"] for c in chunks])
h1 = np.concatenate([c["h1"] for c in chunks])
d1 = np.concatenate([c["d1"] for c in chunks])
ref = np.concatenate([c["ref"] for c in chunks])
lbl = np.concatenate([c["lbl"] for c in chunks])
rr = np.concatenate([c["rr"] for c in chunks])
print(f"Total: {len(lbl):,} samples, WR brut: {lbl.mean()*100:.1f}%", flush=True)
print(f"  m1 stats : mean={m1.mean():.4f} std={m1.std():.4f} min={m1.min():.4f} max={m1.max():.4f}")
print(f"  ref stats: mean={ref.mean():.4f} std={ref.std():.4f}")
print(f"  Any NaN m1? {np.isnan(m1).any()}, ref? {np.isnan(ref).any()}")
print()

device = "cuda"
N = 20000  # mini dataset
m1_t = torch.tensor(m1[:N], dtype=torch.float32).to(device)
m15_t = torch.tensor(m15[:N], dtype=torch.float32).to(device)
h1_t = torch.tensor(h1[:N], dtype=torch.float32).to(device)
d1_t = torch.tensor(d1[:N], dtype=torch.float32).to(device)
ref_t = torch.tensor(ref[:N], dtype=torch.float32).to(device)
lbl_t = torch.tensor(lbl[:N], dtype=torch.float32).to(device)
print(f"Mini dataset: {N} samples on {device}", flush=True)
print()


# ============ TEST 1 : Forward variety V21Net actuel ============
print("=== TEST 1 : V21Net actuel forward variety ===", flush=True)
from bot_v2.v21_model import V21Net
model = V21Net().to(device)
model.eval()
with torch.no_grad():
    p, r = model(m1_t[:200], m15_t[:200], h1_t[:200], d1_t[:200], ref_t[:200])
print(f"  V21Net proba: min={p.min():.6f} max={p.max():.6f} std={p.std():.6f}")
print(f"  Unique values: {len(torch.unique(p))}/200")
print()


# ============ TEST 2 : MLP TRIVIAL - juste flatten + linear ============
print("=== TEST 2 : MLP trivial sur m1 flatten ===", flush=True)
class TrivialMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(240*5, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
    def forward(self, m1):
        return torch.sigmoid(self.fc(self.flatten(m1)).squeeze(-1))

mlp = TrivialMLP().to(device)
opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
mlp.train()
for epoch in range(5):
    # Mini-batch shuffle
    idx = torch.randperm(N)
    losses = []
    for i in range(0, N, 512):
        b = idx[i:i+512]
        p = mlp(m1_t[b])
        loss = nn.functional.binary_cross_entropy(p, lbl_t[b])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    # Eval
    mlp.eval()
    with torch.no_grad():
        p_all = mlp(m1_t)
    auc = roc_auc_score(lbl_t.cpu().numpy(), p_all.cpu().numpy())
    print(f"  TrivialMLP epoch {epoch+1}: loss={np.mean(losses):.4f} AUC={auc:.4f} | proba std={p_all.std():.4f}")
    mlp.train()
print()


# ============ TEST 3 : V21Net micro-train ============
print("=== TEST 3 : V21Net train sur mini dataset ===", flush=True)
from bot_v2.v21_model import V21Net as Net21
model = Net21().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-3)
model.train()
for epoch in range(5):
    idx = torch.randperm(N)
    losses = []
    grads_norms = []
    for i in range(0, N, 256):
        b = idx[i:i+256]
        p, r = model(m1_t[b], m15_t[b], h1_t[b], d1_t[b], ref_t[b])
        loss = nn.functional.binary_cross_entropy(p, lbl_t[b])
        opt.zero_grad()
        loss.backward()
        # Check gradients
        total_grad = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_grad += param.grad.data.norm(2).item()**2
        grads_norms.append(total_grad**0.5)
        opt.step()
        losses.append(loss.item())
    model.eval()
    with torch.no_grad():
        p_all, _ = model(m1_t, m15_t, h1_t, d1_t, ref_t)
    auc = roc_auc_score(lbl_t.cpu().numpy(), p_all.cpu().numpy())
    print(f"  V21Net epoch {epoch+1}: loss={np.mean(losses):.4f} AUC={auc:.4f} | "
          f"proba std={p_all.std():.4f} | grad_norm avg={np.mean(grads_norms):.4f}")
    model.train()

print()
print("=== DEBUG DONE ===", flush=True)
