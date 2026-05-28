"""V21 SIMPLE : V20-like architecture mais avec cross-asset.

Pourquoi simple :
- V21 Transformer complexe ne convergait pas (AUC 0.5 stuck 3 versions)
- On garde le CNN encoder qui marche (comme V20)
- On ajoute juste les 5 actifs ref via concat simple
- Le model peut grandir plus tard (V22)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=5, dilation=1):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, padding=0, dilation=dilation)

    def forward(self, x):
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class TimeframeEncoder(nn.Module):
    """Identique V20 : CNN simple sur OHLCV."""
    def __init__(self, in_ch=5, hidden=64, n_layers=3, output_dim=128):
        super().__init__()
        layers = []
        ch = in_ch
        for i in range(n_layers):
            layers.append(CausalConv1d(ch, hidden, kernel=5, dilation=2**i))
            layers.append(nn.BatchNorm1d(hidden))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(0.2))
            ch = hidden
        self.cnn = nn.Sequential(*layers)
        self.proj = nn.Conv1d(hidden, output_dim, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = self.proj(x)
        return x.mean(dim=-1)


class V21Net(nn.Module):
    """V21 = V20 sans asset_emb + cross-asset (5 refs concat).

    Architecture proche V20 (qui marchait, AUC OOS 0.82) avec :
    - Encoder M1/M15/H1/D1 (focus actif)
    - Encoder partage pour les 5 actifs ref M15
    - Fusion : concat focus + mean ref + head

    Inputs :
        focus_m1   (B, 240, 5)
        focus_m15  (B, 60, 5)
        focus_h1   (B, 30, 5)
        focus_d1   (B, 15, 5)
        ref_m15    (B, 5, 60, 5)
    """

    def __init__(self, m1_len=240, m15_len=60, h1_len=30, d1_len=15,
                 n_ref=5, hidden=64, embed_dim=128, dropout=0.3):
        super().__init__()
        self.n_ref = n_ref

        # Encoders focus (memes que V20 + 1 pour D1)
        self.enc_m1 = TimeframeEncoder(5, hidden, n_layers=4, output_dim=embed_dim)
        self.enc_m15 = TimeframeEncoder(5, hidden, n_layers=3, output_dim=embed_dim)
        self.enc_h1 = TimeframeEncoder(5, hidden, n_layers=2, output_dim=embed_dim)
        self.enc_d1 = TimeframeEncoder(5, hidden, n_layers=2, output_dim=embed_dim)

        # Encoder ref (partage entre les 5)
        self.enc_ref = TimeframeEncoder(5, hidden, n_layers=3, output_dim=embed_dim)

        # Head : concat focus (4 TFs) + ref aggregated (mean over n_ref)
        total_dim = embed_dim * 4 + embed_dim  # focus + ref_mean
        self.head_proba = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self.head_rr = nn.Sequential(
            nn.Linear(total_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Softplus(),
        )

    def forward(self, focus_m1, focus_m15, focus_h1, focus_d1, ref_m15):
        x_m1 = self.enc_m1(focus_m1)
        x_m15 = self.enc_m15(focus_m15)
        x_h1 = self.enc_h1(focus_h1)
        x_d1 = self.enc_d1(focus_d1)

        # Encode refs (flatten batch)
        B, n_ref, T, C = ref_m15.shape
        ref_flat = ref_m15.reshape(B * n_ref, T, C)
        ref_emb_flat = self.enc_ref(ref_flat)
        ref_emb = ref_emb_flat.reshape(B, n_ref, -1).mean(dim=1)  # (B, embed_dim) - mean over refs

        fused = torch.cat([x_m1, x_m15, x_h1, x_d1, ref_emb], dim=-1)
        proba = torch.sigmoid(self.head_proba(fused).squeeze(-1))
        rr = self.head_rr(fused).squeeze(-1)
        return proba, rr


def v21_multitask_loss(proba, rr_pred, target_win, target_rr, w_rr=0.1):
    eps = 1e-7
    bce = -(target_win * torch.log(proba + eps) + (1 - target_win) * torch.log(1 - proba + eps)).mean()
    win_mask = target_win > 0.5
    if win_mask.any():
        rr_mse = F.mse_loss(rr_pred[win_mask], target_rr[win_mask])
    else:
        rr_mse = torch.tensor(0.0, device=proba.device)
    return bce + w_rr * rr_mse, bce, rr_mse


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = V21Net()
    n = count_params(model)
    print(f"V21Net SIMPLE : {n:,} params (~{n/1e6:.1f}M)")
    B = 4
    p, r = model(torch.randn(B,240,5), torch.randn(B,60,5),
                  torch.randn(B,30,5), torch.randn(B,15,5),
                  torch.randn(B,5,60,5))
    print(f"proba: {p.shape} range=[{p.min():.3f}, {p.max():.3f}] std={p.std():.4f}")
    print(f"rr   : {r.shape} range=[{r.min():.3f}, {r.max():.3f}]")
