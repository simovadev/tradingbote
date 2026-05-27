"""V20 : architecture DL ICT UNIVERSELLE (sans asset_embedding).

Difference cle vs V19 :
- PAS d'embedding par actif -> le model apprend ICT comme pattern universel
- Force la generalisation : un OB sur EURUSD est traite comme un OB sur XAUUSD
- Entraine sur 78 actifs Vantage (vs 28 dans V19)

Hypothese : si ICT est universel (ce que la validation adversariale V19 suggere),
retirer l'asset_emb force le model a apprendre la structure ICT pure plutot que
de memoriser les habitudes par actif.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """Conv1D causale (pas de future leak)."""
    def __init__(self, in_ch, out_ch, kernel=5, dilation=1):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, padding=0, dilation=dilation)

    def forward(self, x):
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class TimeframeEncoder(nn.Module):
    """Encode une sequence OHLCV brute en embedding."""
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


class V20ICTNet(nn.Module):
    """V20 : ICT universel sans asset_embedding.

    Inputs :
        m1_seq:   (B, 240, 5)
        m15_seq:  (B, 60, 5)
        h1_seq:   (B, 30, 5)
        ict_feat: (B, n_ict)
    Output :
        proba_win: (B,) - probabilite WIN [0, 1]
    """

    def __init__(self, n_ict_features=52,
                 m1_len=240, m15_len=60, h1_len=30,
                 ohlcv_ch=5, hidden=64, embed_dim=128):
        super().__init__()

        self.enc_m1 = TimeframeEncoder(ohlcv_ch, hidden, n_layers=4, output_dim=embed_dim)
        self.enc_m15 = TimeframeEncoder(ohlcv_ch, hidden, n_layers=3, output_dim=embed_dim)
        self.enc_h1 = TimeframeEncoder(ohlcv_ch, hidden, n_layers=2, output_dim=embed_dim)

        self.ict_branch = nn.Sequential(
            nn.Linear(n_ict_features, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.GELU(),
        )

        # Fusion : concat(m1, m15, h1, ict) -> head. PAS d'asset_emb.
        total_dim = embed_dim * 3 + 64
        self.head = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, m1_seq, m15_seq, h1_seq, ict_feat, asset_id=None):
        """asset_id ignore (gardé pour compat API V19)."""
        x_m1 = self.enc_m1(m1_seq)
        x_m15 = self.enc_m15(m15_seq)
        x_h1 = self.enc_h1(h1_seq)
        x_ict = self.ict_branch(ict_feat)

        x = torch.cat([x_m1, x_m15, x_h1, x_ict], dim=-1)
        logit = self.head(x).squeeze(-1)
        return torch.sigmoid(logit)


def ict_aware_loss(pred_proba, target, ict_feat,
                    daily_bias_idx=0, sweep_strength_idx=1, has_fvg_idx=2,
                    rr_idx=3, w_ict=0.5):
    """Loss BCE + penalites ICT (identique V19)."""
    eps = 1e-7
    bce = -(target * torch.log(pred_proba + eps)
            + (1 - target) * torch.log(1 - pred_proba + eps)).mean()

    high_conf = pred_proba > 0.6
    bias_violation = high_conf & (ict_feat[:, daily_bias_idx] < 0.5)
    sweep_violation = high_conf & (ict_feat[:, sweep_strength_idx] < 0.3)
    fvg_violation = high_conf & (ict_feat[:, has_fvg_idx] < 0.5)
    rr_violation = high_conf & (ict_feat[:, rr_idx] < 1.5)

    penalty = (bias_violation.float().mean()
               + sweep_violation.float().mean()
               + fvg_violation.float().mean()
               + rr_violation.float().mean()) / 4.0
    return bce + w_ict * penalty, bce, penalty


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = V20ICTNet(n_ict_features=52)
    n_params = count_params(model)
    print(f"V20 ICTNet : {n_params:,} parameters (vs V19 ~240k)")

    B = 4
    m1 = torch.randn(B, 240, 5)
    m15 = torch.randn(B, 60, 5)
    h1 = torch.randn(B, 30, 5)
    ict = torch.randn(B, 52)
    proba = model(m1, m15, h1, ict)
    print(f"Output : shape={proba.shape}, range=[{proba.min():.3f}, {proba.max():.3f}]")
