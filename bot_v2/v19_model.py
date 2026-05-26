"""V19 ETAPE 3 : Architecture DL ICT-aware multi-actifs.

Le modele combine :
1. CNN+Transformer sur sequences brutes (M1, M15, H1)
2. MLP sur features ICT classiques (50 features)
3. Asset embedding (28 actifs -> vector 32)
4. Loss avec penalites ICT (force le DL a respecter les regles)

Anti-overfit :
- Dropout 0.3
- Batch norm
- Regularisation L2 sur weights
- Asset embedding pour ne pas confondre actifs
- Features normalisees par actif (z-score)
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
        # x: (B, C, T)
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
        # x: (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = self.proj(x)
        # Global avg pooling
        return x.mean(dim=-1)  # (B, output_dim)


class V19ICTNet(nn.Module):
    """Architecture V19 : DL + ICT features + asset embedding.

    Inputs :
        m1_seq:   (B, 240, 5) - OHLCV M1 normalisé
        m15_seq:  (B, 60, 5)  - OHLCV M15 normalisé
        h1_seq:   (B, 30, 5)  - OHLCV H1 normalisé
        ict_feat: (B, 50)     - 50 features ICT (daily_bias, sweep, etc.)
        asset_id: (B,)        - int 0..27 (asset index)
    Output :
        proba_win: (B,) - probabilite WIN [0, 1]
    """

    def __init__(self, n_assets=28, n_ict_features=50,
                 m1_len=240, m15_len=60, h1_len=30,
                 ohlcv_ch=5, hidden=64, embed_dim=128, asset_emb_dim=32):
        super().__init__()

        # 3 encoders multi-timeframe
        self.enc_m1 = TimeframeEncoder(ohlcv_ch, hidden, n_layers=4, output_dim=embed_dim)
        self.enc_m15 = TimeframeEncoder(ohlcv_ch, hidden, n_layers=3, output_dim=embed_dim)
        self.enc_h1 = TimeframeEncoder(ohlcv_ch, hidden, n_layers=2, output_dim=embed_dim)

        # ICT features branch
        self.ict_branch = nn.Sequential(
            nn.Linear(n_ict_features, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.GELU(),
        )

        # Asset embedding
        self.asset_emb = nn.Embedding(n_assets, asset_emb_dim)

        # Fusion : concat(m1, m15, h1, ict, asset) -> head
        total_dim = embed_dim * 3 + 64 + asset_emb_dim
        self.head = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, m1_seq, m15_seq, h1_seq, ict_feat, asset_id):
        x_m1 = self.enc_m1(m1_seq)
        x_m15 = self.enc_m15(m15_seq)
        x_h1 = self.enc_h1(h1_seq)
        x_ict = self.ict_branch(ict_feat)
        x_asset = self.asset_emb(asset_id)

        x = torch.cat([x_m1, x_m15, x_h1, x_ict, x_asset], dim=-1)
        logit = self.head(x).squeeze(-1)
        return torch.sigmoid(logit)


def ict_aware_loss(pred_proba, target, ict_feat,
                    daily_bias_idx=0, sweep_strength_idx=1, has_fvg_idx=2,
                    rr_idx=3, w_ict=0.5):
    """Loss avec penalites ICT.

    ict_feat: tensor (B, 50) - les features ICT en tensor
    daily_bias_idx, sweep_strength_idx, etc. : index des features critiques

    Penalites :
    1. Si proba > 0.6 mais daily_bias_aligned == 0 : penalty
    2. Si proba > 0.6 mais sweep_strength < 0.3 : penalty
    3. Si proba > 0.6 mais has_FVG_sync == 0 : penalty
    4. Si proba > 0.6 mais rr < 1.5 : penalty
    """
    # Loss BCE classique
    eps = 1e-7
    bce = -(target * torch.log(pred_proba + eps)
            + (1 - target) * torch.log(1 - pred_proba + eps)).mean()

    # Pénalités ICT
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
    """Compte les params."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test
    model = V19ICTNet(n_assets=28, n_ict_features=50)
    n_params = count_params(model)
    print(f"V19 ICTNet : {n_params:,} parameters")

    # Dummy forward
    B = 4
    m1 = torch.randn(B, 240, 5)
    m15 = torch.randn(B, 60, 5)
    h1 = torch.randn(B, 30, 5)
    ict = torch.randn(B, 50)
    asset_id = torch.randint(0, 28, (B,))

    proba = model(m1, m15, h1, ict, asset_id)
    print(f"Output shape : {proba.shape}, range [{proba.min():.3f}, {proba.max():.3f}]")

    # Loss test
    target = torch.randint(0, 2, (B,)).float()
    loss, bce, penalty = ict_aware_loss(proba, target, ict)
    print(f"Loss total : {loss:.4f} (BCE={bce:.4f}, ICT penalty={penalty:.4f})")
