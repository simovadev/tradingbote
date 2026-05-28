"""V21 MONSTER : Transformer profond + cross-attention robuste + multi-task.

Objectif : bot ULTRA fiable pour toutes conditions de marche.
- Self-discovery depuis bougies brutes (pas de features hand-crafted)
- Cross-attention sophistique : focus query, 5 refs keys/values
- Multi-task : proba + RR optimal + volatility estimee
- Init Xavier + LayerNorm pre-attention (stable)
- 7-15M params (configurable)
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


class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class CNNFront(nn.Module):
    """CNN front-end : downsample + feature extraction avant Transformer.

    Reduit la longueur de la sequence pour rendre l'attention tractable.
    Ex : 240 -> 60 ou 60 -> 30.
    """
    def __init__(self, in_ch=5, d_model=256, n_layers=3, downsample=4):
        super().__init__()
        layers = []
        ch = in_ch
        for i in range(n_layers):
            layers.append(CausalConv1d(ch, d_model, kernel=5, dilation=2**i))
            layers.append(nn.BatchNorm1d(d_model))
            layers.append(nn.GELU())
            ch = d_model
        self.cnn = nn.Sequential(*layers)
        self.pool = nn.AvgPool1d(kernel_size=downsample, stride=downsample)

    def forward(self, x):
        # x: (B, T, C)
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = self.pool(x)
        return x.transpose(1, 2)  # (B, T//downsample, d_model)


class TimeframeBlock(nn.Module):
    """CNN front + Transformer encoder + CLS token pooling."""
    def __init__(self, in_ch=5, d_model=256, n_heads=8, n_layers=4,
                 max_len=512, dropout=0.2, downsample=4, init_gain=0.5):
        super().__init__()
        self.cnn_front = CNNFront(in_ch, d_model, n_layers=3, downsample=downsample)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))  # init zero, stable
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.pe = SinusoidalPE(d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(d_model)
        # Init poids stable (Xavier scaled)
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=init_gain)

    def forward(self, x):
        # x: (B, T, in_ch)
        B = x.size(0)
        x = self.cnn_front(x)  # (B, T', d_model)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pe(x)
        x = self.transformer(x)
        return self.out_norm(x[:, 0])


class CrossAssetBlock(nn.Module):
    """Cross-attention focus <- 5 refs avec residual + ffn."""
    def __init__(self, d_model=256, n_heads=8, dropout=0.15):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, focus, refs):
        # focus: (B, d_model), refs: (B, n_ref, d_model)
        q = focus.unsqueeze(1)
        attn_out, _ = self.attn(q, refs, refs)
        focus = self.norm1(focus + self.dropout(attn_out.squeeze(1)))
        focus = self.norm2(focus + self.dropout(self.ffn(focus)))
        return focus


class V21MonsterNet(nn.Module):
    """V21 MONSTER : self-discovery + cross-asset deep.

    Inputs :
        focus_m1   (B, 240, 5)
        focus_m15  (B, 60, 5)
        focus_h1   (B, 30, 5)
        focus_d1   (B, 15, 5)
        ref_m15    (B, 5, 60, 5)
    Outputs :
        proba_win : (B,)
        rr_pred   : (B,)
        vol_pred  : (B,) - auxiliary head (volatility estimee)
    """
    def __init__(self, n_ref=5, d_model=192, n_heads=6,
                 n_layers_m1=4, n_layers_m15=3, n_layers_h1=2, n_layers_d1=2,
                 n_layers_ref=3, n_cross=2, dropout=0.2):
        super().__init__()
        self.n_ref = n_ref
        self.d_model = d_model

        # Encoders focus (4 TFs)
        self.enc_m1 = TimeframeBlock(5, d_model, n_heads, n_layers_m1, max_len=80, dropout=dropout, downsample=4)  # 240/4=60
        self.enc_m15 = TimeframeBlock(5, d_model, n_heads, n_layers_m15, max_len=40, dropout=dropout, downsample=2)  # 60/2=30
        self.enc_h1 = TimeframeBlock(5, d_model, n_heads, n_layers_h1, max_len=40, dropout=dropout, downsample=2)  # 30/2=15
        self.enc_d1 = TimeframeBlock(5, d_model, n_heads, n_layers_d1, max_len=20, dropout=dropout, downsample=1)  # 15

        # Encoder ref partage
        self.enc_ref = TimeframeBlock(5, d_model, n_heads, n_layers_ref, max_len=40, dropout=dropout, downsample=2)

        # Fusion focus 4 TFs -> 1 vecteur
        self.focus_fusion = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

        # Cross-attention blocks (2 layers)
        self.cross_blocks = nn.ModuleList([
            CrossAssetBlock(d_model, n_heads, dropout=dropout)
            for _ in range(n_cross)
        ])

        # Heads multi-task
        self.head_proba = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self.head_rr = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Softplus(),
        )
        self.head_vol = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Softplus(),
        )

    def forward(self, focus_m1, focus_m15, focus_h1, focus_d1, ref_m15):
        x_m1 = self.enc_m1(focus_m1)
        x_m15 = self.enc_m15(focus_m15)
        x_h1 = self.enc_h1(focus_h1)
        x_d1 = self.enc_d1(focus_d1)
        focus = self.focus_fusion(torch.cat([x_m1, x_m15, x_h1, x_d1], dim=-1))

        # Encode refs
        B, n_ref, T, C = ref_m15.shape
        ref_flat = ref_m15.reshape(B * n_ref, T, C)
        ref_emb_flat = self.enc_ref(ref_flat)
        refs = ref_emb_flat.reshape(B, n_ref, -1)

        # Cross-attention deep
        for block in self.cross_blocks:
            focus = block(focus, refs)

        proba = torch.sigmoid(self.head_proba(focus).squeeze(-1))
        rr = self.head_rr(focus).squeeze(-1)
        vol = self.head_vol(focus).squeeze(-1)
        return proba, rr, vol


def v21_monster_loss(proba, rr_pred, vol_pred, target_win, target_rr,
                      w_rr=0.1, w_vol=0.05):
    """BCE + RR MSE (sur WINs) + vol MSE (auxiliary)."""
    eps = 1e-7
    bce = -(target_win * torch.log(proba + eps) + (1 - target_win) * torch.log(1 - proba + eps)).mean()
    win_mask = target_win > 0.5
    if win_mask.any():
        rr_mse = F.mse_loss(rr_pred[win_mask], target_rr[win_mask])
    else:
        rr_mse = torch.tensor(0.0, device=proba.device)
    # Vol target = abs(target_rr) (proxy : magnitude expected)
    vol_target = torch.abs(target_rr)
    vol_mse = F.mse_loss(vol_pred, vol_target)
    return bce + w_rr * rr_mse + w_vol * vol_mse, bce, rr_mse


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = V21MonsterNet(d_model=192, n_heads=6, n_cross=2)
    n = count_params(model)
    print(f"V21 MONSTER : {n:,} params (~{n/1e6:.1f}M)")
    B = 4
    p, r, v = model(torch.randn(B,240,5), torch.randn(B,60,5),
                     torch.randn(B,30,5), torch.randn(B,15,5),
                     torch.randn(B,5,60,5))
    print(f"proba: {p.shape} std={p.std():.4f}")
    print(f"rr   : {r.shape} std={r.std():.4f}")
    print(f"vol  : {v.shape} std={v.std():.4f}")
