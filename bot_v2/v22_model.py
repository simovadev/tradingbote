"""v22_model.py - Modele DL V22 PROPRE.

Fix les bugs V21 #154-166 :
- LayerNorm au lieu BatchNorm1d (#154) -> stable en inference batch=1
- ~1.2M params au lieu de 9M (#157) -> moins overfit pour 816 samples
- BCEWithLogitsLoss (#160) -> stabilite numerique
- Pas de head_vol (#159)
- Init unifie (#158)
- Pas de dropout cumule (#156) : 1 seul dropout en fin de transformer
- LayerNorm sur refs concatenees (#162)
"""
from __future__ import annotations
import torch
import torch.nn as nn
import math


class V22Model(nn.Module):
    """Architecture compacte :
    - Encoder M5 sequence (60 bougies, 4 features OHLC)
      -> 1D Conv stem (extract local patterns sans avg pooling)
      -> 2-layer Transformer encoder (LayerNorm, multi-head attention)
      -> Mean pooling temporel
    - Tabular MLP pour les 12 features ICT
    - Concat + head binaire (1 logit, BCEWithLogitsLoss)
    """

    def __init__(self, n_feat_tab: int = 12, d_model: int = 96,
                 n_heads: int = 4, n_layers: int = 2, seq_len: int = 60,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        # Stem CNN sur (B, 4, seq_len) -> (B, d_model, seq_len)
        self.stem = nn.Sequential(
            nn.Conv1d(4, d_model // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model // 2, d_model, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # Positional encoding
        self.pos_enc = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(self.pos_enc, std=0.02)

        # Transformer encoder LayerNorm (#154 fix)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, dropout=dropout,
            activation="gelu", batch_first=True,
            norm_first=True,   # pre-norm = plus stable
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Norm finale + pool
        self.final_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Tabular MLP
        self.tab_mlp = nn.Sequential(
            nn.Linear(n_feat_tab, 32),
            nn.GELU(),
            nn.LayerNorm(32),
            nn.Linear(32, 32),
            nn.GELU(),
        )

        # Head : concat (d_model + 32) -> 1 logit
        self.head = nn.Sequential(
            nn.Linear(d_model + 32, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, 1),
        )

        # Init unifie (#158 fix)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, seq, feat):
        """
        seq : (B, seq_len, 4) -- OHLC normalises
        feat : (B, n_feat_tab) -- features tabulaires
        Retourne logit (B, 1)
        """
        # Stem CNN (besoin (B, 4, seq_len))
        x = seq.transpose(1, 2)            # (B, 4, T)
        x = self.stem(x)                    # (B, d_model, T)
        x = x.transpose(1, 2)               # (B, T, d_model)

        # Pos encoding
        x = x + self.pos_enc

        # Transformer
        x = self.transformer(x)             # (B, T, d_model)
        x = self.final_norm(x)
        # Mean pool temporel
        x = x.mean(dim=1)                   # (B, d_model)
        x = self.dropout(x)

        # Tabular
        f = self.tab_mlp(feat)              # (B, 32)

        # Concat + head
        z = torch.cat([x, f], dim=-1)       # (B, d_model + 32)
        logit = self.head(z)                # (B, 1)
        return logit


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = V22Model()
    print(f"V22Model params : {count_params(m):,}")
    # Test forward
    seq = torch.randn(8, 60, 4)
    feat = torch.randn(8, 12)
    out = m(seq, feat)
    print(f"Output shape : {out.shape}, mean={out.mean():.3f}, std={out.std():.3f}")
