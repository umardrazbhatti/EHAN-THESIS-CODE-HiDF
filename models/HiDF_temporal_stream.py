"""
models/temporal_stream.py — 4-layer Transformer Encoder over spatial tokens.

Stores per-layer attention weights for attention rollout XAI.
Returns temporal queries Q (B, T*N, d_model) and CLS embedding cls_out (B, d_model).
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as _ckpt
import math


class CustomTransformerEncoderLayer(nn.Module):
    """Standard pre-norm Transformer layer that stores the last attention matrix."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.drop1   = nn.Dropout(dropout)
        self.drop2   = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)
        self.act     = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.attn_weights: torch.Tensor = None   # stored for rollout

    def forward(self, x: torch.Tensor,
                src_key_padding_mask=None) -> torch.Tensor:
        # Self-attention with residual
        attn_out, self.attn_weights = self.self_attn(
            x, x, x,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=True,   # average over heads → (B, N, N)
        )
        x = self.norm1(x + self.drop1(attn_out))
        # Feed-forward with residual
        ff_out = self.linear2(self.dropout(self.act(self.linear1(x))))
        x = self.norm2(x + self.drop2(ff_out))
        return x


class TemporalStream(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 785,
    ):
        super().__init__()
        self.d_model = d_model
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        self.layers = nn.ModuleList([
            CustomTransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                activation="gelu",
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.layer_attention_weights: list = []
        self._grad_ckpt = False

    def enable_gradient_checkpointing(self):
        self._grad_ckpt = True

    def forward(self, spatial_tokens: torch.Tensor):
        """
        Args:
            spatial_tokens : (B, T*N, d_model)
        Returns:
            Q      : (B, T*N, d_model)   temporal queries (spatial positions only)
            cls_out: (B, d_model)        CLS token output
        """
        B, seq_len, _ = spatial_tokens.shape
        cls = self.cls_token.expand(B, 1, -1)
        x   = torch.cat([cls, spatial_tokens], dim=1)          # (B, 1+T*N, d)

        pos = self.pos_embed[:, :1 + seq_len, :]
        x   = x + pos

        self.layer_attention_weights = []
        for layer in self.layers:
            if self._grad_ckpt and self.training:
                x = _ckpt.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
            self.layer_attention_weights.append(layer.attn_weights)  # (B, N+1, N+1)

        x = self.norm(x)
        cls_out = x[:, 0, :]     # (B, d_model)
        Q       = x[:, 1:, :]    # (B, T*N, d_model)
        return Q, cls_out
