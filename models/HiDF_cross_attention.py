"""
models/cross_attention.py — Cross-Attention Fusion with learnable temperature.

Returns (M_t, attn_pool):
  M_t      : (B, T, h, w)  intrinsic explanation maps (softmax probability distributions)
  attn_pool : (B, d_model)  attention-weighted spatial pooling for classifier gradient path

The attn_pool → classifier residual path ensures that L_cls gradients flow back
through the attention weights into M_t (fixes the attention-collapse bug).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFusion(nn.Module):
    def __init__(self, d_model: int = 256, num_heads: int = 8,
                 attn_temp_init: float = 0.0):
        super().__init__()
        self.d_model   = d_model
        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads
        self.scale     = math.sqrt(self.head_dim)

        # Learnable temperature; τ = exp(log_temp). Phase 8 default: τ = 1.0
        # (was hardcoded log(4.0)=1.386 → τ=4, which over-smoothed attention
        # rows from initialization). τ can still grow during training.
        self.log_temp = nn.Parameter(torch.tensor(float(attn_temp_init)))

        self.q_proj   = nn.Linear(d_model, d_model, bias=False)
        self.k_proj   = nn.Linear(d_model, d_model, bias=False)
        self.v_proj   = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        Q: torch.Tensor,   # (B, T, L, d_model)  temporal queries
        S: torch.Tensor,   # (B, T, L, d_model)  spatial keys/values
    ):
        B, T, L, d = Q.shape
        h = w = int(math.sqrt(L))   # h=w=7 for 224px input, stride-32 backbone

        Q_flat = Q.reshape(B * T, L, d)
        S_flat = S.reshape(B * T, L, d)

        Qp = self.q_proj(Q_flat)    # (B·T, L, d)
        Kp = self.k_proj(S_flat)
        Vp = self.v_proj(S_flat)

        # Temperature-scaled attention
        tau    = torch.exp(self.log_temp).clamp(min=0.5, max=10.0)
        scores = torch.bmm(Qp, Kp.transpose(-2, -1)) / (self.scale * tau)  # (B·T, L, L)
        A      = F.softmax(scores, dim=-1)  # softmax over key dimension

        # Phase 8 CHANGE 1 (root-cause fix): M_flat is already a probability
        # distribution — it is the column-mean of a row-stochastic matrix, so
        # each entry lies in [0,1] and they sum to 1 per frame. The previous
        # phase-7 code applied a SECOND softmax to this distribution, which
        # exponentially compresses it toward the uniform centroid (1/L per cell)
        # and structurally pins inter_sample_cosine above 0.95. Removing that
        # softmax lets M_t carry real spatial signal.
        M_flat = A.mean(dim=-2)                        # (B·T, L), already sums to 1
        M_t    = M_flat.reshape(B, T, h, w)            # use directly, NO softmax

        # attn_pool: use M_flat as weights (sums to 1 per frame → proper weighted pool).
        # Classifier gradient now flows through learned M_flat → attention parameters.
        W = M_flat.unsqueeze(-1)                       # (B·T, L, 1)
        S_pool    = (W * Vp).sum(dim=1)                # (B·T, d)
        attn_pool = S_pool.reshape(B, T, d).mean(dim=1) # (B, d)

        return M_t, attn_pool
