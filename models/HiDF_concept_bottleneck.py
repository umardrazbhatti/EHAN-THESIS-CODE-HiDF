"""
models/HiDF_concept_bottleneck.py — Phase 26 Concept Slot Bottleneck.

Adds a parallel interpretable head to the EAHN classifier:

  Transformer Q (B, T*N, d)
        │
        ├─ K=8 learned slot queries → soft-attention over (T*N) positions
        │   → K pooled vectors (B, K, d)
        │   → K scalar concept scores (B, K)
        │   → Linear(K → 1)            ─── CBM logit
        │
        └─ existing M_t + temporal_gate path ── main logit

  final_logit = sigmoid(blend) * main_logit + (1 - sigmoid(blend)) * cbm_logit

WHY THIS IMPROVES FAITHFULNESS
  The standard classifier reads a single 256-D pooled vector and predicts;
  there's no architectural reason for any specific spatial cell to be
  "important" — the classifier can rebalance arbitrary linear combinations.

  CBM forces prediction through K=8 scalar bottlenecks.  Each slot attention
  map is therefore exposed as "the evidence behind concept k."  Since the
  classifier can only see those K scalars (no other path), the slot
  attention IS causally coupled to the prediction.

  Slot diversity loss (mean off-diagonal cosine similarity between slot
  attention vectors) keeps the K slots from collapsing to the same region
  — the bottleneck must distribute its evidence across distinct concepts.

  Net effect on the faithfulness metric (Spearman corr between intrinsic
  M_t and gradient saliency): the model now MUST route gradient through
  the same regions M_t highlights, otherwise CBM concepts have no signal.

PARAMETER COST
  slot_q: (K, d) = 8 * 256 = 2048
  slot_v: (K, d) = 8 * 256 = 2048
  fc.weight: (1, K)        = 8
  blend:                   = 1
  ─────────────────────────
  total: ~4,105 params (0.02% of EfficientNet-B4 backbone)

COMPUTE COST
  Slot attention: B * K * (T*N) * d  flops  =  2 * 8 * 784 * 256 ≈ 3.2M MACs
  Pooling:        B * K * (T*N) * d  flops  ≈ 3.2M MACs
  Concept scores: B * K * d                ≈ 4K  MACs
  Total: ≈ 6.4M MACs/forward = 1-2% of full EAHN forward.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConceptSlotBottleneck(nn.Module):
    """K learned concept slots that soft-attend over transformer features.

    Forward
    -------
    Q : (B, L, d) where L = T*N
        Temporal-context features from the transformer.

    Returns
    -------
    cbm_logit       : (B,)       scalar prediction through K-concept bottleneck
    concept_scores  : (B, K)     scalar activation per concept
    slot_attn       : (B, K, L)  per-slot soft attention over positions
                                 (sums to 1 along L per slot — used for
                                 faithfulness diagnostics and slot diversity loss)
    blend_value     : float      sigmoid(blend) — fraction of MAIN logit in
                                 the combined output; logged for diagnostics
    """

    def __init__(self, d_model: int = 256, num_slots: int = 8):
        super().__init__()
        self.K = int(num_slots)
        self.d = int(d_model)

        # K learned query vectors — each defines one concept
        self.slot_q = nn.Parameter(torch.randn(self.K, self.d) * 0.02)
        # K learned value vectors — projects each pooled slot vector to a scalar
        self.slot_v = nn.Parameter(torch.randn(self.K, self.d) * 0.02)
        # Final classifier on K concept scores
        self.fc = nn.Linear(self.K, 1)
        # Learnable blend between main classifier logit and CBM logit.
        # sigmoid(0) = 0.5 → 50/50 blend at init.
        self.blend = nn.Parameter(torch.tensor(0.0))

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, Q: torch.Tensor):
        """Q: (B, L, d). Returns (cbm_logit, concept_scores, slot_attn, blend_val)."""
        B, L, d = Q.shape
        assert d == self.d, f"Q feature dim {d} != cbm.d {self.d}"

        # Scaled dot-product attention: (B, K, L)
        # logits[b, k, l] = (Q[b, l, :] · slot_q[k, :]) / sqrt(d)
        attn_logits = torch.einsum("bld,kd->bkl", Q, self.slot_q) / (d ** 0.5)
        slot_attn   = F.softmax(attn_logits, dim=-1)           # (B, K, L), sums to 1 over L

        # Slot-pooled features: each slot gets its own d-dim vector
        # slot_pool[b, k, :] = sum_l slot_attn[b, k, l] * Q[b, l, :]
        slot_pool = torch.einsum("bkl,bld->bkd", slot_attn, Q) # (B, K, d)

        # Scalar concept score per slot
        # concept_scores[b, k] = slot_pool[b, k, :] · slot_v[k, :]
        concept_scores = torch.einsum("bkd,kd->bk", slot_pool, self.slot_v)  # (B, K)

        # CBM logit through K-scalar bottleneck
        cbm_logit = self.fc(concept_scores).squeeze(-1)        # (B,)

        blend_val = float(torch.sigmoid(self.blend).item())

        return cbm_logit, concept_scores, slot_attn, blend_val


def slot_diversity_loss(slot_attn: torch.Tensor) -> torch.Tensor:
    """Mean off-diagonal cosine similarity between slot attention vectors.

    slot_attn : (B, K, L)  — per-slot attention distributions over L positions
    Returns scalar in [0, 1] (clamped).  Minimised → slots attend to
    different positions.

    Implementation: per-sample compute KxK cosine sim of (slot_attn[b])
    rows; zero the diagonal; average.  Result averaged over batch.

    Why cosine on attention vectors (not concepts):
      The K concept *scores* can be different even if K slots look at the
      same place (different slot_v weights).  Diversity should pressure
      the SPATIAL distribution to be different so the K slots aren't
      redundant heatmaps of the same region.
    """
    B, K, L = slot_attn.shape
    # Normalise along position dimension
    sa_norm = F.normalize(slot_attn, dim=-1)                  # (B, K, L)
    # KxK cosine similarity per sample
    cos = sa_norm @ sa_norm.transpose(-1, -2)                  # (B, K, K)
    # Zero the diagonal
    eye = torch.eye(K, dtype=torch.bool, device=cos.device)
    off = cos.masked_fill(eye, 0.0)
    # Mean off-diagonal element per sample, then mean over batch
    # K*(K-1) off-diagonal entries per sample
    mean_off = off.sum(dim=(-1, -2)) / max(K * (K - 1), 1)
    return mean_off.mean().clamp(min=0.0)
