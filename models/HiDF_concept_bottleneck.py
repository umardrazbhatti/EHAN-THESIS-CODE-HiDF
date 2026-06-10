"""
models/HiDF_concept_bottleneck.py — Phase 26+27 Concept Slot Bottleneck.

PHASE 26 (parallel head — DEPRECATED): cbm was wired alongside the main
classifier and blended via a learnable sigmoid gate. Result: the model
escaped through the main path, M_t got decoupled from prediction, and
faithfulness_corr regressed (0.225 -> 0.079 over 8 epochs).

PHASE 27 (serial bottleneck — current): the CBM is the SOLE classifier
path. The main Linear(d -> 1) head still exists but is used only as an
auxiliary supervision signal (lambda_cbm_main_aux, small weight) and its
output is exposed as `main_logit` for diagnostics — it is NOT part of the
prediction `out.logit`. This forces all gradient through the K-concept
bottleneck and removes the escape hatch.

  Transformer Q (B, T*N, d)
        │
        └─ K learned slot queries -> soft-attention over (T*N) positions
            -> K pooled vectors (B, K, d)
            -> K scalar concept scores (B, K)
            -> Linear(K -> 1)            ─── cbm_logit = out.logit  (PRIMARY)

  attn_pool (B, d) -> Linear(d -> 1) -> main_logit  (DIAGNOSTIC AUX ONLY)

WHY SERIAL IS THE CORRECT DESIGN
  Phase 26 evidence: when both paths exist with a sigmoid blend, the
  classifier learns to mix them in whatever way minimises loss without
  caring whether M_t is informative.  Slot attention learns concepts but
  they live in a separate gradient lane.

  Serial means: prediction = f(K concept scores) only.  K concept scores
  are computed via slot attention over Q, which is gated by M_t.  So the
  gradient of prediction w.r.t. input pixels MUST flow through the slots
  and the M_t-gated tokens.  Faithfulness (corr between M_t and gradient
  saliency) is now coupled by construction.

PARAMETER COST  (K=12 for Phase 27)
  slot_q: (K, d) = 12 * 256 = 3072
  slot_v: (K, d) = 12 * 256 = 3072
  fc.weight: (1, K)        = 12
  fc.bias                   = 1
  ─────────────────────────
  total: ~6,160 params (0.03% of EfficientNet-B4 backbone)

COMPUTE COST  (K=12)
  Slot attention: B * K * (T*N) * d  =  2 * 12 * 784 * 256 ≈ 4.8M MACs
  Pooling:        B * K * (T*N) * d  ≈ 4.8M MACs
  Concept scores: B * K * d           ≈ 6k  MACs
  Total: ≈ 9.6M MACs/forward = 2-3% of full EAHN forward.

NOTE
  The `blend` parameter remains in the module for backwards compatibility
  (load checkpoints from Phase 26) but is NOT used when caller sets
  serial=True.  In serial mode out.cbm_blend just reports sigmoid(blend)
  for diagnostic continuity.
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
