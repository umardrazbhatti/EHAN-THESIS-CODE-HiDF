"""
models/HiDF_concept_bottleneck.py — Phase 26+27 Concept Slot Bottleneck.

PHASE 26 (parallel head — DEPRECATED): cbm was wired alongside the main
classifier and blended via a learnable sigmoid gate. Result: the model
escaped through the main path, M_t got decoupled from prediction, and
faithfulness_corr regressed (0.225 -> 0.079 over 8 epochs).

PHASE 27 (serial bottleneck): the CBM is the SOLE classifier
path. The main Linear(d -> 1) head still exists but is used only as an
auxiliary supervision signal (lambda_cbm_main_aux, small weight) and its
output is exposed as `main_logit` for diagnostics — it is NOT part of the
prediction `out.logit`. This forces all gradient through the K-concept
bottleneck and removes the escape hatch.

PHASE 28 (multiplicative coupling — DEAD): Q tokens scaled by
w = M_t ⊙ M_frame before the CBM.  Scaling token MAGNITUDES by a peaky
map starved the slot attention (782/784 near-zero keys soak up the
softmax mass), slot_pool collapsed to ~0, cbm_logit froze at the fc
bias and training never started (run 6-11-26 1300hrs: val AUC ~0.5 for
9 epochs, cls loss constant at 0.1838).

PHASE 29 (pooled-frame input + log-prior — current): the CBM reads the
M_t-pooled per-frame vectors and receives M_frame as a LOG-space
attention prior.  Both couplings are renormalised forms — pooling is a
convex combination, the prior shifts softmax logits — so magnitude
starvation is structurally impossible while M_t and M_frame remain
load-bearing for the prediction (P23 precedent: same pooled path gave
0.904 AUC / k1 1.64 under the old artifact-diluted protocol).

  Transformer Q (B, T, N, d)
        │
        ├─ attn_pool_per_frame[t] = Σ_n M_t[t,n] · Q[t,n]   (B, T, d)
        │       (spatial coupling: FORCED — suppressed tokens are absent)
        │
        └─ K slot queries -> soft-attention over T frames with
           attn_logits += log(M_frame)   (temporal coupling: prior)
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

    def forward(self, Q: torch.Tensor, prior: torch.Tensor = None):
        """Q: (B, L, d). Returns (cbm_logit, concept_scores, slot_attn, blend_val).

        prior : (B, L) optional — a probability distribution over the L
            positions (e.g. M_frame when L = T).  Added to the attention
            logits in LOG space, so it biases WHERE slots look while the
            softmax renormalises — slot_pool stays a convex combination of
            full-magnitude tokens regardless of how peaky the prior is.
            This is the Phase 29 coupling mechanism.  Phase 28's
            multiplicative input scaling is the cautionary tale: scaling
            token MAGNITUDES by a peaky map drove 782/784 keys to ~0, the
            softmax diluted the survivors, slot_pool collapsed to the fc
            bias, and the cls gradient died (run 6-11-26 1300hrs: cls
            frozen at 0.1838 for 9 epochs).  A log-space prior cannot
            reproduce that failure: softmax output always sums to 1 over
            real tokens.
        """
        B, L, d = Q.shape
        assert d == self.d, f"Q feature dim {d} != cbm.d {self.d}"

        # Scaled dot-product attention: (B, K, L)
        # logits[b, k, l] = (Q[b, l, :] · slot_q[k, :]) / sqrt(d)
        attn_logits = torch.einsum("bld,kd->bkl", Q, self.slot_q) / (d ** 0.5)
        if prior is not None:
            # log-prior coupling: clamp keeps the bias finite even if the
            # prior has exact zeros (log(1e-6) ≈ -13.8 — strong but finite,
            # so the cls gradient can still resurrect a suppressed frame).
            attn_logits = attn_logits + prior.clamp(min=1e-6).log().unsqueeze(1)
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
