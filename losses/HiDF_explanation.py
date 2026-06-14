"""
losses/HiDF_explanation.py — L_exp + faithfulness utilities.

v3 patch — all-three-metrics fix:
  [mt_std]         loss_sharp now operates on M_t_logits (pre-softmax raw
                   scores) not on M_t (softmax). Softmax over 49 cells has a
                   hard ceiling of std≈0.141 — below the 0.15 threshold. Raw
                   logits have no ceiling. Caller (train_real) passes out.M_t_logits.

  [peak_mode_share] PeakSpreadLoss replaced with HardAttentionDiversityLoss.
                   Old loss used entropy of batch-average which could be
                   satisfied even under mode collapse. New loss computes
                   batch-level popularity per cell (how many samples peak at
                   each location) and penalises concentration — directly
                   attacks peak_mode_share. Based on UNITE CVPR-2025 AD-loss.

  [fake_acc]       No change here — handled via focal_alpha in train_real.

  [faithfulness]   Unchanged from v2 — B-pass no_grad removed in train_real.
"""

import math as _math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from torchvision.transforms import functional as TF


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class ExplanationLossOutput:
    loss:             torch.Tensor
    l_h:              float   # entropy term
    l_tv:             float   # total-variation term
    l_div:            float   # inter-sample JS-divergence term
    inter_sample_sim: float   # mean pairwise cosine similarity (diagnostic)


# ── Main explanation loss (entropy + TV + JS-div) ─────────────────────────────

class ExplanationLoss(nn.Module):
    """Weakly-supervised explanation loss — no GT masks required."""
    def __init__(self, alpha: float = 0.2, beta: float = 0.5,
                 diversity_weight: float = 4.0):
        super().__init__()
        self.alpha            = alpha
        self.beta             = beta
        self.diversity_weight = diversity_weight

    def forward(self, M_t: torch.Tensor) -> ExplanationLossOutput:
        """M_t : (B, T, h, w) — softmax maps, sums to 1 per (b,t)"""
        B, T, h, w = M_t.shape
        loss = M_t.new_zeros(1).squeeze()
        l_h_acc  = 0.0
        l_tv_acc = 0.0

        for i in range(B):
            m_avg = M_t[i].mean(0)   # (h, w)
            m_flat  = m_avg.clamp(1e-8, 1 - 1e-8).flatten()
            entropy = -(m_flat * m_flat.log()).sum()
            tv_h = (M_t[i, :, :, 1:] - M_t[i, :, :, :-1]).abs().mean()
            tv_w = (M_t[i, :, 1:, :] - M_t[i, :, :-1, :]).abs().mean()
            tv   = tv_h + tv_w
            loss     = loss + (self.alpha * entropy + self.beta * tv)
            l_h_acc  += entropy.item()
            l_tv_acc += tv.item()

        loss = loss / B

        # ── Inter-sample JS-divergence ─────────────────────────────────────────
        N   = B * T
        eye = torch.eye(N, dtype=torch.bool, device=M_t.device)
        n_pairs = N * (N - 1)
        eps = 1e-8
        P = M_t.reshape(N, h * w) + eps
        P = P / P.sum(dim=-1, keepdim=True)
        log_P   = P.log()
        P_i     = P.unsqueeze(1)
        P_j     = P.unsqueeze(0)
        M_mix   = 0.5 * (P_i + P_j)
        log_M   = M_mix.log()
        log_P_i = log_P.unsqueeze(1)
        log_P_j = log_P.unsqueeze(0)
        kl_im = (P_i * (log_P_i - log_M)).sum(dim=-1)
        kl_jm = (P_j * (log_P_j - log_M)).sum(dim=-1)
        js_matrix = 0.5 * (kl_im + kl_jm)
        js_off = js_matrix.masked_fill(eye, 0.0)
        mean_js = js_off.sum() / max(n_pairs, 1)
        log2        = _math.log(2.0)
        l_div_tensor = (log2 - mean_js).clamp_min(0.0)
        loss = loss + self.diversity_weight * l_div_tensor

        # Cosine similarity diagnostic
        flat = M_t.reshape(N, h * w)
        flat_n = flat / (flat.norm(dim=-1, keepdim=True) + 1e-8)
        cos_matrix = flat_n @ flat_n.T
        inter_sample_sim = float(
            cos_matrix.masked_fill(eye, 0.0).sum().item() / (n_pairs + 1e-8)
        )

        return ExplanationLossOutput(
            loss=loss,
            l_h=l_h_acc / max(B, 1),
            l_tv=l_tv_acc / max(B, 1),
            l_div=float(l_div_tensor.item()),
            inter_sample_sim=inter_sample_sim,
        )


# ── Hard Attention Diversity Loss (replaces PeakSpreadLoss) ──────────────────

class HardAttentionDiversityLoss(nn.Module):
    """UNITE-style attention diversity loss (CVPR 2025).

    Directly penalises batch-level popularity concentration — the exact
    quantity that peak_mode_share measures.

    For each spatial cell, compute how much "probability mass" it receives
    across all samples in the batch. Penalise when one cell monopolises
    attention (Herfindahl concentration on batch popularity).

    With B=2 and temperature=0.05 this approaches the hard argmax behaviour,
    directly attacking the peak_mode_share metric.

    peak_mode_share = fraction of batch samples sharing the argmax cell.
    This loss = sum_c (popularity_c)^2 where popularity_c ∝ sum_b M_t[b,c].
    Minimised when every sample peaks at a different cell (popularity uniform).
    """
    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, M_t: torch.Tensor) -> torch.Tensor:
        """
        M_t : (B, T, h, w) — softmax maps
        Returns scalar. Higher = more concentrated = worse.
        """
        B, T, h, w = M_t.shape
        # Time-average per sample → (B, h*w)
        m_avg = M_t.mean(dim=1).reshape(B, h * w)

        # Soft-argmax popularity: sharpen each sample's map then sum across batch
        # temperature=0.05 → very close to hard argmax, directly mimics peak_mode_share
        p_sharp = F.softmax(m_avg / self.temperature, dim=1)  # (B, h*w)

        # Sum across batch: how popular is each cell? (h*w,)
        popularity = p_sharp.sum(dim=0)  # (h*w,), sums to B

        # Normalise so popularity sums to 1, then Herfindahl concentration
        popularity = popularity / (popularity.sum() + 1e-8)

        # Herfindahl: sum of squares. Uniform → 1/(h*w). One-cell → 1.0.
        concentration = (popularity ** 2).sum() * (h * w)  # scaled: uniform→1

        return concentration


# ── Sharpness loss on raw logits (fixes mt_std ceiling) ──────────────────────

def sharpness_loss(M_t_logits: torch.Tensor) -> torch.Tensor:
    """Sharpness loss on pre-softmax logits — bounded to [-1, 0].

    WHY LOGITS NOT SOFTMAX:
      Softmax over 49 cells has a hard std ceiling of ≈0.141, below the 0.15
      threshold. Raw logits have no ceiling so they can signal real sharpness.

    WHY BOUNDED:
      Raw logit std grows without limit as the conv learns (observed: 0.85 at
      init, growing to 2000+ by E2 causing fp16 overflow → NaN). We use tanh
      to squash std into [0,1] then negate, giving loss in [-1, 0].
      - Uniform map  → std ≈ 0 → tanh(0) = 0   → loss = 0  (worst)
      - Sharp map    → std → ∞ → tanh(∞) = 1   → loss = -1 (best)
      This makes lambda_sharp=0.15 safe regardless of logit scale.

    M_t_logits : (B, T, h, w) — raw scores from EarlyAttnHead before softmax.
    """
    B, T, h, w = M_t_logits.shape
    flat = M_t_logits.reshape(B * T, h * w)
    # centre per-map so std isn't affected by mean offset
    flat = flat - flat.mean(dim=1, keepdim=True)
    std_per_map = flat.std(dim=1)          # (B*T,), unbounded positive
    # squash to [0,1] with tanh, scale so typical init std≈0.85 → tanh(0.85)≈0.69
    sharpness = torch.tanh(std_per_map)    # (B*T,), in [0, 1)
    return -sharpness.mean()               # in [-1, 0]; minimise → maximise sharpness


# ── Phase 21 utilities ────────────────────────────────────────────────────────

def _gaussian_blur_5d(x: torch.Tensor,
                      kernel_size: int = 21,
                      sigma: float = 10.0) -> torch.Tensor:
    B, T, C, H, W = x.shape
    x_flat = x.reshape(B * T, C, H, W)
    blurred = TF.gaussian_blur(
        x_flat,
        kernel_size=[kernel_size, kernel_size],
        sigma=[sigma, sigma],
    )
    return blurred.reshape(B, T, C, H, W)


def build_bottlenecked_input(x: torch.Tensor,
                              M_t: torch.Tensor,
                              blur_kernel: int = 21,
                              blur_sigma: float = 10.0,
                              peak_floor: float = 0.0,
                              hard_topk_frac: float = 0.0,
                              invert: bool = False) -> torch.Tensor:
    """Construct an M_t-gated input at image resolution.

    x   : (B, T, 3, H, W)
    M_t : (B, T, h, w) — softmax map (used here, not logits)

    Gradient path: loss_faith → logits_B → model(x_b) → x_b → M_norm → M_t → EarlyAttnHead

    Phase 22 — peak_floor fix:
      Old behaviour (peak_floor=0.0): M_norm = M_up / M_up.max()
        A diffuse softmax (peak≈0.10) divides by its own max → M_norm ≈ 0.5–1.0
        across ~20 cells → bottleneck barely blurs → logits_B ≈ logits_A → KL ≈ 0.

      New behaviour (peak_floor=0.25): denominator = max(M_up.max(), 0.25)
        Same diffuse map (peak=0.10 < 0.25) → M_norm.max() ≈ 0.10/0.25 = 0.40
        → most of the image is blurred (weight ≥ 0.60) → logits_B degrades
        → KL > 0 → real gradient reaches EarlyAttnHead.
        The model must concentrate mass above 0.25 to recover logits.

    Phase 25 — hard_topk_frac (insertion-AUC training):
      When hard_topk_frac > 0 (typical: 0.20), build a binary top-K mask of the
      most-attended (hard_topk_frac * H * W) pixels.  Forward uses the binary
      mask; backward propagates as if mask = soft M_up (straight-through
      estimator).  This makes loss_ins / loss_faith depend on a TRUE top-K
      subset (the same subset the insertion metric uses at evaluation time),
      so optimising those losses directly optimises the metric.

      hard_topk_frac=0.0 (default) preserves Phase 22/24 behaviour.

    Phase 30 — invert (deletion / necessity pass):
      invert=False (default): x_b = M_norm·x + (1-M_norm)·blur — KEEP the
        attended region, blur the rest (sufficiency: B-pass).
      invert=True:            x_d = M_norm·blur + (1-M_norm)·x — ERASE the
        attended region, keep the rest (necessity: D-pass).  Used by
        loss_del, which demands the prediction fall toward "real" once the
        M_t-marked evidence is removed.  Run 6-12-26 0020hrs showed why
        sufficiency alone is not enough: the model classified fine from the
        top region (loss_ins 0.14) but ALSO from everything else (deletion
        AUC 0.502, worse than the 0.436 random control) — evidence was
        redundant and the maps carried no necessity information.  The
        degenerate solution "make M_norm small everywhere so x_d ≈ x" is
        blocked by the same peak_floor that protects the B-pass: a diffuse
        map gets divided by the 0.25 floor, which CRUSHES M_norm in the
        B-pass and destroys loss_ins — the two passes pin the map from
        both sides.
    """
    B, T, C, H, W = x.shape
    M_up = F.interpolate(
        M_t.reshape(B * T, 1, M_t.shape[-2], M_t.shape[-1]),
        size=(H, W), mode="bilinear", align_corners=False,
    ).reshape(B, T, 1, H, W)

    if hard_topk_frac and hard_topk_frac > 0.0:
        # Hard top-K binary mask with straight-through estimator.
        # Forward: mask is exactly the top K pixels (matches insertion metric).
        # Backward: gradient flows as if mask = M_up_soft (so M_t learns from
        # which-pixels-helped-classification signal).
        K = max(1, int(float(hard_topk_frac) * H * W))
        flat       = M_up.reshape(B, T, H * W)
        # threshold = K-th largest value per (b, t)
        threshold  = flat.topk(K, dim=-1).values[..., -1:]                 # (B, T, 1)
        M_hard     = (flat >= threshold).to(flat.dtype)                    # (B, T, H*W) binary
        # Soft path for backward — normalise M_up to [0, 1] so it has the same scale
        M_soft     = (flat / flat.amax(dim=-1, keepdim=True).clamp(min=1e-8)).clamp(0.0, 1.0)
        # Straight-through: forward=M_hard, backward=M_soft
        M_pass     = M_hard + (M_soft - M_soft.detach())
        M_norm     = M_pass.reshape(B, T, 1, H, W)
    else:
        M_peak = M_up.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
        if peak_floor and peak_floor > 0.0:
            floor_t = torch.as_tensor(peak_floor, device=M_up.device, dtype=M_up.dtype)
            M_peak  = torch.maximum(M_peak, floor_t)
        M_norm = (M_up / M_peak).clamp(0.0, 1.0)

    with torch.no_grad():
        x_blur = _gaussian_blur_5d(x.detach(), blur_kernel, blur_sigma)
    if invert:
        # Phase 30 D-pass: erase the attended region, keep the rest.
        x_b = M_norm * x_blur + (1.0 - M_norm) * x
    else:
        x_b = M_norm * x + (1.0 - M_norm) * x_blur
    return x_b


def faithfulness_loss(logits_A: torch.Tensor,
                       logits_B: torch.Tensor) -> torch.Tensor:
    """One-way KL: sg(A) as target, B as prediction."""
    pA = torch.sigmoid(logits_A.detach()).clamp(1e-6, 1.0 - 1e-6)
    pB = torch.sigmoid(logits_B).clamp(1e-6, 1.0 - 1e-6)
    kl = (pA * (pA.log() - pB.log())
          + (1.0 - pA) * ((1.0 - pA).log() - (1.0 - pB).log()))
    return kl.mean()


def sparsity_loss(M_t: torch.Tensor) -> torch.Tensor:
    """Negative mean peak-energy per (b, t) frame.

    Phase 31: SUPERSEDED by spatial_band_loss — this is an open-ended
    reward (-max keeps paying all the way to one-hot), the same ratchet
    pathology as temporal_sparsity_loss on the temporal axis.  It squeezed
    eff_sp from 3.4 (P29) to 1.8/49 cells (P30) and starved detection.
    Kept for ablation history; run with lambda_sparse=0.0.
    """
    return -M_t.amax(dim=(-2, -1)).mean()


def cbm_diversity_loss(slot_attn: torch.Tensor) -> torch.Tensor:
    """Phase 26: re-export of ConceptSlotBottleneck.slot_diversity_loss so
    scripts/HiDF_train_real.py can import everything from this module.

    Computes mean off-diagonal cosine similarity between K slot attention
    distributions per sample.  Minimised → slots attend to different
    spatial-temporal positions, preventing K-way attention collapse.

    slot_attn : (B, K, L)  where L = T * N
    Returns scalar tensor in [0, 1].
    """
    from models.HiDF_concept_bottleneck import slot_diversity_loss as _impl
    return _impl(slot_attn)


def temporal_sparsity_loss(M_frame: torch.Tensor) -> torch.Tensor:
    """Phase 25: negative mean peak-weight per sample — pushes temporal_gate
    toward peaky M_frame.

    M_frame : (B, T)  softmax over time, sums to 1 per sample.
    Uniform M_frame (1/T per cell) → loss = -1/T  (worst possible).
    One-hot M_frame                → loss = -1    (best possible).

    Why this matters: the k1/k2/k4 frame-drop test ranks frames by M_frame
    score, then zeros the top-K vs zeros random-K and compares confidence
    drops.  When M_frame is near-uniform (as in 6-9-26 0800hrs sample 2,
    where values were 0.033–0.079 against uniform 0.0625), the ranking is
    essentially random → top-K and random-K drops are equal → ratio ≈ 1.0
    or below by noise.  This loss adds pressure for the gate to commit.

    Phase 27/28 post-mortem: this form is an OPEN-ENDED reward (-max keeps
    paying all the way to one-hot) and drove M_frame to 99.998% one-hot in
    P27 and accelerated the P28 collapse.  Superseded by temporal_band_loss
    (Phase 30) — kept for ablation history.
    """
    return -M_frame.amax(dim=-1).mean()


def temporal_band_loss(M_frame: torch.Tensor, target_eff: float = 6.0) -> torch.Tensor:
    """Phase 30: BOUNDED temporal-selectivity penalty on M_frame.

    eff(p) = 1 / Σ_t p_t²  (inverse Herfindahl — "effective frame count";
    uniform over T frames → T, one-hot → 1).  Penalty:

        loss = relu(eff - target_eff) / max(T - target_eff, 1)   per sample

    Scaled to [0, 1]: uniform M_frame → 1, eff ≤ target_eff → EXACTLY 0.

    Why a band and not a reward: temporal_sparsity_loss (-max) is an
    open-ended ratchet — it kept paying gradient all the way to one-hot
    (Phase 27: M_frame 99.998% on one frame; Phase 28: collapse
    accelerant).  The hinge has zero gradient once eff_fr ≤ target_eff,
    so the gate concentrates to ~target_eff frames and then ONLY the
    classification signal decides where mass goes.  One-hot is reachable
    only if cls itself wants it.

    Why it matters (run 6-12-26 0020hrs): with lambda_temp_sparse=0 there
    was no temporal pressure at all — eff_fr sat at 10–13/16, the top
    frame carried 13% vs 6.25% uniform, and masking it moved confidence
    by 0.0015 (noise).  k-drop absolute numbers cannot grow until the
    gate commits to a subset of frames.

    M_frame : (B, T) softmax over time, sums to 1 per sample.
    """
    T = M_frame.shape[-1]
    eff = 1.0 / M_frame.pow(2).sum(dim=-1).clamp(min=1e-12)     # (B,)
    scale = max(float(T) - float(target_eff), 1.0)
    return F.relu(eff - float(target_eff)).mean() / scale


def spatial_band_loss(M_t: torch.Tensor,
                      lo: float = 4.0,
                      hi: float = 10.0) -> torch.Tensor:
    """Phase 31: BOUNDED two-sided spatial-footprint penalty on M_t.

    Per (b, t) frame: eff_sp = 1 / Σ_cells p²  (inverse Herfindahl over the
    N=h*w softmax cells — "effective cell count"; uniform → N, one-hot → 1).

        loss = relu(eff_sp - hi) / (N - hi)     too diffuse
             + relu(lo - eff_sp) / lo           too concentrated
        averaged over (B, T).  Zero everywhere inside [lo, hi].

    Why this replaces sparsity_loss (-peak): the -max form is an OPEN-ENDED
    reward — exactly the pathology that drove M_frame to 99.998% one-hot in
    P27 on the temporal axis.  On the spatial axis it squeezed eff_sp from
    3.4 (P29, detection 0.879) to 1.8 cells (P30 run 6-12-26 1300hrs,
    detection 0.796): the classifier was starved down to ~2 of 49 cells per
    frame, and the insertion metric's pixel ordering carried ~2 cells of
    signal before degenerating to a random tail (ins_gain_over_random
    -0.061).  The B/D bottleneck passes add their own concentration
    pressure, so without a lower hinge there is nothing to stop the
    collapse.

    Interplay with bottleneck_peak_floor=0.25: a map with eff_sp ≈ 4–5 can
    still hold its peak ≥ 0.25 (e.g. {0.3, 0.25, 0.2, 0.15, 0.1} → eff 4.4),
    so the band does not fight the B-pass floor.

    M_t : (B, T, h, w) softmax over the h*w cells per frame.
    """
    B, T = M_t.shape[0], M_t.shape[1]
    N = M_t.shape[-2] * M_t.shape[-1]
    p = M_t.reshape(B, T, N)
    eff = 1.0 / p.pow(2).sum(dim=-1).clamp(min=1e-12)           # (B, T)
    hi_scale = max(float(N) - float(hi), 1.0)
    lo_scale = max(float(lo), 1.0)
    too_diffuse      = F.relu(eff - float(hi)) / hi_scale
    too_concentrated = F.relu(float(lo) - eff) / lo_scale
    return (too_diffuse + too_concentrated).mean()


def localization_loss(M_t: torch.Tensor,
                      boundary: torch.Tensor) -> torch.Tensor:
    """Phase 33: pull the intrinsic attention M_t onto the self-blend seam.

    This is the direct fix for the insertion wall.  Runs 6-13/6-14 proved the
    detector's evidence is HOLISTIC: deleting the attended region crashes
    fake-confidence (necessity) but revealing it on a blur canvas does not
    restore it (sufficiency), so insertion loses to random for ANY compact map.
    A self-blended pseudo-fake (data.HiDF_self_blend.make_sbi_batch) has a
    LOCAL, causal artifact -- the blend boundary -- which IS sufficient.  Pull
    M_t onto that boundary and the attended region becomes sufficient too, so
    insertion in attention order recovers confidence fast (and faithfulness
    rises because gradient and attention now agree on the seam).

    M_t      : (B, T, h, w) softmax over the h*w cells per frame.
    boundary : (B, 1, H, W) non-negative seam-energy map (peaks at the blend
               boundary).  Downsampled to (h, w) by average pooling and
               normalised per sample to a target distribution.

    Loss = soft cross-entropy CE(target, M_t) averaged over (B, T):
        -sum_cells target * log(M_t)
    Minimised when M_t puts its mass on the boundary cells.  Gradient flows
    M_t -> EarlyAttnHead / refinement gate, so the attention head learns to
    seek blend seams.  Samples whose boundary is empty (no seam) contribute a
    uniform target (a no-op pull), so the term is always finite.
    """
    B, T, h, w = M_t.shape
    tgt = F.adaptive_avg_pool2d(boundary.float(), (h, w)).reshape(B, h * w)  # (B, h*w)
    tgt_sum = tgt.sum(dim=-1, keepdim=True)
    # Empty-boundary guard: fall back to uniform target (no directional pull).
    uniform = torch.full_like(tgt, 1.0 / float(h * w))
    tgt = torch.where(tgt_sum > 1e-8, tgt / tgt_sum.clamp(min=1e-8), uniform)  # (B, h*w)
    log_M = M_t.reshape(B, T, h * w).clamp(min=1e-8).log()                    # (B,T,h*w)
    ce = -(tgt.unsqueeze(1) * log_M).sum(dim=-1)                              # (B, T)
    return ce.mean()


def full_blur_input(x: torch.Tensor,
                    blur_kernel: int = 21,
                    blur_sigma: float = 10.0) -> torch.Tensor:
    """Phase 31: fully-blurred clip for the D-pass ANCHOR step.

    Returns gauss_blur(x) with NO dependence on M_t — used every
    del_anchor_every-th batch with target REAL for all samples.  A fully
    blurred clip contains no manipulation evidence by construction, so
    "REAL" is the epistemically correct label for both classes; this
    teaches the model the blur-end anchor of the deletion/insertion curves
    without ever pairing VISIBLE evidence with a REAL label (the P30
    poisoning).  Run 6-12-26 1300hrs: blurred_conf was 0.40, which put a
    hard floor under deletion AUC (del_at_100pct = 0.386) regardless of
    map quality.
    """
    with torch.no_grad():
        return _gaussian_blur_5d(x.detach(), blur_kernel, blur_sigma)
