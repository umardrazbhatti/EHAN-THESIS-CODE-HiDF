"""
data/HiDF_self_blend.py - Phase 33 online self-blended images (SBI).

WHY THIS EXISTS (root cause, proven by runs 6-13-26 / 6-14-26):
  The HiDF detector's fake-evidence is HOLISTIC, not local.  On the insertion
  probe, deleting the top-attended region crashes fake-confidence
  (0.45 -> 0.10, necessity holds) but revealing that same region on a blur
  canvas barely moves it (0.23 -> 0.27, sufficiency fails); revealing RANDOM
  pixels even beats revealing the attended ones.  So no compact attention map
  can be "sufficient" for a holistic detector and insertion structurally
  loses to random -- it is NOT a tuning bug.

THE FIX:
  Give the model a LOCAL, causal target.  Each optimizer step a small batch of
  REAL clips is self-blended on-GPU into pseudo-fakes: a mildly warped +
  colour/contrast-shifted copy of the clip is composited UNDER a soft convex
  mask, leaving a misaligned seam (the blend boundary).  The boundary is a
  compact artifact that IS sufficient by construction, so when the intrinsic
  attention M_t is supervised onto it (see losses.HiDF_explanation
  .localization_loss) the attended region becomes sufficient -> insertion in
  attention order recovers confidence fast.  Self-blends are also the SOTA
  cross-dataset generalizer (Face X-ray / SBI, Shiohara & Yamasaki 2022), so
  this doubles as the cross-dataset fix.

DESIGN NOTES:
  - Pure cheap tensor ops; no new dependencies; ASCII-only.
  - Temporally CONSISTENT: one warp + one mask per clip, applied to all T
    frames (HiDF swaps are present in every frame, so the pseudo-fake should be
    too -- this protects the temporal stream / M_frame from a spurious signal).
  - The main A/B/D detection passes are UNTOUCHED by this module; it only
    produces extra (frames, boundary) pairs the training loop consumes in a
    bounded auxiliary pass, so the met detection/deletion numbers are
    structurally protected.
  - All arithmetic is float32 regardless of AMP (caller disables autocast for
    generation) so grid_sample / affine_grid never see fp16 surprises.
"""

import math
import torch
import torch.nn.functional as F

# ImageNet normalisation (matches data/HiDF_transforms.py _MEAN/_STD exactly).
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def _rand_affine(b: int, device, dtype) -> torch.Tensor:
    """Small random similarity transform per clip: +-6 deg, +-5% scale,
    +-5% translation.  Returns theta (b, 2, 3) for F.affine_grid."""
    ang = (torch.rand(b, device=device, dtype=dtype) - 0.5) * (12.0 * math.pi / 180.0)
    sc = 1.0 + (torch.rand(b, device=device, dtype=dtype) - 0.5) * 0.10
    tx = (torch.rand(b, device=device, dtype=dtype) - 0.5) * 0.10
    ty = (torch.rand(b, device=device, dtype=dtype) - 0.5) * 0.10
    cos = torch.cos(ang) * sc
    sin = torch.sin(ang) * sc
    theta = torch.zeros(b, 2, 3, device=device, dtype=dtype)
    theta[:, 0, 0] = cos
    theta[:, 0, 1] = -sin
    theta[:, 0, 2] = tx
    theta[:, 1, 0] = sin
    theta[:, 1, 1] = cos
    theta[:, 1, 2] = ty
    return theta


def _soft_blob_mask(b: int, h: int, w: int, lo: float, hi: float,
                    device, dtype) -> torch.Tensor:
    """Random soft convex (elliptical) mask per sample, smooth edges, in [0,1].

    Centred near the middle (aligned face crops keep the face central) with
    semi-axes sampled in [lo, hi] of the half-extent.  Returns (b, 1, h, w).
    A soft (sigmoid) edge gives a transition RING; the seam energy m*(1-m)
    peaks there and is used as the localization target.
    """
    ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype).view(1, h, 1)
    xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype).view(1, 1, w)
    cy = (torch.rand(b, 1, 1, device=device, dtype=dtype) - 0.5) * 0.4
    cx = (torch.rand(b, 1, 1, device=device, dtype=dtype) - 0.5) * 0.4
    ry = lo + (hi - lo) * torch.rand(b, 1, 1, device=device, dtype=dtype)
    rx = lo + (hi - lo) * torch.rand(b, 1, 1, device=device, dtype=dtype)
    d = (((ys - cy) / ry.clamp(min=1e-3)) ** 2
         + ((xs - cx) / rx.clamp(min=1e-3)) ** 2)         # (b, h, w); =1 on ellipse
    edge = 10.0
    m = torch.sigmoid((1.0 - d) * edge)                   # ~1 inside, ~0 outside
    return m.unsqueeze(1)                                  # (b, 1, h, w)


def make_sbi_batch(frames: torch.Tensor,
                   blend_lo: float = 0.25,
                   blend_hi: float = 0.55):
    """Generate self-blended pseudo-fakes + boundary targets from REAL clips.

    frames : (B, T, C, H, W) float, ImageNet-NORMALISED real clips.
    Returns:
        sbi      : (B, T, C, H, W) float (same dtype as input) normalised
                   pseudo-fake clips with a blend seam.
        boundary : (B, 1, H, W) float32, non-negative seam-energy map (peaks at
                   the blend boundary; zero in flat regions).  Feed to
                   losses.HiDF_explanation.localization_loss together with M_t.
    """
    B, T, C, H, W = frames.shape
    device = frames.device
    out_dtype = frames.dtype
    dtype = torch.float32

    mean = torch.tensor(_MEAN, device=device, dtype=dtype).view(1, 1, C, 1, 1)
    std = torch.tensor(_STD, device=device, dtype=dtype).view(1, 1, C, 1, 1)

    x = frames.to(dtype)
    x01 = (x * std + mean).clamp(0.0, 1.0)                 # (B,T,C,H,W) in [0,1]

    # ---- warped source (temporally consistent: one grid per clip) -----------
    theta = _rand_affine(B, device, dtype)                # (B,2,3)
    grid = F.affine_grid(theta, size=(B, C, H, W), align_corners=False)  # (B,H,W,2)
    grid = grid.unsqueeze(1).expand(B, T, H, W, 2).reshape(B * T, H, W, 2)
    src = F.grid_sample(
        x01.reshape(B * T, C, H, W), grid,
        mode="bilinear", padding_mode="border", align_corners=False,
    ).reshape(B, T, C, H, W)

    # ---- mild colour + contrast shift per clip (statistical seam mismatch) --
    bright = 1.0 + (torch.rand(B, 1, 1, 1, 1, device=device, dtype=dtype) - 0.5) * 0.20
    contrast = 1.0 + (torch.rand(B, 1, 1, 1, 1, device=device, dtype=dtype) - 0.5) * 0.20
    cmean = src.mean(dim=(1, 3, 4), keepdim=True)         # (B,1,C,1,1)
    src = (((src - cmean) * contrast + cmean) * bright).clamp(0.0, 1.0)

    # ---- composite under the soft mask, renormalise -------------------------
    m = _soft_blob_mask(B, H, W, blend_lo, blend_hi, device, dtype)  # (B,1,H,W)
    m5 = m.unsqueeze(1)                                    # (B,1,1,H,W)
    blended01 = (m5 * src + (1.0 - m5) * x01).clamp(0.0, 1.0)
    sbi = (blended01 - mean) / std

    boundary = m * (1.0 - m)                               # (B,1,H,W) peaks at seam
    return sbi.to(out_dtype), boundary
