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


# Phase 35: artifact MODES -> (use_warp, use_color).  All three composite a
# modified source under the SAME soft elliptical mask, so the seam-energy target
# m*(1-m) is valid for every mode; they differ only in HOW the source is altered.
#   blend = warp + colour  (P33 default; blend-fakes Deepfakes/FaceShifter)
#   warp  = geometric warp only  (reenactment cue; Face2Face/NeuralTextures)
#   color = colour/contrast shift only  (graphics-swap cue; FaceSwap)
_MODE_OPS = {"blend": (True, True), "warp": (True, False), "color": (False, True)}


def make_sbi_batch(frames: torch.Tensor,
                   blend_lo: float = 0.25,
                   blend_hi: float = 0.55,
                   modes=("blend",),
                   partial_lo: float = 1.0,
                   partial_hi: float = 1.0):
    """Generate self-blended pseudo-fakes + boundary + frame_mask from REAL clips.

    frames : (B, T, C, H, W) float, ImageNet-NORMALISED real clips.

    Phase 35 additions (back-compat: modes=("blend",), partial_lo=hi=1.0 returns
    the EXACT Phase-33 pseudo-fake):
      modes       : iterable of artifact families (blend/warp/color); one is
                    sampled PER CLIP so a batch mixes manipulation types -> the
                    classifier sees graphics-swap / reenactment cues, not only
                    blend seams (the cross-dataset fix).
      partial_lo/hi: fraction of the T frames that carry the artifact, sampled
                    U[lo,hi] per clip.  <1.0 manipulates only k of T frames, so a
                    real KEY FRAME exists and the k1/k2/k4 frame-drop test stops
                    being noise on otherwise temporally-uniform fakes.

    Returns:
        sbi        : (B, T, C, H, W) normalised pseudo-fake clips (input dtype).
        boundary   : (B, 1, H, W) float32 seam-energy map (peaks at the mask edge).
        frame_mask : (B, T) float32, 1.0 on manipulated frames, 0.0 on the frames
                     left REAL.  Feed to localization_loss as a per-frame weight so
                     clean frames are not pulled onto a seam they do not have, and
                     use it as the temporal-localization target for the k-drop fix.
    """
    B, T, C, H, W = frames.shape
    device = frames.device
    out_dtype = frames.dtype
    dtype = torch.float32

    mean = torch.tensor(_MEAN, device=device, dtype=dtype).view(1, 1, C, 1, 1)
    std = torch.tensor(_STD, device=device, dtype=dtype).view(1, 1, C, 1, 1)

    x = frames.to(dtype)
    x01 = (x * std + mean).clamp(0.0, 1.0)                 # (B,T,C,H,W) in [0,1]

    # ---- per-clip artifact mode -> use_warp / use_color gates ----------------
    valid = [m for m in tuple(modes) if m in _MODE_OPS] or ["blend"]
    midx = torch.randint(len(valid), (B,), device=device)
    mode_w = torch.tensor([_MODE_OPS[m][0] for m in valid],
                          device=device, dtype=dtype)       # (n_modes,)
    mode_c = torch.tensor([_MODE_OPS[m][1] for m in valid],
                          device=device, dtype=dtype)
    use_warp  = mode_w[midx].view(B, 1, 1, 1, 1)            # (B,1,1,1,1) in {0,1}
    use_color = mode_c[midx].view(B, 1, 1, 1, 1)

    # ---- warped source (temporally consistent: one grid per clip) -----------
    theta = _rand_affine(B, device, dtype)                # (B,2,3)
    grid = F.affine_grid(theta, size=(B, C, H, W), align_corners=False)  # (B,H,W,2)
    grid = grid.unsqueeze(1).expand(B, T, H, W, 2).reshape(B * T, H, W, 2)
    src_warp = F.grid_sample(
        x01.reshape(B * T, C, H, W), grid,
        mode="bilinear", padding_mode="border", align_corners=False,
    ).reshape(B, T, C, H, W)
    src = use_warp * src_warp + (1.0 - use_warp) * x01    # warp only where selected

    # ---- mild colour + contrast shift per clip (statistical seam mismatch) --
    bright = 1.0 + (torch.rand(B, 1, 1, 1, 1, device=device, dtype=dtype) - 0.5) * 0.20
    contrast = 1.0 + (torch.rand(B, 1, 1, 1, 1, device=device, dtype=dtype) - 0.5) * 0.20
    cmean = src.mean(dim=(1, 3, 4), keepdim=True)         # (B,1,C,1,1)
    src_col = (((src - cmean) * contrast + cmean) * bright).clamp(0.0, 1.0)
    src = use_color * src_col + (1.0 - use_color) * src   # colour only where selected

    # ---- composite under the soft mask --------------------------------------
    m = _soft_blob_mask(B, H, W, blend_lo, blend_hi, device, dtype)  # (B,1,H,W)
    m5 = m.unsqueeze(1)                                    # (B,1,1,H,W)
    blended01 = (m5 * src + (1.0 - m5) * x01).clamp(0.0, 1.0)

    # ---- temporally-partial: manipulate only k of T frames ------------------
    # Sample a per-clip fraction, keep the top-k of a random score per row so the
    # chosen frames are spread arbitrarily (a clip-specific key frame set).
    p_lo = float(min(max(partial_lo, 0.0), 1.0))
    p_hi = float(min(max(partial_hi, p_lo), 1.0))
    if p_hi >= 1.0 and p_lo >= 1.0:
        frame_mask = torch.ones(B, T, device=device, dtype=dtype)
    else:
        frac   = torch.empty(B, device=device, dtype=dtype).uniform_(p_lo, p_hi)
        k      = (frac * T).round().clamp(min=1.0, max=float(T)).long()   # (B,)
        scores = torch.rand(B, T, device=device)
        order  = scores.argsort(dim=1, descending=True)                  # (B,T)
        rank   = torch.empty_like(order)
        rank.scatter_(1, order, torch.arange(T, device=device).expand(B, T))
        frame_mask = (rank < k.view(B, 1)).to(dtype)                     # (B,T) top-k = 1

    fm5 = frame_mask.view(B, T, 1, 1, 1)
    final01 = fm5 * blended01 + (1.0 - fm5) * x01         # clean frames stay REAL
    sbi = (final01 - mean) / std

    boundary = m * (1.0 - m)                               # (B,1,H,W) peaks at seam
    return sbi.to(out_dtype), boundary, frame_mask
