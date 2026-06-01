"""
xai/sanity_checks.py — Adebayo et al. 2018 sanity checks for explanation faithfulness.

Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., & Kim, B. (2018).
Sanity Checks for Saliency Maps. NeurIPS 2018.
"""

import copy
import torch
import numpy as np
from typing import Optional


def _cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    """Cosine similarity between two explanation maps, with NaN guard for degenerate inputs."""
    a_flat = a.reshape(-1).astype(float)
    b_flat = b.reshape(-1).astype(float)
    if a_flat.sum() < eps or b_flat.sum() < eps:
        print("[WARN] _cosine_sim: one or both maps are near-zero — returning NaN")
        return float("nan")
    num = float(np.dot(a_flat, b_flat))
    den = float(np.linalg.norm(a_flat) * np.linalg.norm(b_flat)) + eps
    return float(np.clip(num / den, -1.0, 1.0))


def _get_M_t(model, frames: torch.Tensor) -> np.ndarray:
    """
    Run a single forward pass and return M_t_up[0] as a numpy array (T, H, W).
    frames: (1, T, C, H, W)
    """
    device = next(model.parameters()).device
    with torch.no_grad():
        out = model(frames.to(device))
    return out.M_t_up[0].cpu().numpy()   # (T, H, W)


def model_randomization_check(
    model,
    frames: torch.Tensor,
    n_random: int = 3,
) -> float:
    """
    Phase 9 hotfix — three NaN fixes applied:
      (a) Construct random_model with pretrained backbone so BatchNorm
          running stats are valid → no NaN in forward pass from uninitialised
          running_var (EfficientNet has ~50 BN layers; one NaN propagates).
      (b) Add epsilon to cosine denominator → no 0/0 on zero-norm M_t frames.
      (c) NaN-filter per-sample before averaging → one degenerate sample no
          longer poisons the whole metric.

    Parameters
    ----------
    model    : trained EAHN model (not modified)
    frames   : (B, T, C, H, W) sample tensor
    n_random : kept for API compatibility (not used in single-pass variant)

    Returns
    -------
    mean_cosine_similarity : float  in [0, 1], or NaN only if ALL samples fail.
        LOW (< 0.5)  = explanation changes a lot when weights are random (good).
        HIGH (> 0.7) = explanation barely changes (bad — check faithfulness).
    """
    device = next(model.parameters()).device

    # (a) Build a randomly-initialised EAHN that still has valid BN stats.
    #     Start from the same config with backbone_pretrained=True so the CNN's
    #     BN running_mean / running_var are populated; then re-init everything
    #     else (temporal stream, cross-attention, classifier) to random weights.
    cfg = copy.copy(model.config)
    cfg.backbone_pretrained = True
    random_model = type(model)(cfg).to(device).eval()

    for name, module in random_model.named_modules():
        if name.startswith("spatial_stream.backbone"):
            continue   # keep pretrained weights + BN running stats
        if hasattr(module, "reset_parameters"):
            try:
                module.reset_parameters()
            except Exception:
                pass

    eps = 1e-8
    per_sample_cos = []

    with torch.no_grad():
        frames_dev = frames.to(device)

        m_trained = model(frames_dev).M_t        # (B, T, h, w)
        m_random  = random_model(frames_dev).M_t  # (B, T, h, w)

    if torch.isnan(m_trained).any() or torch.isnan(m_random).any():
        print(f"[mt_vs_random] WARN: NaN in M_t "
              f"(trained={torch.isnan(m_trained).any().item()}, "
              f"random={torch.isnan(m_random).any().item()}); returning NaN")
        return float("nan")

    # (b) Flatten per (sample, frame) → cosine with epsilon denominator
    a = m_trained.flatten(2)                  # (B, T, L)
    b = m_random.flatten(2)                   # (B, T, L)
    num = (a * b).sum(dim=-1)                 # (B, T)
    den = a.norm(dim=-1) * b.norm(dim=-1) + eps
    cos = num / den                           # (B, T)
    cos_per_sample = cos.mean(dim=-1)         # (B,)

    # (c) NaN-filter per sample before averaging
    finite = cos_per_sample[~torch.isnan(cos_per_sample)]
    per_sample_cos = finite.cpu().tolist()

    if not per_sample_cos:
        print("[mt_vs_random] ALL samples NaN — returning NaN deliberately.")
        return float("nan")

    mean_cos = sum(per_sample_cos) / len(per_sample_cos)
    print(f"[mt_vs_random] n_samples={len(per_sample_cos)}  mean_cos={mean_cos:.4f}")
    return mean_cos


def label_randomization_check(
    model,
    train_loader,
    config,
    n_batches: int = 5,
) -> Optional[float]:
    """
    Optional label-randomization sanity check (Adebayo et al., Figure 2b).

    Retrain the model for a few steps with randomly shuffled labels and check
    whether the explanation maps change. This is expensive and skipped by default.

    Returns None to indicate this check was skipped.
    """
    return None
