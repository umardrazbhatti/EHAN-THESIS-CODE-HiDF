"""
models/HiDF_grl.py - Phase 27 Gradient Reversal Layer + Domain Classifier head.

DANN (Ganin & Lempitsky, 2015) wires a domain classifier to an intermediate
feature with a gradient-reversal layer in between:

    feature -> GRL(lambda) -> domain_head -> domain_logits
                       |
                       backward sign-flip + scaling

  forward:  identity
  backward: gradient * (-lambda)

Effect: minimising domain CE with normal gradient *and* propagating that
through the GRL pushes the upstream feature to be domain-INVARIANT (because
the upstream sees the negated gradient and tries to confuse the domain head).

PHASE 27 SETUP
  Wire point: attn_pool (B, d) -- the highest-level shared feature consumed
              by both the main aux classifier and the serial CBM head.
  Domains   : D=4 synthetic augmentation domains
                 0 = minimal (clean)
                 1 = heavy JPEG (codec shift)
                 2 = Gaussian noise (sensor shift)
                 3 = Gaussian blur (resolution shift)
              Labels are assigned at data/HiDF_datasets.py per sample.
  Warmup    : lambda_grl ramps 0 -> lambda_grl_max over domain_warmup_epochs.
              At lambda=0 the GRL is identity in both directions, so the
              domain head still trains but does NOT yet pull features
              toward invariance.  This gives the classifier time to find
              useful features before the adversary kicks in.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class _GradReverseFn(torch.autograd.Function):
    """Forward = identity. Backward = -lambda * grad."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Negate + scale; no gradient for lambda_ (scalar hyperparam)
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    """Functional wrapper around the GRL autograd function."""
    return _GradReverseFn.apply(x, lambda_)


class DomainHead(nn.Module):
    """Small MLP that classifies a feature vector into D domains.

    Architecture (~50k params):
        LayerNorm(d) -> Linear(d, d//2) -> GELU -> Dropout(0.2)
                     -> Linear(d//2, num_domains)

    Returns
    -------
    domain_logits : (B, num_domains)
    """

    def __init__(self, d_model: int = 256, num_domains: int = 4,
                 dropout: float = 0.2):
        super().__init__()
        hidden = max(d_model // 2, 64)
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_domains),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def domain_warmup(epoch: int, warmup_epochs: int, target_lambda: float) -> float:
    """Linear ramp 0 -> target_lambda over warmup_epochs (matches faith_warmup)."""
    if warmup_epochs <= 0:
        return float(target_lambda)
    return float(target_lambda) * min(1.0, float(epoch) / float(warmup_epochs))


def domain_accuracy(domain_logits: torch.Tensor,
                    domain_labels: Optional[torch.Tensor]) -> float:
    """Top-1 accuracy of the domain classifier.

    Diagnostic only. High accuracy early = healthy; should remain >chance
    even after GRL kicks in (because some domain signal is unavoidable
    from a fixed-dim feature).  If accuracy collapses to ~1/D, the domain
    head has been over-confused and DANN has gone too far.
    """
    if domain_labels is None or domain_labels.numel() == 0:
        return float("nan")
    with torch.no_grad():
        pred = domain_logits.argmax(dim=-1)
        # Treat -1 (sentinel for val/test or "no domain") as ignored
        mask = (domain_labels >= 0)
        if mask.sum() == 0:
            return float("nan")
        acc = (pred[mask] == domain_labels[mask]).float().mean().item()
    return float(acc)
