import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationLoss(nn.Module):
    """BCE with optional label smoothing (maps 0→ε, 1→1-ε)."""

    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logit: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        label = label.float()
        if logit.shape != label.shape:
            label = label.view_as(logit)
        if self.label_smoothing > 0:
            label = label * (1.0 - 2 * self.label_smoothing) + self.label_smoothing
        return self.bce(logit, label)


class FocalLoss(nn.Module):
    """
    Focal loss for class-imbalanced binary classification, with optional label
    smoothing.  Label smoothing is applied to the BCE target only; the focal
    factor pt is computed from the original (unsmoothed) targets so the
    weighting scheme is not distorted.

    Use when WeightedRandomSampler is turned off (e.g., Celeb-DF where
    sampler may cause overfitting on the 890 real samples).
    With WeightedRandomSampler active, default to BCE.

    alpha : LEGACY global scale.  NOTE (Phase 31 finding): this multiplies
            BOTH classes equally — `alpha * (1-pt)^gamma * bce` — so despite
            the v4 comment "raised to penalise fake misses harder" it never
            weighted fakes at all; it is a pure scale on the loss.  Two runs
            of evidence (P29 fake_acc@0.5 = 0.497, P30 = 0.569 while
            real_acc ≈ 0.86) motivated the class-conditional form below.
    alpha_pos / alpha_neg : Phase 31 class-conditional weights — alpha_pos
            multiplies FAKE (label 1) terms, alpha_neg REAL (label 0) terms
            (the standard Lin et al. alpha_t form, made explicit).  Both
            must be ≥ 0 to activate; otherwise the legacy global alpha is
            used, bit-for-bit identical to the old behaviour.
            Run choice 1.0/0.5: 2:1 fake-error weighting with mean scale
            0.75 on a balanced batch = unchanged total magnitude vs the
            legacy global 0.75.
    gamma : focusing parameter — higher = more focus on hard examples
    label_smoothing : maps 0→ε, 1→1-ε before BCE
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 label_smoothing: float = 0.0,
                 alpha_pos: float = -1.0, alpha_neg: float = -1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg

    def forward(self, logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        if logit.shape != target.shape:
            target = target.view_as(logit)

        # Focal factor uses *original* (unsmoothed) targets
        prob = torch.sigmoid(logit)
        pt   = torch.where(target >= 0.5, prob, 1 - prob)

        # Phase 31: class-conditional alpha_t when both weights are set;
        # legacy global scale otherwise (exact back-compat).
        if self.alpha_pos >= 0.0 and self.alpha_neg >= 0.0:
            alpha_t = torch.where(
                target >= 0.5,
                torch.as_tensor(self.alpha_pos, dtype=prob.dtype, device=prob.device),
                torch.as_tensor(self.alpha_neg, dtype=prob.dtype, device=prob.device),
            )
        else:
            alpha_t = self.alpha

        # Apply label smoothing to the BCE target only
        if self.label_smoothing > 0:
            smooth_target = target * (1.0 - 2 * self.label_smoothing) + self.label_smoothing
        else:
            smooth_target = target

        bce   = F.binary_cross_entropy_with_logits(logit, smooth_target, reduction='none')
        focal = alpha_t * (1 - pt).pow(self.gamma) * bce
        return focal.mean()


def build_classification_loss(config) -> nn.Module:
    """Factory: returns FocalLoss or ClassificationLoss based on config.cls_loss_type."""
    ls = float(getattr(config, "label_smoothing", 0.0))
    if getattr(config, "cls_loss_type", "bce") == "focal":
        return FocalLoss(
            alpha=getattr(config, "focal_alpha", 0.25),
            gamma=getattr(config, "focal_gamma", 2.0),
            label_smoothing=ls,
            alpha_pos=float(getattr(config, "focal_alpha_pos", -1.0)),
            alpha_neg=float(getattr(config, "focal_alpha_neg", -1.0)),
        )
    return ClassificationLoss(label_smoothing=ls)
