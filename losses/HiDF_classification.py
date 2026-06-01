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

    alpha : down-weights the easy-majority-class loss
    gamma : focusing parameter — higher = more focus on hard examples
    label_smoothing : maps 0→ε, 1→1-ε before BCE
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        if logit.shape != target.shape:
            target = target.view_as(logit)

        # Focal factor uses *original* (unsmoothed) targets
        prob = torch.sigmoid(logit)
        pt   = torch.where(target >= 0.5, prob, 1 - prob)

        # Apply label smoothing to the BCE target only
        if self.label_smoothing > 0:
            smooth_target = target * (1.0 - 2 * self.label_smoothing) + self.label_smoothing
        else:
            smooth_target = target

        bce   = F.binary_cross_entropy_with_logits(logit, smooth_target, reduction='none')
        focal = self.alpha * (1 - pt).pow(self.gamma) * bce
        return focal.mean()


def build_classification_loss(config) -> nn.Module:
    """Factory: returns FocalLoss or ClassificationLoss based on config.cls_loss_type."""
    ls = float(getattr(config, "label_smoothing", 0.0))
    if getattr(config, "cls_loss_type", "bce") == "focal":
        return FocalLoss(
            alpha=getattr(config, "focal_alpha", 0.25),
            gamma=getattr(config, "focal_gamma", 2.0),
            label_smoothing=ls,
        )
    return ClassificationLoss(label_smoothing=ls)
