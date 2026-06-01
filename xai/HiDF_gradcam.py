"""
xai/gradcam.py — Grad-CAM on a per-frame spatial model.

FIX: The original code used ClassifierOutputTarget(1) on a binary sigmoid
classifier that outputs a (B,1) or even scalar per sample.  That causes
    IndexError: index 1 is out of bounds for dimension 0 with size 1
The fix wraps the spatial model to output the raw logit (scalar per sample)
and uses a simple custom target that returns the logit directly, bypassing
the category-index lookup.
"""

import torch
import torch.nn as nn
import numpy as np


class _SpatialModel(nn.Module):
    """Frame-level model: backbone → proj → global avg-pool → linear → scalar."""

    def __init__(self, spatial_stream, d_model: int, device: str):
        super().__init__()
        self.backbone    = spatial_stream.backbone
        self.proj        = spatial_stream.proj
        self.avg_pool    = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier  = nn.Linear(d_model, 1)
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,3,H,W) → (B,1) logit"""
        feats     = self.backbone(x)
        last_feat = feats[-1]
        proj      = self.proj(last_feat)                # (B, d_model, h, w)
        pooled    = self.avg_pool(proj).reshape(x.size(0), -1)
        logit     = self.classifier(pooled)              # (B, 1)
        return logit


class _ScalarOutputTarget:
    """GradCAM target that returns output[:, 0] for binary classifiers."""
    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        if model_output.dim() == 2:
            return model_output[:, 0].sum()
        return model_output.sum()


class GradCAMExplainer:
    def __init__(self, eahn_model, target_layer: nn.Module):
        """
        NOTE: GradCAM is computed on the SpatialStream alone (backbone → proj →
        global_avg_pool → linear).  This is a spatial-stream approximation of the
        full EAHN explanation, not a faithful attribution of the CLS-token-based
        classifier. It is used as a post-hoc comparison baseline, not as a primary
        explanation. The primary explanation is M_t from CrossAttentionFusion.
        """
        self.model = eahn_model
        device = eahn_model.config.device

        self.spatial_model = _SpatialModel(
            spatial_stream=eahn_model.spatial_stream,
            d_model=eahn_model.config.d_model,
            device=device,
        )

        # Share weights with the trained EAHN spatial head
        self.spatial_model.classifier.weight.data.copy_(
            eahn_model.classifier.weight.data
        )
        self.spatial_model.classifier.bias.data.copy_(
            eahn_model.classifier.bias.data
        )

        from pytorch_grad_cam import GradCAM
        self.cam = GradCAM(
            model=self.spatial_model,
            target_layers=[target_layer],
        )
        self.device = device

    def explain(self, frames: torch.Tensor) -> np.ndarray:
        """
        Args:
            frames : (B, T, 3, H, W)
        Returns:
            heatmaps : np.ndarray (B, T, H, W) in [0, 1]
        """
        B, T, C, H, W = frames.shape
        frames_flat = frames.reshape(B * T, C, H, W).to(self.device)

        targets = [_ScalarOutputTarget()] * (B * T)

        # aug_smooth=False avoids the secondary augmentation passes that
        # trigger the same index error if the model architecture changes
        grayscale_cams = self.cam(
            input_tensor=frames_flat,
            targets=targets,
            aug_smooth=False,
            eigen_smooth=False,
        )  # (B*T, H, W)

        heatmaps = torch.from_numpy(grayscale_cams).reshape(B, T, H, W)

        # Per-frame normalise to [0, 1]
        mn = heatmaps.reshape(B, T, -1).min(-1, keepdim=True)[0].unsqueeze(-1)
        mx = heatmaps.reshape(B, T, -1).max(-1, keepdim=True)[0].unsqueeze(-1)
        heatmaps = (heatmaps - mn) / (mx - mn + 1e-8)

        return heatmaps.numpy()
