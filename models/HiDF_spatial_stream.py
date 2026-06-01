"""
models/spatial_stream.py — EfficientNet/ConvNeXt backbone wrapper.

Projects last feature map to d_model tokens (B*T, N, d_model).
Caches low-level features for the gated temporal loss gate φ(f_t).
"""

import torch
import torch.nn as nn
import timm


class SpatialStream(nn.Module):
    def __init__(
        self,
        backbone_name: str = "efficientnet_b4",
        pretrained: bool = True,
        d_model: int = 256,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, features_only=True
        )
        self.feat_channels = self.backbone.feature_info.channels()[-1]
        self.proj = nn.Conv2d(self.feat_channels, d_model, kernel_size=1)

        # Low-level extractor used as gate signal φ(f_t) in L_temp.
        # Must NOT require gradients (detached, no parameters shared with classifier path).
        if "efficientnet" in backbone_name:
            self.low_level_extractor = nn.Sequential(
                self.backbone.conv_stem,
                self.backbone.bn1,
                nn.SiLU(inplace=True),
            )
        elif "convnext" in backbone_name:
            self.low_level_extractor = self.backbone.stem
        else:
            self.low_level_extractor = None   # fallback: use first feature map

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self._cached_low_level: torch.Tensor = None
        self.feat_h: int = None
        self.feat_w: int = None

    @property
    def grad_cam_target_layer(self):
        """
        Returns the correct target layer for Grad-CAM: the last convolutional
        block of the backbone (has spatial extent → produces spatial maps).
        Using self.proj (1×1 conv) gives all-zero maps because it has no
        spatial receptive field variation.
        """
        if hasattr(self.backbone, "blocks"):
            return self.backbone.blocks[-1]   # EfficientNet
        elif hasattr(self.backbone, "stages"):
            return self.backbone.stages[-1]   # ConvNeXt
        else:
            return self.proj                  # fallback

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames : (B*T, 3, H, W)
        Returns:
            tokens : (B*T, N, d_model)   where N = feat_h * feat_w
        """
        # Compute and cache low-level features for gating
        with torch.no_grad():
            if self.low_level_extractor is not None:
                low = self.low_level_extractor(frames)
            else:
                low = self.backbone(frames)[0]
        self._cached_low_level = low.detach()

        # Backbone forward (full feature pyramid, take last stage)
        feats = self.backbone(frames)
        last  = feats[-1]                               # (B*T, C, h, w)
        self.feat_h, self.feat_w = last.shape[-2], last.shape[-1]

        proj = self.proj(last)                          # (B*T, d_model, h, w)
        tokens = proj.flatten(2).transpose(1, 2)        # (B*T, N, d_model)
        return tokens

    def low_level_features(self) -> torch.Tensor:
        """Return the cached low-level feature map from the last forward pass."""
        if self._cached_low_level is None:
            raise RuntimeError("Call forward() before low_level_features().")
        return self._cached_low_level
