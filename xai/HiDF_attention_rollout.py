"""
xai/attention_rollout.py — Attention rollout over Transformer encoder layers.

Rollout: A_roll = A_L ⊗ ... ⊗ A_1 where A_i = 0.5·A_i_raw + 0.5·I.
CLS row of rollout gives relative importance of each spatial token → (T, h, w).
"""

import torch
import torch.nn.functional as F
import numpy as np


class AttentionRolloutExplainer:
    def __init__(self, model):
        self.model = model

    def explain(self, frames: torch.Tensor) -> np.ndarray:
        """
        Args:
            frames : (1, T, 3, H, W) — single video
        Returns:
            saliency : np.ndarray (T, H, W) in [0, 1]
        """
        device = frames.device
        B, T, _, H, W = frames.shape
        assert B == 1, "AttentionRollout expects a single video (B=1)."

        with torch.no_grad():
            _ = self.model(frames)

        attn_list = self.model.temporal_stream.layer_attention_weights
        if not attn_list or attn_list[0] is None:
            return np.zeros((T, H, W), dtype=np.float32)

        rollout = None
        for attn in attn_list:
            # attn : (B, N+1, N+1) — averaged over heads
            I    = torch.eye(attn.shape[-1], device=device).unsqueeze(0)
            attn = 0.5 * attn + 0.5 * I
            attn = attn / attn.sum(dim=-1, keepdim=True)
            rollout = attn if rollout is None else torch.bmm(attn, rollout)

        if rollout is None:
            return np.zeros((T, H, W), dtype=np.float32)

        # CLS row: importance of each spatial token
        N_total = rollout.shape[-1] - 1            # T * h * w
        cls_row = rollout[0, 0, 1:]               # (T*h*w,)

        h = w = self.model.feat_h
        N_per_frame = h * w

        if N_total != T * N_per_frame:
            # Fallback: distribute equally across frames
            cls_row = cls_row[:T * N_per_frame]

        M_t = cls_row.reshape(T, h, w)            # (T, h, w)

        # Per-frame normalise
        mn = M_t.reshape(T, -1).min(-1, keepdim=True)[0].unsqueeze(-1)
        mx = M_t.reshape(T, -1).max(-1, keepdim=True)[0].unsqueeze(-1)
        M_t = (M_t - mn) / (mx - mn + 1e-8)

        # Upsample to original frame size: (T,1,h,w) → (T,1,H,W) → (T,H,W)
        M_up = F.interpolate(
            M_t.unsqueeze(1),
            size=(H, W), mode="bilinear", align_corners=False
        ).squeeze(1)                          # (T, H, W)

        return M_up.cpu().numpy()
