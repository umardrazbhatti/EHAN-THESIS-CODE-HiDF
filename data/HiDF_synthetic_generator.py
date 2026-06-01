"""
data/synthetic_generator.py — lightweight synthetic deepfake generator for
CPU-only pipeline validation (no real video data required).
"""

import torch
import numpy as np
from typing import Tuple, Optional


class SyntheticDataGenerator:
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def generate_sequence(
        self,
        num_frames: int = 16,
        frame_size: Tuple[int, int] = (224, 224),
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """
        Returns:
            frames  : (T, 3, H, W) float tensor in [0, 1]  (NOT normalised)
            label   : int, 0 = real, 1 = fake
            mask    : (H, W) float tensor in [0, 1]
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        H, W = frame_size
        label = int(self.rng.choice([0, 1]))
        background = self.rng.uniform(0.2, 0.8, size=(H, W, 3)).astype(np.float32)

        if label == 0:
            frames = [background.copy() for _ in range(num_frames)]
            mask = np.zeros((H, W), dtype=np.float32)
        else:
            mask = np.zeros((H, W), dtype=np.float32)
            ph, pw = H // 3, W // 3
            y1, x1 = H // 3, W // 3
            y2, x2 = y1 + ph, x1 + pw
            mask[y1:y2, x1:x2] = 1.0
            rect_color = self.rng.uniform(0, 1, size=3).astype(np.float32)
            frames = []
            for t in range(num_frames):
                frame = background.copy()
                if t >= num_frames // 2:
                    patch = frame[y1:y2, x1:x2] * 0.5 + rect_color * 0.5
                    patch[::2, ::2] = 1 - rect_color
                    frame[y1:y2, x1:x2] = np.clip(patch, 0, 1)
                frames.append(frame)

        frames_np = np.stack(frames)                                    # (T, H, W, 3)
        frames_t  = torch.from_numpy(frames_np).permute(0, 3, 1, 2)  # (T, 3, H, W)

        # ImageNet normalisation expected by pretrained EfficientNet/ConvNeXt backbones
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        std  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
        frames_t = (frames_t - mean[None, :, None, None]) / std[None, :, None, None]

        mask_t = torch.from_numpy(mask)                               # (H, W)
        return frames_t, label, mask_t
