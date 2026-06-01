"""
xai/shap_explainer.py — Integrated Gradients (Captum) as SHAP approximation.
"""

import torch
import numpy as np


class SHAPExplainer:
    def __init__(self, model, method: str = "integratedgrads"):
        self.model  = model
        self.method = method
        if method == "integratedgrads":
            from captum.attr import IntegratedGradients
            self.ig = IntegratedGradients(self._forward_wrapper)

    def _forward_wrapper(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 3, H, W) → logit (B,)"""
        return self.model(x).logit

    def explain(self, frames: torch.Tensor) -> np.ndarray:
        """
        Args:
            frames : (1, T, 3, H, W) — single video, on model device
        Returns:
            saliency : np.ndarray (T, H, W) in [0, 1]
        """
        frames = frames.float().requires_grad_(True)
        try:
            attributions = self.ig.attribute(
                frames, target=None, n_steps=20, internal_batch_size=1
            )
        except Exception:
            # Fallback: simple gradient saliency
            out = self.model(frames)
            out.logit.backward()
            attributions = frames.grad

        saliency = attributions.abs().mean(dim=2, keepdim=True)  # avg over RGB
        saliency  = saliency.squeeze(0).squeeze(1)               # (T, H, W)

        mn = saliency.min()
        mx = saliency.max()
        saliency = (saliency - mn) / (mx - mn + 1e-8)
        return saliency.detach().cpu().numpy()
