"""
utils/checkpointing.py — save / load model checkpoints.

FIX: PyTorch ≥ 2.6 defaults to weights_only=True which rejects checkpoints
that embed config dataclasses or numpy scalars. We set weights_only=False
because we save our own trusted checkpoints (never load untrusted files this way).
"""

import os
import torch
from typing import Any, Dict, Optional
from HiDF_config import EAHNConfig


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch: int,
    best_metric: float,
    config: EAHNConfig,
    path: str,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ckpt = {
        "epoch":              epoch,
        "model_state_dict":   model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_metric":        best_metric,
        "config":             config,
    }
    torch.save(ckpt, path)


def load_checkpoint(
    path: str,
    model,
    optimizer=None,
    scheduler=None,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load a checkpoint.  weights_only=False is intentional: our checkpoints
    contain Python dataclass objects (EAHNConfig) and numpy scalars.
    Only load from trusted sources.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=strict)
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt
