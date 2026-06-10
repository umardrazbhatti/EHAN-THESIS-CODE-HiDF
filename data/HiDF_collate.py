"""
data/collate.py — custom collate function for deepfake batches.
"""

import torch


def deepfake_collate_fn(batch):
    frames = torch.stack([item["frames"] for item in batch])               # (B,T,3,H,W)
    labels = torch.tensor([item["label"]    for item in batch],
                           dtype=torch.float32)                            # (B,)
    meta   = [item["meta"] for item in batch]
    result = {
        "frames": frames,
        "label":  labels,
        "meta":   meta,
    }
    if "frames_clean" in batch[0]:
        result["frames_clean"] = torch.stack(
            [item["frames_clean"] for item in batch]
        )  # (B,T,3,H,W)
    # ── Phase 27: pass the synthetic-domain label through (default -1) ───────
    # Train loader emits domain in [0, num_domains-1] when DANN is enabled.
    # Val/test (and Phase 26 fallback) emit -1, which the training loop
    # interprets as "no domain loss for this sample".
    if "domain" in batch[0]:
        result["domain"] = torch.tensor(
            [item.get("domain", -1) for item in batch],
            dtype=torch.long,
        )                                                                  # (B,)
    return result
