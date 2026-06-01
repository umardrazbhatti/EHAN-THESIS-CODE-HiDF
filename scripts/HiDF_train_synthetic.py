"""
scripts/train_synthetic.py — Phase 1: CPU-only smoke test using synthetic data.
Runs 2 epochs to verify that the full pipeline is wired correctly before
committing GPU time on Kaggle.
"""

import os
from HiDF_config import EAHNConfig
from scripts.HiDF_train_real import main as train_main


def main():
    config = EAHNConfig(
        dataset_name="synthetic",
        epochs=2,
        batch_size=2,
        num_frames=4,
        frame_size=224,
        transformer_layers=2,
        transformer_heads=2,
        d_model=64,
        mixed_precision=False,
        num_workers=0,
        output_dir="outputs_synthetic/",
        eval_after_train=True,
        heatmap_samples=4,
        device="cpu",
    )
    os.makedirs(config.output_dir, exist_ok=True)
    train_main(config)


if __name__ == "__main__":
    main()
