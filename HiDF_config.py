"""
config.py — single source of truth for all EAHN hyperparameters.
CLI overrides via argparse; no hardcoded paths anywhere else.
"""

import argparse
import warnings
import torch
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class EAHNConfig:
    # ── Paths ─────────────────────────────────────────────────────────────────
    data_root: str = "/kaggle/input/"
    output_dir: str = "/kaggle/working/outputs/"
    cache_dir: str = "/kaggle/working/.face_cache/"
    resume_checkpoint: str = ""

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset_name: Literal["synthetic", "ff++", "celeb_df", "dfdc", "hidf"] = "ff++"
    dataset_compression: str = "c23"
    num_frames: int = 16
    frame_size: int = 224
    train_split: float = 0.8
    val_split: float = 0.1

    # ── Model ─────────────────────────────────────────────────────────────────
    backbone: str = "efficientnet_b4"
    backbone_pretrained: bool = True
    transformer_layers: int = 4
    transformer_heads: int = 8
    d_model: int = 256
    dropout: float = 0.1

    # ── Loss weights ──────────────────────────────────────────────────────────
    lambda1: float = 0.02   # L_exp weight (reduced 0.3→0.1 phase20, 0.1→0.02 phase21: L_sparse takes over sparsity pressure)
    lambda2: float = 0.2   # L_temp weight (raised 0.1→0.2 phase6: loosen temporal grip)
    lambda_consistency: float = 0.3   # weight for consistency regularization loss (MSE between augmented and clean branch probs)
    alpha: float = 0.05    # entropy weight in weak supervision (phase20: alpha=0.3 was driving M_t to one-hot per frame; lowering frees M_t to form face-sized blobs)
    beta: float = 0.5      # TV weight in weak supervision
    gamma: float = 0.1     # gate decay rate in L_temp (was 10.0 — caused exp→0)
    attn_temp_init: float = 0.7    # start at τ=exp(0.7)≈2.0 (smoother softmax); log_temp remains learnable, this is initialization only (phase20)
    attn_diversity_weight: float = 5.0  # weight for JS diversity penalty in L_exp (raised 3.0→5.0 phase8: JS has smaller scale than cosine)
    cls_dropout_p: float = 0.0    # phase7: disabled — attn_pool now informative; joint gradient on every step
    label_smoothing: float = 0.05   # Task 3.2: maps 0→0.05, 1→0.95 to prevent logit saturation at 0.000/1.000
    max_per_class: int = 0         # if > 0, subsample train set to this many samples per class

    # ── Classification loss ───────────────────────────────────────────────────
    cls_loss_type: str = "focal"   # "bce" | "focal" — phase 19.8: activate focal to up-weight hard fakes
    focal_alpha: float = 0.75   # v4: raised 0.65→0.75 to penalise fake misses harder (fixes fake_acc collapse)
    focal_gamma: float = 2.5   # v4: raised 2.0→2.5 for stronger hard-example focus

    # ── Training ──────────────────────────────────────────────────────────────
    epochs: int = 50
    batch_size: int = 4        # T4-safe with AMP+grad_ckpt: B*T=4*16=64 frames; grad_accum_steps=4 → effective 16
    grad_accum_steps: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-2
    mixed_precision: bool = True   # kept for backward compat; use_amp is the authoritative flag
    num_workers: int = 0   # 0 = safe for Kaggle CUDA; increase locally if desired
    use_amp: bool = True           # FP16 automatic mixed precision (T4 supports FP16 not BF16)
    amp_dtype: str = "fp16"        # "fp16" | "bf16"
    grad_checkpoint: bool = True   # gradient checkpointing in TemporalStream to cut VRAM
    clip_grad_norm: float = 1.0    # max gradient norm for clipping

    # ── Evaluation / Visualisation ────────────────────────────────────────────
    eval_after_train: bool = True
    skip_eval: bool = False          # if True, suppress post-training evaluation entirely
    active_manipulation: str = ""           # REQUIRED at CLI; specialist-only mode
    celebdf_root: str = ""                  # path to Celeb-DF v2 dataset root
    celebdf_eval: bool = False              # run Celeb-DF test eval after FF++ test eval
    hidf_root: str = ""
    ffpp_cross_eval: bool = False
    ffpp_cross_root: str = ""
    hidf_split_seed: int = 42
    save_last_checkpoint: bool = False      # Phase 16 leftover; OFF by default
    explanation_suite: bool = True          # run new explanation metrics block after eval
    save_heatmaps: bool = True
    heatmap_samples: int = 20
    random_test_n_samples: int = 30         # Task 1.7: n_random for model-randomization check (was 1)

    # ── Early stopping (Task 3.3) ─────────────────────────────────────────────
    early_stop_patience:  int   = 5                         # epochs without improvement before halt
    early_stop_metric:    str   = "val_balanced_accuracy"   # metric to monitor
    early_stop_min_delta: float = 0.001                     # minimum improvement to count
    no_early_stop:        bool  = False                     # v4: set True to disable ES entirely (run full epochs)

    # ── Phase 21: faithful attention bottleneck ───────────────────────────────
    phase21_enabled:      bool  = True    # master switch; False reverts to Phase 20 behaviour
    lambda_faith:         float = 0.3     # weight for faithfulness KL loss
    lambda_sparse:        float = 0.05    # weight for sparsity (negative peak) loss
    faith_warmup_epochs:  int   = 3       # linear ramp from 0 → lambda_faith over N epochs
    attn_floor:           float = 0.05    # gate floor in EarlyAttnHead
    blur_kernel:          int   = 21      # Gaussian kernel size for bottlenecked input
    lambda_peak_spread:   float = 0.5     # v4: raised 0.3→0.5; weight for HardAttentionDiversityLoss
    lambda_sharp:         float = 0.15     # v4: raised 0.5→1.0; weight for sharpness loss on logits
    disk_guard_gb:        float = 3.0     # v4: min free GB before face-cache write is skipped
    blur_sigma:           float = 10.0    # Gaussian sigma for bottlenecked input
    bottleneck_peak_floor: float = 0.25   # Phase 22: fixed floor for bottleneck mask normalisation;
                                          # diffuse maps (peak < floor) get heavily blurred so L_faith bites
    snapshot_every:       int   = 2       # save snapshot every N epochs

    # ── Device ────────────────────────────────────────────────────────────────
    device: str = "auto"

    def __post_init__(self):
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
                warnings.warn("No GPU found. Switching to CPU with reduced settings.")
                self._apply_cpu_safe_overrides()

    def _apply_cpu_safe_overrides(self):
        self.num_frames = 4
        self.transformer_layers = 2
        self.transformer_heads = 2
        self.batch_size = 2
        self.mixed_precision = False
        self.use_amp = False
        self.grad_checkpoint = False
        self.num_workers = 0
        if "efficientnet_b4" in self.backbone:
            self.backbone = "efficientnet_b0"

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "EAHNConfig":
        cfg = cls()
        for key, val in vars(args).items():
            if hasattr(cfg, key) and val is not None:
                setattr(cfg, key, val)
        if cfg.dataset_name == "ff++" and not cfg.active_manipulation:
            raise ValueError(
                "--active_manipulation is required when --dataset_name ff++. "
                "Choose one of: Deepfakes, Face2Face, FaceShifter, FaceSwap, NeuralTextures."
            )
        return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EAHN Training and Evaluation")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None,
                        choices=["synthetic", "ff++", "celeb_df", "dfdc", "hidf"])
    parser.add_argument("--dataset_compression", type=str, default=None,
                        help="FF++ compression level, e.g. c23 (default) or c40")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None,
                        help="DataLoader worker processes. Use 0 on Kaggle to avoid fork errors.")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lambda1", type=float, default=None)
    parser.add_argument("--lambda2", type=float, default=None)
    parser.add_argument("--lambda_consistency", type=float, default=None,
                        help="Weight for consistency regularization loss (default 0.3). "
                             "MSE between augmented-branch and clean-branch probs.")
    parser.add_argument("--heatmap_samples", type=int, default=None)
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--eval_after_train", action="store_true", default=None)
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--attn_temp_init", type=float, default=None)
    parser.add_argument("--attn_diversity_weight", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--cls_dropout_p", type=float, default=None)
    parser.add_argument("--cls_loss_type", type=str, default=None,
                        choices=["bce", "focal"])
    parser.add_argument("--focal_alpha", type=float, default=None)
    parser.add_argument("--focal_gamma", type=float, default=None)
    parser.add_argument("--grad_accum_steps", type=int, default=None)
    parser.add_argument("--use_amp", dest="use_amp", action="store_true", default=None)
    parser.add_argument("--no_amp", dest="use_amp", action="store_false")
    parser.add_argument("--amp_dtype", type=str, default=None, choices=["fp16", "bf16"])
    parser.add_argument("--grad_checkpoint", dest="grad_checkpoint", action="store_true", default=None)
    parser.add_argument("--no_grad_checkpoint", dest="grad_checkpoint", action="store_false")
    parser.add_argument("--clip_grad_norm", type=float, default=None)
    parser.add_argument("--label_smoothing", type=float, default=None,
                        help="Label smoothing applied to BCE/focal loss target (0.05 = maps 0->0.05, 1->0.95)")
    parser.add_argument("--max_per_class", type=int, default=None,
                        help="If > 0, subsample train set to this many samples per class (balanced 1k/1k)")
    parser.add_argument("--skip_eval", action="store_true", default=False,
                        help="If set, skip post-training evaluation (useful for mid-run Kaggle sessions)")
    parser.add_argument("--active_manipulation", type=str, default=None,
                        choices=["Deepfakes", "Face2Face", "FaceShifter",
                                 "FaceSwap", "NeuralTextures"],
                        help="Required: specialist manipulation type to train on.")
    parser.add_argument("--celebdf_root", type=str, default=None,
                        help="Path to Celeb-DF v2 dataset root.")
    parser.add_argument("--celebdf_eval", action="store_true", default=None,
                        help="Run Celeb-DF v2 test evaluation after FF++ test eval.")
    parser.add_argument("--hidf_root", type=str, default=None,
                        help="HiDF dataset root (contains Real-vid/ and Fake-vid/)")
    parser.add_argument("--ffpp_cross_eval", action="store_true", default=None,
                        help="Run FF++ per-manipulation cross-evaluation after training")
    parser.add_argument("--ffpp_cross_root", type=str, default=None,
                        help="FF++ ffpp_data/ root for cross-evaluation")
    parser.add_argument("--hidf_split_seed", type=int, default=None,
                        help="Seed for HiDF source-grouped train/val/test split")
    parser.add_argument("--save_last_checkpoint", action="store_true", default=None,
                        help="Save last_checkpoint.pth after every epoch (for multi-session resume).")
    parser.add_argument("--explanation_suite", dest="explanation_suite",
                        action="store_true", default=None,
                        help="Run explanation metrics suite after evaluation.")
    parser.add_argument("--no_explanation_suite", dest="explanation_suite",
                        action="store_false")
    parser.add_argument("--early_stop_patience", type=int, default=None,
                        help="Epochs without improvement before early stopping (default 5).")
    parser.add_argument("--early_stop_metric", type=str, default=None,
                        help="Metric to monitor for early stopping (default val_balanced_accuracy).")
    parser.add_argument("--early_stop_min_delta", type=float, default=None,
                        help="Minimum improvement to count for early stopping (default 0.001).")
    parser.add_argument("--save_heatmaps", dest="save_heatmaps",
                        action="store_true", default=None,
                        help="Save heatmap PNGs and MP4 overlays after evaluation.")
    parser.add_argument("--no_save_heatmaps", dest="save_heatmaps",
                        action="store_false")
    parser.add_argument("--phase21_enabled", dest="phase21_enabled",
                        action="store_true", default=None,
                        help="Enable Phase 21 faithful attention bottleneck (default True).")
    parser.add_argument("--no_phase21_enabled", dest="phase21_enabled",
                        action="store_false")
    parser.add_argument("--lambda_faith", type=float, default=None,
                        help="Weight for Phase 21 faithfulness KL loss (default 0.3).")
    parser.add_argument("--lambda_sparse", type=float, default=None,
                        help="Weight for Phase 21 sparsity (negative peak) loss (default 0.05).")
    parser.add_argument("--faith_warmup_epochs", type=int, default=None,
                        help="Epochs to linearly ramp lambda_faith from 0 (default 3).")
    parser.add_argument("--attn_floor", type=float, default=None,
                        help="Gate floor for EarlyAttnHead (default 0.05).")
    parser.add_argument("--blur_kernel", type=int, default=None,
                        help="Gaussian kernel size for bottlenecked input (default 21).")
    parser.add_argument("--blur_sigma", type=float, default=None,
                        help="Gaussian sigma for bottlenecked input (default 10.0).")
    parser.add_argument("--bottleneck_peak_floor", type=float, default=None,
                        help="Phase 22: fixed absolute floor for bottleneck mask normalisation "
                             "(default 0.25). Diffuse maps whose peak < floor get blurred harder, "
                             "so the faithfulness loss bites even before sharpening converges.")
    parser.add_argument("--snapshot_every", type=int, default=None,
                        help="Save Phase 21 snapshot every N epochs (default 2).")
    parser.add_argument("--no_early_stop", dest="no_early_stop",
                        action="store_true", default=None,
                        help="Disable early stopping — run all epochs regardless of metric plateau.")
    parser.add_argument("--lambda_peak_spread", type=float, default=None,
                        help="Weight for HardAttentionDiversityLoss (default 0.5).")
    parser.add_argument("--lambda_sharp", type=float, default=None,
                        help="Weight for sharpness loss on M_t_logits (default 1.0).")
    parser.add_argument("--disk_guard_gb", type=float, default=None,
                        help="Min free disk GB before face-cache write is skipped (default 3.0).")
    return parser.parse_args()
