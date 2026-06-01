# EAHN — Explanation-Aware Hybrid Network for Deepfake Detection

## Quick Start

### Local synthetic smoke test (no GPU, no data needed)
```bash
pip install -r requirements.txt
python run_full_pipeline.py \
    --dataset_name synthetic \
    --epochs 2 \
    --batch_size 2 \
    --output_dir outputs_synthetic/
```

### Kaggle Setup

**Dataset directory layout expected on disk** (Kaggle mount):
```
{data_root}/
├── manipulated_sequences/
│   ├── Deepfakes/c23/videos/*.mp4        ← label = 1 (fake)
│   ├── Face2Face/c23/videos/*.mp4        ← label = 1 (fake)
│   ├── FaceShifter/c23/videos/*.mp4      ← label = 1 (fake)
│   ├── FaceSwap/c23/videos/*.mp4         ← label = 1 (fake)
│   └── NeuralTextures/c23/videos/*.mp4   ← label = 1 (fake)
└── original_sequences/
    └── youtube/c23/videos/*.mp4          ← label = 0 (real)
```

> **No mask files exist** in this dataset version.  
> All training uses **weak supervision** (entropy + total variation loss) — `has_masks` is always `False`.

**Step 1 — Verify dataset before training:**
```bash
python scripts/verify_dataset.py \
    --data_root /kaggle/input/datasets/umardrazbhatti/ffpp-c23-custom-layout/ffpp_data
```
This prints a directory table and runs a forward pass. Exit code 0 = ready to train.

**Step 2 — Train:**
```python
%cd /kaggle/working
!git clone https://github.com/umardrazbhatti/EahnCode.git
%cd EahnCode
!pip install -r requirements.txt

!python run_full_pipeline.py \
    --data_root /kaggle/input/datasets/umardrazbhatti/ffpp-c23-custom-layout/ffpp_data \
    --dataset_name ff++ \
    --dataset_compression c23 \
    --epochs 10 \
    --batch_size 4 \
    --num_workers 0 \
    --eval_after_train
```

**Expected outputs in `/kaggle/working/outputs/`:**
```
outputs/
├── best_model.pth
├── metrics.csv
├── roc_curve.png             ← ROC curve with AUC annotation
├── pr_curve.png              ← Precision-Recall curve with AP annotation
├── confusion_matrix.png      ← 2×2 heatmap
├── score_distribution.png    ← Real vs fake score histogram
├── heatmaps/
│   └── {video_id}_{intrinsic,gradcam,rollout,shap}.mp4
└── explanations/
    └── {video_id}_explanation.txt   ← Plain-English per-video report
```

### Evaluation only (after training)
```python
from config import EAHNConfig
from scripts.evaluate import run_evaluation

config = EAHNConfig()
config.data_root           = "/kaggle/input/.../ffpp_data"
config.dataset_name        = "ff++"
config.dataset_compression = "c23"
config.output_dir          = "/kaggle/working/outputs"
config.device              = "cuda"
config.num_workers         = 0
config.heatmap_samples     = 5

run_evaluation(config)
```

### Dashboard
```python
from scripts.dashboard import show_dashboard
show_dashboard("/kaggle/working/outputs")
```

---

## Bug Fixes Applied (vs original GitHub code)

| # | File | Bug | Fix |
|---|------|-----|-----|
| 1 | `utils/checkpointing.py` | `torch.load` crashes with `weights_only=True` on PyTorch 2.6+ because checkpoint embeds numpy scalars and EAHNConfig dataclass | Added `weights_only=False`; documented that this is safe for our own checkpoints |
| 2 | `xai/gradcam.py` | `ClassifierOutputTarget(1)` raises `IndexError: index 1 is out of bounds for dim 0 with size 1` — binary classifier outputs scalar logit, not two-class softmax | Replaced with `_ScalarOutputTarget` that returns `output[:, 0]` or `output.sum()` |
| 3 | `scripts/evaluate.py` | `faithfulness_correlation` called with `grad_maps` of shape `(subset, T, 7, 7)` and `M_t_sub` of shape `(subset, 7, 7)` — shape mismatch | Time-average `grad_maps` over T axis before flattening; both inputs are now `(subset, 49)` |
| 4 | `scripts/train_real.py` | `autocast` used inside `torch.no_grad()` context in training loop; correct context manager needed | Replaced with `contextlib.nullcontext()` when not using AMP |
| 5 | `data/datasets.py` | Mask resize target `size=7` hardcoded inline; mismatch with actual feature grid; mask shape inconsistent across datasets | Centralised `_MASK_GRID = 7` constant; all masks returned as `(7, 7)` float tensor |
| 6 | `models/eahn.py` | `reshape` called on potentially non-contiguous tensor after `view` | Replaced all `view` with `reshape` throughout forward pass |
| 7 | `xai/attention_rollout.py` | Broken double `F.interpolate` call (dead code path) | Cleaned up to single `F.interpolate` on `(T,1,h,w)` input |
| 8 | `metrics/explanation.py` | `deletion_insertion_auc` was a placeholder returning zeros | Implemented simplified pixel-deletion/insertion loop |
| 9 | `scripts/train_real.py` | No LR scheduler despite thesis specifying CosineAnnealingLR | Added `CosineAnnealingLR` wired to training loop |

---

## Project Structure
```
eahn_project/
├── config.py                    # EAHNConfig dataclass + CLI override
├── requirements.txt
├── run_full_pipeline.py         # Entry point
├── README.md
├── data/
│   ├── datasets.py              # FF++, Celeb-DF, DFDC, Synthetic
│   ├── face_align.py            # MTCNN crop with tracking + disk cache
│   ├── transforms.py            # Augmentation + ImageNet normalisation
│   ├── synthetic_generator.py   # CPU-only synthetic deepfake generator
│   └── collate.py               # Custom collate for optional masks
├── models/
│   ├── eahn.py                  # EAHN: full model
│   ├── spatial_stream.py        # EfficientNet-B4/ConvNeXt wrapper
│   ├── temporal_stream.py       # 4-layer Transformer + CLS
│   └── cross_attention.py       # Cross-Attention Fusion → M_t
├── losses/
│   ├── classification.py        # BCE
│   ├── explanation.py           # MSE (supervised) or Entropy+TV (weak)
│   └── temporal.py              # Gated temporal consistency
├── xai/
│   ├── gradcam.py               # Grad-CAM (binary-classifier-safe)
│   ├── attention_rollout.py     # Attention rollout over Transformer layers
│   └── shap_explainer.py        # Integrated Gradients via Captum
├── metrics/
│   ├── detection.py             # AUC-ROC, AUC-PR, F1
│   └── explanation.py           # IoU, Temporal SSIM, Faithfulness, Del/Ins AUC
├── utils/
│   ├── checkpointing.py         # save / load (weights_only=False)
│   ├── logging_utils.py         # TensorBoard + CSV
│   └── visualization.py         # Overlay heatmaps; save MP4
├── scripts/
│   ├── train_synthetic.py       # Phase 1: CPU smoke test
│   ├── train_real.py            # Phase 2: GPU training
│   ├── evaluate.py              # Full evaluation pipeline
│   ├── dashboard.py             # Metrics table + bar chart + video display
│   └── data_analysis.py         # Class distribution statistics
└── user_study/
    └── generate_stimuli.py      # Conditions A, B, C for user study
```
