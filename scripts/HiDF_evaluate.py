"""
scripts/evaluate.py — Full evaluation: detection + explanation metrics + heatmaps.

Output directory layout (canonical, Task 1.4):
  {output_dir}/eval/ffpp_test_metrics.json    — FF++ detection metrics (primary)
  {output_dir}/eval/metrics.json              — backward-compat copy (deprecated)
  {output_dir}/eval/celebdf_test_metrics.json — Celeb-DF cross-domain metrics
  {output_dir}/eval/report.txt               — human-readable summary
  {output_dir}/metrics.csv                   — combined detection+explanation CSV
  {output_dir}/explanation_metrics.json      — intrinsic explanation quality metrics
  {output_dir}/heatmaps/                     — multi-frame strip PNGs + MP4 overlays
  {output_dir}/plots/heatmaps/               — per-frame single-frame XAI overlay PNGs
  {output_dir}/plots/ffpp_*.png              — FF++ detection visualizations
  {output_dir}/plots/celebdf_*.png           — Celeb-DF detection visualizations

Key fixes vs original:
  1. Checkpoint loading uses weights_only=False (PyTorch 2.6+ fix).
  2. GradCAM IndexError fixed via _ScalarOutputTarget in xai/gradcam.py.
  3. faithfulness_correlation receives tensors of matching shape (subset, K)
     — both intrinsic maps and gradient maps are averaged over T before
     flattening.  No shape mismatch.
  4. Deletion/Insertion AUC is now computed (not placeholder zeros) on the
     heatmap subset using metrics/explanation.py.
  5. Video reading falls back gracefully when video path is unavailable
     (synthetic dataset).
  6. ROC, PR, confusion-matrix, score-distribution PNGs are saved.
  7. Per-video plain-English explanation TXT files are saved.
  8. Heatmap videos use annotated overlay with verdict text and green contour.
"""

import os
import csv
import contextlib
import torch
import numpy as np
from pathlib import Path as _PPath
from tqdm import tqdm
from torch.utils.data import DataLoader

from HiDF_config import EAHNConfig
from models.HiDF_eahn import EAHN
from data.HiDF_datasets import DeepfakeDataset
from data.HiDF_collate import deepfake_collate_fn
from metrics.HiDF_detection import DetectionMetrics
from metrics.HiDF_explanation import ExplanationMetrics
from utils.HiDF_checkpointing import load_checkpoint
from utils.HiDF_visualization import (
    save_annotated_frame_strip,
    save_explanation_video,
    overlay_heatmap_on_frame,
    get_region_label,
)
import cv2


# ── Phase 19.9: Test-time augmentation helper ────────────────────────────────────────────

def run_tta_forward(model, frames, n_tta: int = 4, device: str = "cpu"):
    """
    Run N augmented forward passes and average probabilities.

    frames : Tensor (B, T, 3, H, W) already on device.
    Returns averaged prob tensor (B,).
    4 augmented passes + 1 clean pass = 5 total.
    """
    all_probs = []
    for _ in range(n_tta):
        # Horizontal flip with p=0.5
        if torch.rand(1).item() > 0.5:
            frames_aug = torch.flip(frames, dims=[-1])
        else:
            frames_aug = frames
        # Mild brightness jitter (stay within normalised range)
        frames_aug = frames_aug + torch.randn_like(frames_aug) * 0.02
        frames_aug = frames_aug.clamp(-3, 3)

        with torch.no_grad():
            out = model(frames_aug)
        all_probs.append(out.prob.squeeze(-1))

    # Also include the clean (no augmentation) pass
    with torch.no_grad():
        out_clean = model(frames)
    all_probs.append(out_clean.prob.squeeze(-1))

    return torch.stack(all_probs, dim=0).mean(dim=0)


# ── Phase 19.2: Standalone balanced Celeb-DF sample ──────────────────────────────────────

class _CelebDFBalancedSample(torch.utils.data.Dataset):
    """Inline 100R+100F balanced sample of Celeb-DF v2.

    Lists .mp4 files directly from Celeb-real / YouTube-real / Celeb-synthesis
    and samples a fixed-seed balanced subset. Independent of
    List_of_testing_videos.txt. Mirrors the frames+label+meta contract used
    by DeepfakeDataset.
    """

    def __init__(self, root, num_frames, frame_size, face_aligner, transform,
                 cache_dir, n_per_class=100, seed=42):
        import glob
        import random
        rng = random.Random(seed)
        real_paths = sorted(
            glob.glob(os.path.join(root, "Celeb-real",   "*.mp4")) +
            glob.glob(os.path.join(root, "YouTube-real", "*.mp4"))
        )
        fake_paths = sorted(glob.glob(os.path.join(root, "Celeb-synthesis", "*.mp4")))
        if len(real_paths) < n_per_class:
            raise RuntimeError(
                f"Celeb-DF balanced sample: only {len(real_paths)} real "
                f"videos found, need {n_per_class}"
            )
        if len(fake_paths) < n_per_class:
            raise RuntimeError(
                f"Celeb-DF balanced sample: only {len(fake_paths)} fake "
                f"videos found, need {n_per_class}"
            )
        rng.shuffle(real_paths)
        rng.shuffle(fake_paths)
        self.items = (
            [(p, 0) for p in real_paths[:n_per_class]] +
            [(p, 1) for p in fake_paths[:n_per_class]]
        )
        self.num_frames   = num_frames
        self.frame_size   = frame_size
        self.face_aligner = face_aligner
        self.transform    = transform
        self.cache_dir    = cache_dir

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        video_path, label = self.items[idx]

        frames_np = self._read_frames(video_path)

        video_id  = os.path.splitext(os.path.basename(video_path))[0]
        frames_np = self.face_aligner.align_frames(frames_np, video_id)

        from PIL import Image as _Image
        frames_tensor = torch.stack(
            [self.transform(_Image.fromarray(f)) for f in frames_np]
        )  # (T, 3, H, W)

        return {
            "frames": frames_tensor,
            "label":  label,
            "meta":   {"video_path": video_path, "frame_indices": []},
        }

    def _read_frames(self, video_path: str) -> list:
        T = self.num_frames
        try:
            from decord import VideoReader as _VR, cpu as _decord_cpu
            vr      = _VR(video_path, ctx=_decord_cpu(0))
            total   = len(vr)
            indices = np.linspace(0, total - 1, T, dtype=int).tolist()
            batch   = vr.get_batch(indices).asnumpy()
            return [batch[i] for i in range(T)]
        except Exception:
            pass
        cap    = cv2.VideoCapture(video_path)
        total  = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        target = set(np.linspace(0, total - 1, T, dtype=int).tolist())
        frames = []
        fi     = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if fi in target:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            fi += 1
            if len(frames) == T:
                break
        cap.release()
        if not frames:
            frames = [np.zeros((self.frame_size, self.frame_size, 3), dtype=np.uint8)]
        while len(frames) < T:
            frames.append(frames[-1].copy())
        return frames



# ── Detection graph helper ────────────────────────────────────────────────────

def plot_evaluation_visuals(
    probs,
    labels,
    out_dir: str,
    prefix: str,
    optimal_thr: float = 0.5,
) -> None:
    """
    Save all standard detection visualisations for one evaluation split.

    Task 2.2: refactored from save_detection_graphs so it can be called for
    both FF++ (prefix="ffpp") and Celeb-DF (prefix="celebdf").

    Outputs (under out_dir/):
        {prefix}_roc.png
        {prefix}_pr.png
        {prefix}_confusion.png
        {prefix}_confusion_norm.png
        {prefix}_score_distribution.png
        {prefix}_summary_chart.png    (placeholder — full chart via summary_chart.py)

    Parameters
    ----------
    probs       : array-like of float probabilities
    labels      : array-like of int labels (0=real, 1=fake)
    out_dir     : destination directory (will be created if needed)
    prefix      : "ffpp" | "celebdf" or any short identifier
    optimal_thr : optimal decision threshold (shown as second vertical line)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        roc_curve, roc_auc_score,
        precision_recall_curve, average_precision_score,
        confusion_matrix,
    )
    import seaborn as sns

    os.makedirs(out_dir, exist_ok=True)
    probs  = np.array(probs)
    labels = np.array(labels)
    preds  = (probs >= 0.5).astype(int)

    label_set = np.unique(labels)
    if len(label_set) < 2:
        print(f"[plot_evaluation_visuals] Skipping {prefix}: only one class in labels.")
        return

    tag = prefix.upper()

    # 1 — ROC Curve
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"EAHN  AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {tag}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{prefix}_roc.png"), dpi=150)
    plt.close(fig)

    # 2 — Precision-Recall Curve
    prec, rec, _ = precision_recall_curve(labels, probs)
    ap = average_precision_score(labels, probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, lw=2, color="darkorange", label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {tag}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{prefix}_pr.png"), dpi=150)
    plt.close(fig)

    # 3a — Confusion Matrix (raw counts)
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    ax.set_ylabel("Ground Truth")
    ax.set_xlabel("Predicted")
    ax.set_title(f"Confusion Matrix — {tag}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{prefix}_confusion.png"), dpi=150)
    plt.close(fig)

    # 3b — Confusion Matrix (row-normalised)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1e-8)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", ax=ax,
                xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"],
                vmin=0.0, vmax=1.0)
    ax.set_ylabel("Ground Truth")
    ax.set_xlabel("Predicted")
    ax.set_title(f"Confusion Matrix (Normalised) — {tag}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{prefix}_confusion_norm.png"), dpi=150)
    plt.close(fig)

    # 4 — Score Distribution (thr=0.5 + optimal threshold)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(probs[labels == 0], bins=30, alpha=0.6, label="Real", color="steelblue")
    ax.hist(probs[labels == 1], bins=30, alpha=0.6, label="Fake", color="tomato")
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.5, label="thr = 0.5")
    if abs(optimal_thr - 0.5) > 0.01:
        ax.axvline(optimal_thr, color="purple", linestyle=":",
                   linewidth=1.5, label=f"optimal thr = {optimal_thr:.2f}")
    ax.set_xlabel("Predicted Probability (Deepfake)")
    ax.set_ylabel("Count")
    ax.set_title(f"Score Distribution — {tag}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{prefix}_score_distribution.png"), dpi=150)
    plt.close(fig)

    # 5 — Summary chart (reuses plots/summary_chart.py format)
    # Produce a simple bar chart with the 5 headline numbers
    from sklearn.metrics import balanced_accuracy_score
    bal_acc = float(balanced_accuracy_score(labels, preds))
    real_acc = float((preds[labels == 0] == 0).mean()) if (labels == 0).sum() > 0 else 0.0
    fake_acc = float((preds[labels == 1] == 1).mean()) if (labels == 1).sum() > 0 else 0.0
    metrics_bar = {"AUC-ROC": auc, "AP": ap, "Bal.Acc": bal_acc,
                   "Real Acc": real_acc, "Fake Acc": fake_acc}
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(list(metrics_bar.keys()), list(metrics_bar.values()),
                  color=["royalblue", "darkorange", "mediumseagreen", "steelblue", "tomato"])
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1, alpha=0.5)
    for bar, val in zip(bars, metrics_bar.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_title(f"Detection Summary — {tag}")
    ax.set_ylabel("Score")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{prefix}_summary_chart.png"), dpi=150)
    plt.close(fig)

    print(f"[Evaluate] {tag} visuals saved → {out_dir}  (prefix={prefix})")


def save_detection_graphs(probs, labels, output_dir: str) -> None:
    """Backward-compat wrapper → delegates to plot_evaluation_visuals with prefix='ffpp'."""
    plot_evaluation_visuals(probs, labels,
                            out_dir=output_dir, prefix="ffpp", optimal_thr=0.5)
    # Also write legacy filenames so any existing consumers don't break
    import shutil as _shu
    _legacy = {
        "ffpp_roc.png": "roc_curve.png",
        "ffpp_pr.png": "pr_curve.png",
        "ffpp_confusion.png": "confusion_matrix.png",
        "ffpp_confusion_norm.png": "confusion_matrix_norm.png",
        "ffpp_score_distribution.png": "score_distribution.png",
    }
    for src_name, dst_name in _legacy.items():
        _src = os.path.join(output_dir, src_name)
        if os.path.exists(_src):
            _shu.copy2(_src, os.path.join(output_dir, dst_name))


# ── Manipulation-type parser (Phase 15) ──────────────────────────────────────

def parse_manipulation(video_path: str) -> str:
    """
    Infer manipulation type from FF++ video path.

    Handles paths of the form:
      .../manipulated_sequences/<TYPE>/c23/videos/XXX_YYY.mp4  → <TYPE>
      .../original_sequences/youtube/c23/videos/ZZZ.mp4        → "real"
      anything else (synthetic, unknown)                        → "unknown"

    Expected return values:
      {"Deepfakes", "Face2Face", "FaceShifter", "FaceSwap",
       "NeuralTextures", "real", "unknown"}
    """
    parts = _PPath(video_path).parts
    if "manipulated_sequences" in parts:
        idx = parts.index("manipulated_sequences")
        return parts[idx + 1]
    elif "original_sequences" in parts:
        return "real"
    return "unknown"


# ── Main evaluation entry point ───────────────────────────────────────────────

def run_evaluation(config: EAHNConfig, breakdown_by_manipulation: bool = False):
    device = torch.device(config.device)

    # ── Load model ────────────────────────────────────────────────────────────
    model     = EAHN(config).to(device)
    ckpt_path = os.path.join(config.output_dir, f"eahn_{config.dataset_name}_best.pth")
    if not os.path.exists(ckpt_path):
        import glob as _glob
        candidates = sorted(_glob.glob(
            os.path.join(config.output_dir, "checkpoint_epoch*.pth")
        ))
        if candidates:
            ckpt_path = candidates[-1]
            print(f"[Eval] per-dataset checkpoint not found — using {ckpt_path}")
        else:
            raise FileNotFoundError(
                f"No checkpoint found in {config.output_dir}. "
                "Did training complete without errors?"
            )
    load_checkpoint(ckpt_path, model)
    model.eval()
    print("Loaded best model for evaluation.")

    # ── Test dataset ─────────────────────────────────────────────────────────
    test_ds = DeepfakeDataset(config, "test", config.dataset_name)
    test_loader = DataLoader(
        test_ds, batch_size=config.batch_size,
        num_workers=config.num_workers, collate_fn=deepfake_collate_fn,
    )
    print(f"[DataLoader test] batch_size={config.batch_size}  shuffle=False  size={len(test_ds)}")

    # ── Detection pass ────────────────────────────────────────────────────────
    # GPU-resident M_t buffer (Phase 22 OOM fix): keep on GPU and reuse across
    # collapse-diagnostics, explanation suite, overlays.  T4 has 15 GB; this
    # tensor is ~1.3 GB at most for 415 × 16 × 224 × 224 float32.
    all_probs, all_labels = [], []
    all_vid_paths        = []
    _M_chunks            = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating detection"):
            frames = batch["frames"].to(device, non_blocking=True)
            out    = model(frames)
            all_probs.extend(out.prob.detach().cpu().tolist())
            all_labels.extend(batch["label"].cpu().tolist())
            all_vid_paths.extend(batch.get("video_path", [""] * frames.shape[0]))
            _M_chunks.append(out.M_t_up.detach())             # keep GPU
            del frames, out

    all_M_t_up = torch.cat(_M_chunks, dim=0)                  # (N_test, T, H, W) GPU
    del _M_chunks

    det_metrics = DetectionMetrics.compute(all_probs, all_labels)
    print("Detection Metrics:", det_metrics)

    # Phase 19.9: calibrated threshold for cross-dataset eval
    hidf_optimal_threshold = float(det_metrics.get("optimal_threshold", 0.5))
    print(f"[Calibration] HiDF optimal threshold = {hidf_optimal_threshold:.4f}  "
          f"(will be used for F1/balanced-accuracy in all cross-dataset blocks)")

    # ── Confusion matrix (5a) ─────────────────────────────────────────────────
    from sklearn.metrics import confusion_matrix as sk_confusion_matrix
    try:
        preds_arr = (np.array(all_probs) >= 0.5).astype(int)
        cm        = sk_confusion_matrix(np.array(all_labels, dtype=int), preds_arr)
        tn, fp, fn, tp = cm.ravel()
    except Exception:
        tn = fp = fn = tp = 0

    # ── Save detection graphs + structured outputs to eval/ subdir ───────────
    eval_dir   = os.path.join(config.output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    # Task 2.2: Save detection visuals to plots/ subdirectory
    plots_dir  = os.path.join(config.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    labels_arr = np.array(all_labels)
    _opt_thr   = float(det_metrics.get("optimal_threshold", 0.5))
    if len(np.unique(labels_arr)) >= 2:
        plot_evaluation_visuals(all_probs, all_labels,
                                out_dir=plots_dir, prefix="ffpp",
                                optimal_thr=_opt_thr)
        # Also copy legacy names to eval_dir for backward compat (Cell 9)
        import shutil as _shutil
        _legacy_map = {
            "ffpp_roc.png": "roc_curve.png",
            "ffpp_pr.png": "pr_curve.png",
            "ffpp_confusion.png": "confusion_matrix.png",
            "ffpp_confusion_norm.png": "confusion_matrix_norm.png",
            "ffpp_score_distribution.png": "score_distribution.png",
        }
        for _src_name, _dst_name in _legacy_map.items():
            _src = os.path.join(plots_dir, _src_name)
            if os.path.exists(_src):
                _shutil.copy2(_src, os.path.join(config.output_dir, _dst_name))
    else:
        print("[Evaluate] Skipping detection graphs — only one class in test set.")

    # ── Split counts + summary chart (5b, 5c, 5d) ────────────────────────────
    train_ds_tmp = DeepfakeDataset(config, "train", config.dataset_name)
    val_ds_tmp   = DeepfakeDataset(config, "val",   config.dataset_name)
    split_counts = {
        "total":      len(train_ds_tmp) + len(val_ds_tmp) + len(test_ds),
        "train":      len(train_ds_tmp),
        "train_real": train_ds_tmp.n_real,
        "train_fake": train_ds_tmp.n_fake,
        "val":        len(val_ds_tmp),
        "test":       len(test_ds),
        "test_real":  test_ds.n_real,
        "test_fake":  test_ds.n_fake,
    }
    metrics_dict_full = {
        **det_metrics,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }
    from scripts.HiDF_summary_chart import plot_summary_chart
    plot_summary_chart(metrics_dict_full, split_counts, config.output_dir)

    # ── Explanation metrics on a subset ──────────────────────────────────────
    subset_size = min(config.heatmap_samples, len(test_ds))
    rng     = np.random.default_rng(42)
    indices = rng.choice(len(test_ds), subset_size, replace=False)

    # Temporal SSIM
    ssim_val = ExplanationMetrics.temporal_ssim(all_M_t_up[indices])

    # Faithfulness correlation (gradient saliency vs intrinsic maps)
    grad_maps = []
    for idx in tqdm(indices, desc="Computing faithfulness", leave=False):
        sample      = test_ds[idx]
        frames_t    = sample["frames"].unsqueeze(0).to(device)
        frames_t.requires_grad_(True)
        out         = model(frames_t)
        out.logit.backward()
        grads       = frames_t.grad.abs().mean(dim=2)  # (1, T, H, W) avg over RGB
        grads_7 = torch.nn.functional.interpolate(
            grads.reshape(grads.shape[1], 1, *grads.shape[2:]),  # (T, 1, H, W)
            size=(7, 7), mode="bilinear", align_corners=False,
        ).squeeze(1)                                              # (T, 7, 7)
        grad_maps.append(grads_7.detach().cpu())
        frames_t.requires_grad_(False)

    grad_maps = torch.stack(grad_maps)             # (subset, T, 7, 7)

    M_sub     = all_M_t_up[indices].mean(dim=1)   # (subset, H, W)
    M_sub_7   = torch.nn.functional.interpolate(
        M_sub.unsqueeze(1), size=(7, 7), mode="bilinear", align_corners=False
    ).squeeze(1)                                   # (subset, 7, 7)

    grad_7_avg = grad_maps.mean(dim=1)             # (subset, 7, 7)

    faithful_corr = ExplanationMetrics.faithfulness_correlation(
        M_sub_7.reshape(subset_size, -1),
        grad_7_avg.reshape(subset_size, -1),
    )

    # ── Deletion / Insertion AUC (up to 5 heatmap samples) ─────────────────────
    # Uses 5 clips for statistical reliability vs. 1-clip original.
    # Root cause of historic 0.0 values: _gauss_blur used .expand() (non-contiguous)
    # which F.conv2d (grouped) rejects on some PyTorch versions → now fixed with
    # .contiguous() in metrics/HiDF_explanation.py.
    import traceback as _traceback
    del_ins = {"deletion_auc": 0.0, "insertion_auc": 0.0}
    _n_del_ins = min(subset_size, 5)   # 5 clips: reliable AUC, GPU-friendly
    try:
        _di_frames_list = [test_ds[int(indices[i])]["frames"]
                           for i in range(_n_del_ins)]          # each (T,C,H,W)
        _frames_stack   = torch.stack(_di_frames_list, dim=0)   # (N,T,C,H,W)
        _sal_indices    = [int(indices[i]) for i in range(_n_del_ins)]
        _sal_stack      = all_M_t_up[_sal_indices]              # (N,T,H,W) GPU tensor
        _sal_np         = _sal_stack.detach().cpu().numpy()
        del_ins = ExplanationMetrics.deletion_insertion_auc(
            model, _frames_stack, _sal_np, steps=10, n_samples=_n_del_ins
        )
    except Exception as e:
        print(f"  [Deletion/Insertion AUC error: {e}]")
        _traceback.print_exc()

    # ── Collapse diagnostics ──────────────────────────────────────────────────
    collapse_diag = ExplanationMetrics.collapse_diagnostics(all_M_t_up)
    print("Collapse Diagnostics:", collapse_diag)

    # Print a warning if any guardrail trips
    warnings_list = []
    if collapse_diag["inter_sample_cosine_mean"] > 0.95:
        warnings_list.append(
            f"inter_sample_cosine_mean={collapse_diag['inter_sample_cosine_mean']:.3f} (threshold 0.95)"
        )
    if collapse_diag["peak_mode_share"] > 0.5:
        warnings_list.append(
            f"peak_mode_share={collapse_diag['peak_mode_share']:.3f} (threshold 0.5)"
        )
    if collapse_diag["m_t_std_mean"] > 0.13:
        warnings_list.append(
            f"m_t_std_mean={collapse_diag['m_t_std_mean']:.3f} (threshold 0.13)"
        )
    if warnings_list:
        print("\n[COLLAPSE WARNING] Explanation collapse detected:")
        for w in warnings_list:
            print(f"  - {w}")
        print("  Do NOT proceed to longer runs. Diagnose the explanation head first.\n")

    # ── Adebayo model-randomization sanity check ─────────────────────────────
    # Phase 22.4: sentinel 1.0 removed — it was polluting metrics.csv with a
    # fake value.  The real mt_vs_random_model_cosine is computed ONCE by
    # run_explanation_suite (seed=42, n_random=30) and written to
    # explanation_metrics.json["intrinsic"]["mt_vs_random_model_cosine"].
    # Do NOT add it back here as a placeholder; read from that JSON if needed.

    # ── Frame attention drop test (k=1, 2, 4 frames zeroed out) ──────────────
    # Faithfulness test: dropping the top-K attended frames should reduce model
    # confidence more than dropping K random frames. ratio > 1.5 = faithful.
    print("\n[FrameDropTest] Computing k1/k2/k4 attention-drop faithfulness test...")
    frame_drop_results = {}
    try:
        frame_drop_results = ExplanationMetrics.frame_attention_drop_test(
            model, test_loader, device, k_values=(1, 2, 4), seed=42
        )
        print("[FrameDropTest] Results (top-K vs random-K drop):")
        for _k in (1, 2, 4):
            _td = frame_drop_results.get(f"k{_k}_top_conf_drop",    0.0)
            _rd = frame_drop_results.get(f"k{_k}_random_conf_drop", 0.0)
            _rt = frame_drop_results.get(f"k{_k}_ratio",            0.0)
            _tag = "✓" if _rt > 1.5 else ("~" if _rt > 1.0 else "✗")
            print(f"  k={_k}: top_drop={_td:+.4f}  random_drop={_rd:+.4f}  "
                  f"ratio={_rt:.2f}x  {_tag}")
        _k1_ratio = frame_drop_results.get("k1_ratio", 0.0)
        if _k1_ratio > 1.5:
            print("  [FrameDropTest] ✓ Attention maps target discriminative frames.")
        elif _k1_ratio > 1.0:
            print("  [FrameDropTest] ~ Attention maps marginally better than random.")
        else:
            print("  [FrameDropTest] ✗ Attention maps NOT more discriminative than random.")
    except Exception as _fdt_err:
        print(f"  [FrameDropTest skipped: {_fdt_err}]")
        import traceback as _fdt_tb; _fdt_tb.print_exc()

    # ── Assemble explanation metrics for CSV ──────────────────────────────────
    # Only scalar floats go to metrics.csv.  Per-step del/ins values give thesis
    # readers the confidence curve without needing to re-run evaluation.
    _del_ins_scalars = {
        "deletion_auc":  del_ins.get("deletion_auc",  0.0),
        "insertion_auc": del_ins.get("insertion_auc", 0.0),
        "del_at_10pct":  del_ins.get("del_at_10pct",  0.0),
        "del_at_50pct":  del_ins.get("del_at_50pct",  0.0),
        "ins_at_10pct":  del_ins.get("ins_at_10pct",  0.0),
        "ins_at_50pct":  del_ins.get("ins_at_50pct",  0.0),
    }
    exp_metrics = {
        "temporal_ssim":     ssim_val,
        "faithfulness_corr": faithful_corr,
        **_del_ins_scalars,
        **collapse_diag,
        **frame_drop_results,   # k1/k2/k4_top_conf_drop, _random_conf_drop, _ratio
    }
    print("Explanation Metrics:", exp_metrics)

    # ── Mean heatmap entropy (lower = more focused) ───────────────────────────
    def _entropy(m: np.ndarray) -> float:
        flat = m.flatten().astype(np.float64) + 1e-12
        flat = flat / flat.sum()
        return float(-(flat * np.log(flat)).sum())

    h_mean = float(np.mean([
        _entropy(all_M_t_up[i].mean(0).detach().cpu().numpy())
        for i in range(len(all_M_t_up))
    ]))

    # ── Save metrics CSV ──────────────────────────────────────────────────────
    os.makedirs(config.output_dir, exist_ok=True)
    csv_path = os.path.join(config.output_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in {**det_metrics, **exp_metrics}.items():
            writer.writerow([k, v])
    print(f"Metrics saved to {csv_path}")

    # ── metrics.json + report.txt → eval/ subdir ─────────────────────────────
    import json as _json
    N_total = len(all_labels)
    N_real  = int(sum(1 for l in all_labels if l == 0))
    N_fake  = int(sum(1 for l in all_labels if l == 1))
    auc_roc   = float(det_metrics.get("auc_roc",   0.0))
    auc_pr    = float(det_metrics.get("auc_pr",    0.0))
    f1        = float(det_metrics.get("f1_at_0.5", 0.0))
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    _preds_at_05 = (np.array(all_probs) >= 0.5).astype(int)
    _labels_arr  = np.array(all_labels)
    acc  = float(accuracy_score(_labels_arr, _preds_at_05))
    prec = float(precision_score(_labels_arr, _preds_at_05, zero_division=0))
    rec  = float(recall_score(_labels_arr, _preds_at_05, zero_division=0))
    ins_auc   = float(del_ins.get("insertion_auc", 0.0))
    del_auc   = float(del_ins.get("deletion_auc",  0.0))

    # Per-class and threshold-optimal metrics (CHANGE 3)
    real_acc  = float(det_metrics.get("real_accuracy",              0.0))
    fake_acc  = float(det_metrics.get("fake_accuracy",              0.0))
    bal_acc   = float(det_metrics.get("balanced_accuracy",          0.0))
    opt_thr   = float(det_metrics.get("optimal_threshold",          0.5))
    f1_opt    = float(det_metrics.get("f1_at_optimal",             0.0))
    bal_opt   = float(det_metrics.get("balanced_accuracy_at_optimal", 0.0))

    metrics_json = {
        "auc_roc":                      auc_roc,
        "auc_pr":                       auc_pr,
        "f1_at_0.5":                    f1,
        "balanced_accuracy":            bal_acc,
        "real_accuracy":                real_acc,
        "fake_accuracy":                fake_acc,
        "optimal_threshold":            opt_thr,
        "f1_at_optimal":               f1_opt,
        "balanced_accuracy_at_optimal": bal_opt,
        "accuracy": acc, "precision": prec, "recall": rec,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "threshold": 0.5,
        "h_mean": h_mean,
        "temporal_ssim": float(ssim_val),
        "insertion_auc": ins_auc,
        "deletion_auc":  del_auc,
        # Per-step del/ins confidence curve
        "del_at_10pct":  del_ins.get("del_at_10pct",  0.0),
        "del_at_50pct":  del_ins.get("del_at_50pct",  0.0),
        "ins_at_10pct":  del_ins.get("ins_at_10pct",  0.0),
        "ins_at_50pct":  del_ins.get("ins_at_50pct",  0.0),
        # Frame attention drop test
        **{k: v for k, v in frame_drop_results.items()},
    }
    metrics_json["active_manipulation"] = getattr(config, "active_manipulation", "")
    # Task 1.5: primary filename is ffpp_test_metrics.json; keep metrics.json as
    # a backward-compatibility copy (deprecated — remove after one release cycle).
    json_path       = os.path.join(eval_dir, "ffpp_test_metrics.json")
    json_compat_path = os.path.join(eval_dir, "metrics.json")
    with open(json_path, "w") as f:
        _json.dump(metrics_json, f, indent=2)
    import shutil as _shutil_json
    _shutil_json.copy2(json_path, json_compat_path)   # backward-compat copy
    print(f"[Evaluate] ffpp_test_metrics.json saved → {json_path}")
    print(f"[Evaluate] metrics.json (compat copy) → {json_compat_path}")

    faithful_str = "YES — heatmap IS predictive" if ins_auc > del_auc else "NO — heatmap not predictive"
    # Per-step del/ins curve lines for report
    _del10  = del_ins.get("del_at_10pct",  "—")
    _del50  = del_ins.get("del_at_50pct",  "—")
    _ins10  = del_ins.get("ins_at_10pct",  "—")
    _ins50  = del_ins.get("ins_at_50pct",  "—")
    _d10s   = f"{_del10:.4f}" if isinstance(_del10, float) else _del10
    _d50s   = f"{_del50:.4f}" if isinstance(_del50, float) else _del50
    _i10s   = f"{_ins10:.4f}" if isinstance(_ins10, float) else _ins10
    _i50s   = f"{_ins50:.4f}" if isinstance(_ins50, float) else _ins50
    # Frame drop lines
    _fdr_lines = ""
    for _k in (1, 2, 4):
        _td  = frame_drop_results.get(f"k{_k}_top_conf_drop",    None)
        _rd  = frame_drop_results.get(f"k{_k}_random_conf_drop", None)
        _rt  = frame_drop_results.get(f"k{_k}_ratio",            None)
        if _td is not None:
            _tag = "✓" if (_rt or 0) > 1.5 else ("~" if (_rt or 0) > 1.0 else "✗")
            _fdr_lines += (f"  k={_k}: top_drop={_td:+.4f}  rand_drop={_rd:+.4f}  "
                           f"ratio={_rt:.2f}x  {_tag}\n")
    if not _fdr_lines:
        _fdr_lines = "  (not computed)\n"

    report = (
        "EAHN Detection Report\n"
        "---------------------\n"
        f"Tested {N_total} videos ({N_real} real, {N_fake} fake).\n"
        f"AUC-ROC: {auc_roc:.3f}    AUC-PR: {auc_pr:.3f}    F1 (thr=0.5): {f1:.3f}\n"
        f"Per-class accuracy: Real={real_acc:.3f}  Fake={fake_acc:.3f}\n"
        f"Balanced accuracy: {bal_acc:.3f} (at thr=0.5)  |  {bal_opt:.3f} (at optimal thr={opt_thr:.3f})\n"
        f"F1 at optimal threshold: {f1_opt:.3f}\n"
        f"At threshold 0.5:\n"
        f"  True positives  (fakes caught) : {tp}/{N_fake}\n"
        f"  False negatives (fakes missed) : {fn}/{N_fake}\n"
        f"  True negatives  (real correct) : {tn}/{N_real}\n"
        f"  False positives (real flagged) : {fp}/{N_real}\n"
        f"\n"
        f"Explanation quality:\n"
        f"  Mean heatmap entropy : {h_mean:.3f}    (lower = more focused)\n"
        f"  Temporal SSIM        : {ssim_val:.3f}      (1.0 = frozen across time)\n"
        f"  Faithfulness corr    : {faithful_corr:.3f}  (gradient vs. intrinsic)\n"
        f"  Insertion AUC        : {ins_auc:.4f}  (higher = better)\n"
        f"  Deletion AUC         : {del_auc:.4f}  (lower = better)\n"
        f"  Faithful? {faithful_str}\n"
        f"\n"
        f"Del/Ins confidence curve:\n"
        f"  10% removed  → del_conf={_d10s}  ins_conf={_i10s}\n"
        f"  50% removed  → del_conf={_d50s}  ins_conf={_i50s}\n"
        f"\n"
        f"Frame attention drop test (top-K vs random-K frames zeroed):\n"
        f"{_fdr_lines}"
    )
    report_path = os.path.join(eval_dir, "report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[Evaluate] report.txt saved → {report_path}")
    print(report)

    # ── Per-manipulation breakdown (Phase 15) ─────────────────────────────────
    if breakdown_by_manipulation:
        # Manipulation labels in DataLoader iteration order (shuffle=False → same as samples order)
        _bd_manipulations = [parse_manipulation(s["video_path"]) for s in test_ds.samples]

        # Per-sample peak (row, col) from mean M_t
        _M_mean = all_M_t_up.mean(dim=1)           # (N, H, W) GPU
        _bd_peak_rows: list[int] = []
        _bd_peak_cols: list[int] = []
        for _i in range(len(_M_mean)):
            _m    = _M_mean[_i].detach().cpu().numpy()
            _pidx = np.unravel_index(_m.argmax(), _m.shape)
            _bd_peak_rows.append(int(_pidx[0]))
            _bd_peak_cols.append(int(_pidx[1]))

        _probs_np  = np.array(all_probs)
        _labels_np = np.array(all_labels)
        _real_mask = _labels_np == 0
        _real_idxs = np.where(_real_mask)[0]

        _FAKE_TYPES = ["Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]
        _bd_result: dict = {}

        for _mtype in _FAKE_TYPES + ["real"]:
            _idxs = [i for i, m in enumerate(_bd_manipulations) if m == _mtype]
            _n    = len(_idxs)
            if _n == 0:
                continue

            _probs_m = _probs_np[_idxs]
            _pr_m    = [_bd_peak_rows[i] for i in _idxs]
            _pc_m    = [_bd_peak_cols[i] for i in _idxs]

            _entry: dict = {
                "n":             _n,
                "mean_prob":     float(np.mean(_probs_m)),
                "peak_row_mean": float(np.mean(_pr_m)),
                "peak_col_mean": float(np.mean(_pc_m)),
                "peak_row_std":  float(np.std(_pr_m)),
                "peak_col_std":  float(np.std(_pc_m)),
            }

            if _mtype != "real":
                # AUC: this fake type paired with ALL real test samples
                _comb_probs  = np.concatenate([_probs_np[_real_idxs], _probs_m])
                _comb_labels = np.concatenate([np.zeros(len(_real_idxs)), np.ones(_n)])
                if _n >= 5 and len(np.unique(_comb_labels)) >= 2:
                    from sklearn.metrics import roc_auc_score as _roc_auc_bd
                    try:
                        _entry["auc_roc"] = float(_roc_auc_bd(_comb_labels, _comb_probs))
                    except Exception as _e_bd:
                        print(f"[Breakdown] AUC computation failed for {_mtype}: {_e_bd}")
                        _entry["auc_roc"] = None
                else:
                    if _n < 5:
                        print(f"[Breakdown] Skipping AUC for {_mtype}: n={_n} < 5 (threshold)")
                    _entry["auc_roc"] = None
                _entry["fake_acc_at_0.5"]     = float(np.mean(_probs_m >= 0.5))
                _entry["fake_acc_at_optimal"] = float(np.mean(_probs_m >= opt_thr))
            else:
                _entry["real_acc_at_0.5"]     = float(np.mean(_probs_m < 0.5))
                _entry["real_acc_at_optimal"] = float(np.mean(_probs_m < opt_thr))

            _bd_result[_mtype] = _entry

        # — Add to ffpp_test_metrics.json (and keep compat copy in sync) —
        metrics_json["breakdown_by_manipulation"] = _bd_result
        with open(json_path, "w") as _jf:
            _json.dump(metrics_json, _jf, indent=2)
        _shutil_json.copy2(json_path, json_compat_path)
        print(f"[Evaluate] ffpp_test_metrics.json updated with breakdown → {json_path}")

        # — Append breakdown table to report.txt —
        _bd_lines = [
            "",
            "─── Per-manipulation breakdown ───",
            f"{'Type':<18} {'n':>5}  {'AUC':>6}  {'fake@0.5':>8}  {'fake@opt':>8}  peak(r,c)±std",
        ]
        for _mtype in _FAKE_TYPES:
            if _mtype not in _bd_result:
                continue
            _e     = _bd_result[_mtype]
            _auc_s = f"{_e['auc_roc']:.3f}" if _e.get("auc_roc") is not None else "   N/A"
            _bd_lines.append(
                f"{_mtype:<18} {_e['n']:>5}  {_auc_s:>6}  "
                f"{_e['fake_acc_at_0.5']:>8.3f}  {_e['fake_acc_at_optimal']:>8.3f}  "
                f"({_e['peak_row_mean']:.1f},{_e['peak_col_mean']:.1f})"
                f"±({_e['peak_row_std']:.1f},{_e['peak_col_std']:.1f})"
            )
        if "real" in _bd_result:
            _e = _bd_result["real"]
            _bd_lines.append(
                f"{'real':<18} {_e['n']:>5}  {'--':>6}  "
                f"real@0.5={_e['real_acc_at_0.5']:.3f}  "
                f"real@opt={_e['real_acc_at_optimal']:.3f}  "
                f"({_e['peak_row_mean']:.1f},{_e['peak_col_mean']:.1f})"
                f"±({_e['peak_row_std']:.1f},{_e['peak_col_std']:.1f})"
            )
        _bd_text = "\n".join(_bd_lines) + "\n"
        with open(report_path, "a", encoding="utf-8") as _rf:
            _rf.write(_bd_text)
        print(f"[Evaluate] Per-manipulation breakdown appended to report.txt")
        print(_bd_text)

        # — Write CSV —
        _csv_bd_path = os.path.join(eval_dir, "breakdown_by_manipulation.csv")
        _csv_fields  = [
            "type", "n", "auc_roc", "mean_prob",
            "fake_acc_at_0.5", "fake_acc_at_optimal",
            "real_acc_at_0.5", "real_acc_at_optimal",
            "peak_row_mean", "peak_col_mean", "peak_row_std", "peak_col_std",
        ]
        _bd_rows = []
        for _mtype in _FAKE_TYPES + ["real"]:
            if _mtype not in _bd_result:
                continue
            _e = _bd_result[_mtype]
            _bd_rows.append({
                "type":                _mtype,
                "n":                   _e["n"],
                "auc_roc":             _e.get("auc_roc"),
                "mean_prob":           _e.get("mean_prob"),
                "fake_acc_at_0.5":     _e.get("fake_acc_at_0.5"),
                "fake_acc_at_optimal": _e.get("fake_acc_at_optimal"),
                "real_acc_at_0.5":     _e.get("real_acc_at_0.5"),
                "real_acc_at_optimal": _e.get("real_acc_at_optimal"),
                "peak_row_mean":       _e.get("peak_row_mean"),
                "peak_col_mean":       _e.get("peak_col_mean"),
                "peak_row_std":        _e.get("peak_row_std"),
                "peak_col_std":        _e.get("peak_col_std"),
            })
        with open(_csv_bd_path, "w", newline="", encoding="utf-8") as _cf:
            _csv_writer = csv.DictWriter(_cf, fieldnames=_csv_fields)
            _csv_writer.writeheader()
            _csv_writer.writerows(_bd_rows)
        print(f"[Evaluate] Breakdown CSV → {_csv_bd_path}")

    # ── Celeb-DF v2 cross-evaluation (Phase 19.2) ───────────────────────────────
    if getattr(config, "celebdf_eval", False) and getattr(config, "celebdf_root", ""):
        import json as _json
        from data.HiDF_transforms import get_transforms as _get_transforms
        from data.HiDF_face_align import FaceAligner as _FaceAligner

        print("\n" + "=" * 70)
        print("[Celeb-DF cross-eval] balanced 100R+100F sample (Phase 19)")
        print("=" * 70)

        _transform = _get_transforms("test", config.frame_size)

        _face_aligner = _FaceAligner(
            margin=0.30,
            cache_dir=None,           # Phase 19 fix: no disk cache during cross-eval
            device="cpu",
        )

        _celeb_ds = _CelebDFBalancedSample(
            root=config.celebdf_root,
            num_frames=config.num_frames,
            frame_size=config.frame_size,
            face_aligner=_face_aligner,
            transform=_transform,
            cache_dir=None,
            n_per_class=100,
            seed=42,
        )
        print(f"[Celeb-DF cross-eval] dataset size = {len(_celeb_ds)} "
              f"(100R + 100F balanced)")

        for _i, (_path, _lbl) in enumerate(_celeb_ds.items[:3]):
            print(f"[Celeb-DF cross-eval] sample[{_i}] label={_lbl} path={_path}")

        _loader = DataLoader(
            _celeb_ds,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=deepfake_collate_fn,
            num_workers=config.num_workers,
        )

        _probs_list, _labels_list = [], []
        model.eval()
        for _batch in _loader:
            _frames = _batch["frames"].to(device)
            _labels = _batch["label"]
            _avg_probs = run_tta_forward(model, _frames, n_tta=4, device=str(device))
            _probs_list.extend(_avg_probs.cpu().tolist())
            _labels_list.extend(_labels.cpu().tolist())

        _m = DetectionMetrics.compute(_probs_list, _labels_list)
        _m["n_samples"] = len(_probs_list)
        _m["sampling"] = "balanced_100R_100F_seed42"
        _m["tta_passes"] = 5  # 4 augmented + 1 clean
        _m["calibrated_threshold_used"] = hidf_optimal_threshold

        # Extra metrics at HiDF calibrated threshold
        _probs_np_cd = np.array(_probs_list)
        _labels_np_cd = np.array(_labels_list, dtype=int)
        _preds_cal = (_probs_np_cd >= hidf_optimal_threshold).astype(int)
        from sklearn.metrics import f1_score as _f1_score, balanced_accuracy_score as _bal_acc_score
        _m["f1_at_hidf_threshold"] = float(_f1_score(_labels_np_cd, _preds_cal, zero_division=0))
        _m["balanced_accuracy_at_hidf_threshold"] = float(_bal_acc_score(_labels_np_cd, _preds_cal))

        _celebdf_json = os.path.join(eval_dir, "celebdf_test_metrics.json")
        with open(_celebdf_json, "w") as _f:
            _json.dump(_m, _f, indent=2)

        try:
            plot_evaluation_visuals(
                _probs_list, _labels_list, plots_dir,
                prefix="celebdf",
                optimal_thr=_m.get("optimal_threshold", 0.5),
            )
        except Exception as _plot_err:
            print(f"[Celeb-DF cross-eval] plot failed: {_plot_err}")

        print(f"[Celeb-DF cross-eval] AUC-ROC = {_m.get('auc_roc', 0):.4f}  "
              f"AUC-PR = {_m.get('auc_pr', 0):.4f}  "
              f"F1@opt = {_m.get('f1_at_optimal', 0):.4f}")
        print(f"[Celeb-DF cross-eval] metrics saved -> {_celebdf_json}")

        # ── Stage cleanup: release CelebDF working memory before FF++ ─────────
        try:
            del _celeb_ds, _loader, _probs_list, _labels_list
            del _face_aligner, _transform, _m
            del _probs_np_cd, _labels_np_cd, _preds_cal
        except Exception:
            pass
        import gc as _gc_cd
        _gc_cd.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print("[Stage Cleanup] CelebDF buffers released")


    # ────────────────────────────────────────────────────────────────────
    # FF++ per-manipulation cross-evaluation (Phase 19.1 — bug fix)
    # Replaces the Phase 18.4 block which evaluated the same data 5 times.
    # ────────────────────────────────────────────────────────────────────
    if getattr(config, "ffpp_cross_eval", False) and getattr(config, "ffpp_cross_root", ""):
        import copy as _copy
        import json as _json
        from pathlib import Path as _Path

        print("\n" + "=" * 70)
        print("[FF++ cross-eval] starting per-manipulation evaluation (Phase 19)")
        print("=" * 70)

        FFPP_METHODS = ["Deepfakes", "Face2Face", "FaceShifter",
                        "FaceSwap", "NeuralTextures"]

        _eval_root = _Path(config.output_dir) / "eval"
        _eval_root.mkdir(parents=True, exist_ok=True)

        for _manip in FFPP_METHODS:
            print(f"\n[FF++ cross-eval] === {_manip} ===")

            # CRITICAL: fresh accumulators every iteration
            _probs_list = []
            _labels_list = []

            # CRITICAL: fresh config every iteration (deep-copy + explicit overrides)
            _tcfg = _copy.deepcopy(config)
            _tcfg.dataset_name        = "ff++"
            _tcfg.data_root           = config.ffpp_cross_root
            _tcfg.active_manipulation = _manip
            _tcfg.max_per_class       = 100
            _tcfg.ffpp_cross_eval     = False
            _tcfg.celebdf_eval        = False
            _tcfg.explanation_suite   = False
            _tcfg.cache_dir           = None   # no disk cache during cross-eval — avoids filling /kaggle/working

            # CRITICAL: fresh dataset construction every iteration
            _ffpp_ds = DeepfakeDataset(_tcfg, "test", "ff++")
            print(f"[FF++ cross-eval] {_manip}: dataset size = {len(_ffpp_ds)}")

            # SANITY: print first 3 sample paths so we can prove the data differs
            _first3 = _ffpp_ds.samples[:3]
            for _i, _s in enumerate(_first3):
                _path = _s["video_path"] if isinstance(_s, dict) else _s[0]
                _lbl  = _s["label"]      if isinstance(_s, dict) else _s[1]
                print(f"[FF++ cross-eval] {_manip}: sample[{_i}] label={_lbl} path={_path}")

            _loader = DataLoader(
                _ffpp_ds,
                batch_size=config.batch_size,
                shuffle=False,
                collate_fn=deepfake_collate_fn,
                num_workers=config.num_workers,
            )

            # Forward loop — TTA (4 augmented + 1 clean) for cross-dataset robustness
            model.eval()
            for _batch in _loader:
                _frames = _batch["frames"].to(device)
                _labels = _batch["label"]
                _avg_probs = run_tta_forward(model, _frames, n_tta=4, device=str(device))
                _probs_list.extend(_avg_probs.cpu().tolist())
                _labels_list.extend(_labels.cpu().tolist())

            print(f"[FF++ cross-eval] {_manip}: collected {len(_probs_list)} predictions")
            print(f"[FF++ cross-eval] {_manip}: prob[min,mean,max] = "
                  f"[{min(_probs_list):.4f}, {sum(_probs_list)/len(_probs_list):.4f}, "
                  f"{max(_probs_list):.4f}]")

            _m = DetectionMetrics.compute(_probs_list, _labels_list)
            _m["manipulation"] = _manip
            _m["n_samples"] = len(_probs_list)
            _m["tta_passes"] = 5  # 4 augmented + 1 clean
            _m["calibrated_threshold_used"] = hidf_optimal_threshold

            # Extra metrics at HiDF calibrated threshold
            _probs_np_ff = np.array(_probs_list)
            _labels_np_ff = np.array(_labels_list, dtype=int)
            _preds_cal_ff = (_probs_np_ff >= hidf_optimal_threshold).astype(int)
            from sklearn.metrics import f1_score as _f1_score_ff, balanced_accuracy_score as _bal_acc_ff
            _m["f1_at_hidf_threshold"] = float(_f1_score_ff(_labels_np_ff, _preds_cal_ff, zero_division=0))
            _m["balanced_accuracy_at_hidf_threshold"] = float(_bal_acc_ff(_labels_np_ff, _preds_cal_ff))

            _cell_dir = _eval_root / f"ffpp_cross_{_manip}"
            _cell_dir.mkdir(exist_ok=True)
            with open(_cell_dir / "metrics.json", "w") as _f:
                _json.dump(_m, _f, indent=2)

            try:
                plot_evaluation_visuals(
                    _probs_list, _labels_list, str(_cell_dir),
                    prefix=f"ffpp_cross_{_manip}",
                    optimal_thr=_m.get("optimal_threshold", 0.5),
                )
            except Exception as _plot_err:
                print(f"[FF++ cross-eval] {_manip}: plot failed: {_plot_err}")

            print(f"[FF++ cross-eval] {_manip}: AUC-ROC = {_m.get('auc_roc', 0):.4f}  "
                  f"AUC-PR = {_m.get('auc_pr', 0):.4f}  F1@opt = {_m.get('f1_at_optimal', 0):.4f}")

            # ── Per-iteration cleanup: release this manipulation's working set ──
            try:
                del _ffpp_ds, _loader, _probs_list, _labels_list, _tcfg, _m
                del _probs_np_ff, _labels_np_ff, _preds_cal_ff
            except Exception:
                pass
            import gc as _gc_ff
            _gc_ff.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        print("\n[FF++ cross-eval] done.")
        print("[Stage Cleanup] FF++ cross-eval buffers released")


    # ── Explanation suite ────────────────────────────────────────────────────
    if getattr(config, "explanation_suite", True):
        # Pre-suite full sweep: ensure CPU RAM is at minimum before allocating
        # the suite's targeted frame buffer.  Reuses all_M_t_up_gpu from the
        # detection pass so no duplicate forward pass is needed.
        import gc as _gc_pre
        _gc_pre.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            _free_gb = (torch.cuda.get_device_properties(0).total_memory
                        - torch.cuda.memory_reserved()) / 1e9
            print(f"[Pre-Suite] GPU free: ~{_free_gb:.2f} GB")
        try:
            from scripts.HiDF_run_explanation_suite import run_explanation_suite
            from scripts.HiDF_save_xai_overlays import save_xai_overlays
            _exp_out_path = _PPath(config.output_dir) / "explanation_metrics.json"
            run_explanation_suite(
                model, test_loader, config, _exp_out_path,
                all_M_t_up_gpu=all_M_t_up,
                all_probs=all_probs,
                all_labels=all_labels,
                all_vid_paths=all_vid_paths,
            )
            # Inter-stage cleanup before overlays
            import gc as _gc_post
            _gc_post.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            _overlay_dir = _PPath(config.output_dir) / "plots" / "heatmaps"
            save_xai_overlays(model, test_loader, config, _overlay_dir)
        except Exception as _suite_err:
            print(f"[ExplanationSuite] skipped: {_suite_err}")
            import traceback as _suite_tb; _suite_tb.print_exc()

    # ── Heatmap generation ────────────────────────────────────────────────────
    if config.save_heatmaps:
        _generate_heatmaps(config, model, test_ds, indices[:5], device, all_probs,
                           batch_inter_sample_sim=collapse_diag["inter_sample_cosine_mean"])

    # ── Representative heatmaps (C.6) ────────────────────────────────────────
    # Pick 1 confidently-real-correct, 1 confidently-fake-correct, 1 misclassified
    _save_representative_heatmaps(
        config, model, test_ds, all_probs, all_labels, device,
        batch_inter_sample_sim=collapse_diag["inter_sample_cosine_mean"],
        temporal_ssim=ssim_val,
        inter_sample_cosine=collapse_diag["inter_sample_cosine_mean"],
    )

    # ── Task 2.3: Auto-bundle analysis essentials ─────────────────────────────
    if getattr(config, "active_manipulation", ""):
        try:
            from scripts.HiDF_package_analysis_bundle import package_analysis_bundle
            _bundle_dir = os.path.join(config.output_dir, "analysis_essentials")
            package_analysis_bundle(
                output_dir=config.output_dir,
                manipulation=config.active_manipulation,
                bundle_dir=_bundle_dir,
                do_zip=False,
            )
        except Exception as _bundle_err:
            print(f"[Bundle] analysis bundle skipped: {_bundle_err}")

    print("Evaluation complete. Outputs saved to", config.output_dir)


# ── Heatmap + explanation helper ─────────────────────────────────────────────

def _generate_heatmaps(config, model, test_ds, sample_indices, device, all_probs,
                       batch_inter_sample_sim: float = 0.0):
    from xai.HiDF_gradcam import GradCAMExplainer
    from xai.HiDF_attention_rollout import AttentionRolloutExplainer
    from xai.HiDF_shap_explainer import SHAPExplainer

    # Canonical layout (Task 1.4):
    #   heatmaps/       → multi-frame strip PNGs + MP4 overlay videos
    #   plots/heatmaps/ → per-frame single-frame XAI overlay PNGs
    heatmap_dir = os.path.join(config.output_dir, "heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)

    gradcam_exp = GradCAMExplainer(model, target_layer=model.spatial_stream.grad_cam_target_layer)
    rollout_exp = AttentionRolloutExplainer(model)
    shap_exp    = SHAPExplainer(model, method="integratedgrads")

    print("Generating heatmaps and explanations...")
    for idx in tqdm(sample_indices, desc="Saving heatmap videos"):
        idx    = int(idx)
        sample = test_ds[idx]
        frames_tensor = sample["frames"].unsqueeze(0).to(device)

        video_path = sample["meta"].get("video_path", "")
        video_id   = os.path.splitext(os.path.basename(video_path))[0] if video_path else str(idx)

        sampled_orig = _denormalize_aligned_tensor_to_bgr(sample["frames"])

        with torch.no_grad():
            out = model(frames_tensor)
        intrinsic = out.M_t_up[0].cpu().numpy()     # (T, H, W)
        prob      = float(out.prob[0].cpu())
        verdict   = "FAKE" if prob > 0.5 else "REAL"

        # Convert intrinsic to list form for new viz API
        intrinsic_maps = [intrinsic[t] for t in range(intrinsic.shape[0])]

        def _peakiness(m: np.ndarray) -> float:
            flat = m.flatten().astype(np.float64) + 1e-12
            flat = flat / flat.sum()
            H_val = -(flat * np.log(flat)).sum()
            return float(1.0 - H_val / np.log(flat.size))

        intrinsic_scores = [_peakiness(m) for m in intrinsic_maps]

        # ── Annotated frame strip + companion text explanation (5f) ───────────
        # Strips go to heatmaps/ (canonical layout, Task 1.4)
        save_annotated_frame_strip(
            sampled_orig, intrinsic_maps, intrinsic_scores, verdict, prob,
            os.path.join(heatmap_dir, f"{video_id}_strip.png"),
            sample_id=video_id,
            batch_inter_sample_sim=batch_inter_sample_sim,
            active_manipulation=getattr(config, "active_manipulation", ""),
        )

        # ── Intrinsic explanation video (5f) ──────────────────────────────────
        save_explanation_video(
            sampled_orig, intrinsic_maps, intrinsic_scores, verdict, prob,
            os.path.join(heatmap_dir, f"{video_id}_intrinsic.mp4"),
        )

        # ── Post-hoc heatmaps ─────────────────────────────────────────────────
        for method_name, explainer in [
            ("gradcam", gradcam_exp),
            ("rollout", rollout_exp),
            ("shap",    shap_exp),
        ]:
            try:
                if method_name == "gradcam":
                    heat = explainer.explain(frames_tensor)[0]   # (T,H,W) numpy
                else:
                    heat = explainer.explain(frames_tensor)      # (T,H,W) numpy
            except Exception as e:
                print(f"  [{method_name} failed for idx {idx}: {e}]")
                heat = intrinsic

            maps_list   = [heat[t] for t in range(heat.shape[0])]
            scores_list = [float(m.max()) for m in maps_list]
            save_explanation_video(
                sampled_orig, maps_list, scores_list, verdict, prob,
                os.path.join(heatmap_dir, f"{video_id}_{method_name}.mp4"),
            )


def _save_representative_heatmaps(
    config, model, test_ds, all_probs, all_labels, device,
    batch_inter_sample_sim: float = 0.0,
    temporal_ssim: float = 0.0,
    inter_sample_cosine: float = 0.0,
):
    """
    Save heatmap overlay MP4 + frame strip PNG + plain-English summary TXT for
    three representative test videos:
      - 1 confidently-real-correct   (true real, predicted real, prob < 0.2)
      - 1 confidently-fake-correct   (true fake, predicted fake, prob > 0.8)
      - 1 misclassified              (any, predicted wrong)

    Outputs go to  OUTPUT_DIR/heatmaps/heatmap_overlay_{video_id}.{mp4,png,txt}
    """
    heatmap_dir = os.path.join(config.output_dir, "heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)

    probs_arr  = np.array(all_probs)
    labels_arr = np.array(all_labels, dtype=int)
    preds_arr  = (probs_arr >= 0.5).astype(int)

    def _find(condition_mask, max_tries=50):
        idxs = np.where(condition_mask)[0]
        if len(idxs) == 0:
            return None
        # Pick the one with most extreme probability (most confident)
        best = idxs[np.argmax(np.abs(probs_arr[idxs] - 0.5))]
        return int(best)

    candidates = {
        "real_correct": _find((labels_arr == 0) & (preds_arr == 0) & (probs_arr < 0.2)),
        "fake_correct": _find((labels_arr == 1) & (preds_arr == 1) & (probs_arr > 0.8)),
        "misclassified": _find(labels_arr != preds_arr),
    }
    # Fill gaps with any available sample
    for key in list(candidates.keys()):
        if candidates[key] is None:
            candidates[key] = _find(labels_arr >= 0)

    saved = []
    for role, idx in candidates.items():
        if idx is None:
            continue
        sample        = test_ds[idx]
        frames_tensor = sample["frames"].unsqueeze(0).to(device)
        video_path    = sample["meta"].get("video_path", "")
        video_id      = (
            os.path.splitext(os.path.basename(video_path))[0]
            if video_path else f"sample_{idx}"
        )
        video_id = f"{role}_{video_id}"

        orig_frames = _denormalize_aligned_tensor_to_bgr(sample["frames"])

        with torch.no_grad():
            out = model(frames_tensor)
        intrinsic      = out.M_t_up[0].cpu().numpy()   # (T, H, W)
        prob           = float(out.prob[0].cpu())
        verdict        = "FAKE" if prob > 0.5 else "REAL"
        confidence     = abs(prob - 0.5) * 2.0
        intrinsic_maps = [intrinsic[t] for t in range(intrinsic.shape[0])]

        def _peakiness(m: np.ndarray) -> float:
            flat = m.flatten().astype(np.float64) + 1e-12
            flat = flat / flat.sum()
            return float(1.0 - (-(flat * np.log(flat)).sum()) / np.log(flat.size))

        intrinsic_scores = [_peakiness(m) for m in intrinsic_maps]

        # Collapse flags for this single sample
        sp_stds  = [float(m.std()) for m in intrinsic_maps]
        is_uniform = float(np.mean(sp_stds)) < 0.01
        if len(intrinsic_maps) > 1:
            f0  = intrinsic_maps[0].flatten();  f0  = f0  / (np.linalg.norm(f0)  + 1e-8)
            fl  = intrinsic_maps[-1].flatten(); fl  = fl  / (np.linalg.norm(fl)  + 1e-8)
            is_frozen = float(np.dot(f0, fl)) > 0.99
        else:
            is_frozen = False
        is_class_agnostic = inter_sample_cosine > 0.95

        # ── MP4 overlay ─────────────────────────────────────────────────
        mp4_path = os.path.join(heatmap_dir, f"heatmap_overlay_{video_id}.mp4")
        save_explanation_video(
            orig_frames, intrinsic_maps, intrinsic_scores, verdict, prob, mp4_path
        )

        # ── Frame strip PNG ──────────────────────────────────────────────
        png_path = os.path.join(heatmap_dir, f"heatmap_strip_{video_id}.png")
        save_annotated_frame_strip(
            orig_frames, intrinsic_maps, intrinsic_scores, verdict, prob,
            png_path, sample_id=video_id,
            batch_inter_sample_sim=batch_inter_sample_sim,
            active_manipulation=getattr(config, "active_manipulation", ""),
        )

        # ── Plain-English summary TXT ────────────────────────────────────
        peak_t  = int(np.argmax(intrinsic_scores))
        mean_map = np.mean(intrinsic_maps, axis=0)
        region   = get_region_label(mean_map)

        health_notes = []
        if is_uniform:
            health_notes.append("Attention is spatially uniform (possible collapse).")
        if is_frozen:
            health_notes.append("Attention map frozen across frames (possible collapse).")
        if is_class_agnostic:
            health_notes.append("Heatmaps similar across all test samples (class-agnostic).")
        if not health_notes:
            health_notes.append(
                f"Heatmap varies across frames (temporal_ssim={temporal_ssim:.2f}) "
                f"and across samples (inter_sample_cosine={inter_sample_cosine:.2f}) "
                f"— explanation looks healthy."
            )

        frame_range = f"frames {min(range(len(intrinsic_maps)), key=lambda t: intrinsic_scores[t])+1}–"
        frame_range += f"{max(range(len(intrinsic_maps)), key=lambda t: intrinsic_scores[t])+1}"
        summary_lines = [
            f"Role: {role}",
            f"Video: {video_path}",
            f"Ground truth: {'FAKE' if all_labels[idx] == 1 else 'REAL'}",
            f"Model predicted {verdict} with confidence {confidence:.2f} (prob={prob:.3f}).",
            f"Attention focused on the {region}.",
            f"Peak attention at t={peak_t+1} ({frame_range}).",
        ] + health_notes
        txt_path = os.path.join(heatmap_dir, f"heatmap_summary_{video_id}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(summary_lines) + "\n")

        saved.append(video_id)
        print(f"[Representative] {role} → {video_id}  prob={prob:.3f}  verdict={verdict}")

    print(f"[Representative heatmaps] Saved {len(saved)} videos: {saved}")


def _denormalize_aligned_tensor_to_bgr(frames_tensor) -> list:
    """
    Convert a face-aligned, ImageNet-normalized tensor (T, 3, H, W) float32
    to a list of uint8 BGR numpy arrays (H, W, 3), one per frame.

    This is the inverse of the torchvision Normalize(mean, std) transform
    applied by DeepfakeDataset, so the underlay image matches exactly what
    the model consumed — not the raw video frame.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    x = frames_tensor.detach().cpu().float()   # (T, 3, H, W)
    x = (x * std + mean).clamp(0.0, 1.0) * 255.0
    x = x.permute(0, 2, 3, 1).numpy().astype(np.uint8)  # (T, H, W, 3) RGB
    return [frame[:, :, ::-1].copy() for frame in x]    # → BGR list


def _get_original_frames(video_path: str, num_frames: int, frame_size: int):
    """Read original BGR frames; falls back to blank frames if path unavailable."""
    if not video_path or not os.path.exists(video_path):
        return [np.zeros((frame_size, frame_size, 3), np.uint8)] * num_frames

    cap   = cv2.VideoCapture(video_path)
    total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    idxs  = np.linspace(0, total - 1, num_frames, dtype=int)
    buf   = {}
    fi    = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fi in set(idxs.tolist()):
            buf[fi] = cv2.resize(frame, (frame_size, frame_size))
        fi += 1
    cap.release()
    blank = np.zeros((frame_size, frame_size, 3), np.uint8)
    return [buf.get(i, blank) for i in idxs]


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse as _ap_main
    import sys as _sys_main

    # Parse evaluate-only flags first, before config's parse_args consumes sys.argv
    _eval_parser = _ap_main.ArgumentParser(add_help=False)
    _eval_parser.add_argument(
        "--breakdown_by_manipulation",
        action="store_true",
        default=False,
        help=(
            "Segment test predictions by manipulation type (Deepfakes, Face2Face, "
            "FaceShifter, FaceSwap, NeuralTextures, real). Computes per-type "
            "AUC-ROC, fake/real accuracy at thr=0.5 and optimal threshold, and "
            "mean peak (row, col) of M_t. Adds 'breakdown_by_manipulation' to "
            "metrics.json, appends a table to report.txt, and writes "
            "outputs/eval/breakdown_by_manipulation.csv."
        ),
    )
    _eval_ns, _remaining_argv = _eval_parser.parse_known_args()

    # Patch sys.argv so config.parse_args() sees only the standard flags
    _sys_main.argv = [_sys_main.argv[0]] + _remaining_argv

    from HiDF_config import parse_args as _parse_config_args, EAHNConfig as _EAHNConfig
    _args   = _parse_config_args()
    _config = _EAHNConfig.from_args(_args)

    run_evaluation(_config, breakdown_by_manipulation=_eval_ns.breakdown_by_manipulation)
