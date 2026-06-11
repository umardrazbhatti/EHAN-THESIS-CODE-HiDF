"""
scripts/save_xai_overlays.py — Save Grad-CAM + Attention-Rollout + intrinsic M_t
overlay PNGs for config.xai_overlay_videos selected test videos, split evenly
real/fake (Phase 30: default 50 = 25 real + 25 fake; was hardcoded 10).

Selection:
  Per class, in a 2:2:1 high:mid:low confidence ratio
  (50 videos → 10 high + 10 mid + 5 low per class).
  High:  prob >= 0.7 (fake) or prob <= 0.3 (real)
  Mid:   0.4 <= prob <= 0.6
  Low:   prob closest to 0.5

For each selected video, saves 4 frames (evenly spaced across T=16 input)
× 3 maps (intrinsic M_t, Grad-CAM, Attention-Rollout) as overlay PNGs.

Filename pattern:
  {video_id}_{label}_conf{prob:.2f}_{method}_f{frame_idx}.png

Does NOT invoke xai/shap_explainer.py.
"""

import os
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from utils.HiDF_visualization import overlay_heatmap_on_frame


def _ensure_uint8_bgr(img) -> np.ndarray:
    """
    Guarantee img is a uint8 numpy array with shape (H, W, 3) in BGR channel order.
    Handles: torch.Tensor (C,H,W or H,W,C), float arrays [0..1], and already-uint8 BGR.
    """
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    img = np.asarray(img)
    # CHW → HWC
    if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[0] < img.shape[1]:
        img = img.transpose(1, 2, 0)
    # Float [0,1] → uint8
    if img.dtype != np.uint8:
        img = (img.clip(0.0, 1.0) * 255.0).astype(np.uint8)
    return img


def _select_samples(probs, labels, n_high=2, n_mid=2, n_low=1):
    """
    Select n_high + n_mid + n_low indices per class.
    Returns dict {"real": [idx, ...], "fake": [idx, ...]}.

    Phase 30: n_low can be > 1 — the low bucket takes the n_low samples
    whose prob is closest to 0.5 (was: single closest).
    """
    probs  = np.array(probs)
    labels = np.array(labels, dtype=int)
    result = {}

    for cls_label, cls_name in [(0, "real"), (1, "fake")]:
        cls_idxs = np.where(labels == cls_label)[0]
        if len(cls_idxs) == 0:
            result[cls_name] = []
            continue

        cls_probs = probs[cls_idxs]
        if cls_label == 1:  # fake: higher prob = more confident
            high_mask = cls_probs >= 0.7
            mid_mask  = (cls_probs >= 0.4) & (cls_probs <= 0.6)
        else:              # real: lower prob = more confident
            high_mask = cls_probs <= 0.3
            mid_mask  = (cls_probs >= 0.4) & (cls_probs <= 0.6)

        # High confidence: sort by confidence descending
        high_idxs = cls_idxs[high_mask]
        if cls_label == 1:
            high_idxs = high_idxs[np.argsort(cls_probs[high_mask])[::-1]]
        else:
            high_idxs = high_idxs[np.argsort(cls_probs[high_mask])]
        selected = list(high_idxs[:n_high])

        # Mid confidence
        mid_idxs = cls_idxs[mid_mask]
        selected += list(mid_idxs[:n_mid])

        # Low confidence: the n_low samples closest to 0.5
        remaining = [i for i in cls_idxs if i not in set(selected)]
        if remaining and n_low > 0:
            rem_probs = probs[remaining]
            _order    = np.argsort(np.abs(rem_probs - 0.5))
            selected += [remaining[int(j)] for j in _order[:n_low]]

        # Pad or trim to n_high + n_mid + n_low
        target_n = n_high + n_mid + n_low
        if len(selected) < target_n:
            extra = [i for i in cls_idxs if i not in set(selected)]
            selected += list(extra[:target_n - len(selected)])
        selected = selected[:target_n]

        result[cls_name] = selected

    return result


def _denormalize(frames_tensor) -> list:
    """(T,3,H,W) normalised float → list of uint8 RGB ndarrays."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = frames_tensor.detach().cpu().float()
    x = (x * std + mean).clamp(0.0, 1.0) * 255.0
    x = x.permute(0, 2, 3, 1).numpy().astype(np.uint8)  # (T, H, W, 3) RGB
    return [x[t] for t in range(x.shape[0])]


def save_xai_overlays(model, test_loader, config, output_dir: Path):
    """
    Generate and save Grad-CAM + Attention-Rollout + intrinsic M_t overlay PNGs
    for config.xai_overlay_videos selected test videos, split evenly real/fake
    (Phase 30 default: 50 = 25 real + 25 fake, in a 2:2:1 high:mid:low
    confidence ratio per class).

    Args:
        model       : trained EAHN model
        test_loader : DataLoader for test set (no shuffle)
        config      : EAHNConfig
        output_dir  : Path where overlay PNGs will be saved
    """
    import cv2

    device = torch.device(config.device)
    model.eval()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Two-pass strategy (6-7-26 OOM fix) ───────────────────────────────────
    # Pass 1: build probs/labels/meta + per-sample M_t (M_t kept on GPU, only
    #         ~1.3 GB for 415 samples at 224×224 — fits in 15 GB T4).
    # Pass 2: re-iterate loader picking out frames ONLY for sampled indices
    #         (~10 videos = ~200 MB).  Avoids ~5 GB CPU RAM accumulation that
    #         was killing the run at "Suite Pass 414/415".
    all_probs   = []
    all_labels  = []
    all_meta    = []
    _M_chunks   = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="XAI overlay inference", leave=False):
            frames = batch["frames"].to(device, non_blocking=True)
            out    = model(frames)
            all_probs.extend(out.prob.detach().cpu().tolist())
            all_labels.extend(batch["label"].cpu().tolist())
            _M_chunks.append(out.M_t_up.detach())             # GPU
            all_meta.extend(batch["meta"])
            del frames, out

    all_M_t_up = torch.cat(_M_chunks, dim=0)                  # (N, T, H, W) GPU
    del _M_chunks

    # ── Select N/2 real + N/2 fake (Phase 30: config-driven, default 50) ─────
    _n_videos  = max(2, int(getattr(config, "xai_overlay_videos", 50)))
    _per_class = max(1, _n_videos // 2)
    # 2:2:1 high:mid:low confidence split per class (50 → 10/10/5 per class)
    _n_high = max(1, round(_per_class * 0.4))
    _n_mid  = max(1, round(_per_class * 0.4))
    _n_low  = max(0, _per_class - _n_high - _n_mid)
    selected = _select_samples(all_probs, all_labels,
                               n_high=_n_high, n_mid=_n_mid, n_low=_n_low)
    chosen_indices = selected.get("real", []) + selected.get("fake", [])
    print(f"[XAI overlays] Selected {len(chosen_indices)} videos: "
          f"real={len(selected.get('real',[]))} fake={len(selected.get('fake',[]))} "
          f"(per-class high/mid/low = {_n_high}/{_n_mid}/{_n_low})")

    # ── Pass 2: collect frames only for chosen indices ───────────────────────
    # Phase 30: frames stay on GPU (50 videos × 16×3×224×224 fp32 ≈ 480 MB —
    # well inside the eval-time budget; the old CPU dict was the path that
    # caused the 6-7-26 CPU-RAM OOM in the first place).  _denormalize moves
    # ONE video at a time to CPU when it is actually rendered.
    chosen_set = set(int(i) for i in chosen_indices)
    frames_by_idx = {}                                        # {int_idx: (T,C,H,W) GPU}
    _cursor = 0
    with torch.no_grad():
        for batch in test_loader:
            _frames_batch = batch["frames"]                   # (b, T, C, H, W)
            _b = _frames_batch.shape[0]
            _global = list(range(_cursor, _cursor + _b))
            _local_keep = [(i, g) for i, g in enumerate(_global) if g in chosen_set]
            if _local_keep:
                _frames_gpu = _frames_batch.to(device, non_blocking=True)
                for _i, _g in _local_keep:
                    frames_by_idx[_g] = _frames_gpu[_i].detach().clone()
                del _frames_gpu
            _cursor += _b
            if len(frames_by_idx) >= len(chosen_set):
                break
    print(f"[XAI overlays] collected {len(frames_by_idx)} frame tensors on GPU "
          f"(~{sum(f.numel()*4 for f in frames_by_idx.values())/1e6:.1f} MB)")

    # ── Load explainers ───────────────────────────────────────────────────────
    from xai.HiDF_gradcam import GradCAMExplainer
    from xai.HiDF_attention_rollout import AttentionRolloutExplainer

    gradcam_exp = GradCAMExplainer(
        model, target_layer=model.spatial_stream.grad_cam_target_layer
    )
    rollout_exp = AttentionRolloutExplainer(model)

    # ── Generate overlays ─────────────────────────────────────────────────────
    # Save 4 evenly-spaced frames × 3 methods per video
    T      = config.num_frames
    frame_indices = np.linspace(0, T - 1, min(4, T), dtype=int).tolist()

    for idx in tqdm(chosen_indices, desc="Saving overlays"):
        idx = int(idx)
        prob      = float(all_probs[idx])
        label     = int(all_labels[idx])
        label_str = "fake" if label == 1 else "real"
        meta      = all_meta[idx] if idx < len(all_meta) else {}
        video_path = meta.get("video_path", "") if isinstance(meta, dict) else ""
        video_id   = (
            os.path.splitext(os.path.basename(video_path))[0]
            if video_path else f"sample{idx}"
        )

        if idx not in frames_by_idx:
            print(f"  [XAI overlay] skip idx={idx} (no frames collected)")
            continue
        _frame_gpu = frames_by_idx[idx]                 # (T, C, H, W) GPU (Phase 30)
        frames_t   = _frame_gpu.unsqueeze(0)            # (1, T, C, H, W) GPU
        orig_rgb   = _denormalize(_frame_gpu)           # moves this ONE video to CPU

        # Intrinsic M_t
        intrinsic = all_M_t_up[idx].detach().cpu().numpy()   # (T, H, W)

        # Grad-CAM
        try:
            gradcam_maps = gradcam_exp.explain(frames_t)[0]   # (T, H, W) numpy
        except Exception as e:
            print(f"  [GradCAM failed idx={idx}: {e}]")
            gradcam_maps = intrinsic

        # Attention Rollout
        try:
            rollout_maps = rollout_exp.explain(frames_t)   # (T, H, W) numpy
        except Exception as e:
            print(f"  [Rollout failed idx={idx}: {e}]")
            rollout_maps = intrinsic

        # Save overlays for selected frames × methods
        for fi in frame_indices:
            fi = int(fi)
            rgb_frame = orig_rgb[fi]   # (H, W, 3) uint8 RGB

            for method_name, maps in [
                ("intrinsic", intrinsic),
                ("gradcam",   gradcam_maps),
                ("rollout",   rollout_maps),
            ]:
                heatmap = maps[fi]   # (H, W)

                # overlay_heatmap_on_frame returns (overlay_bgr, attn_norm) tuple
                bgr_frame        = rgb_frame[:, :, ::-1].copy()
                overlay_bgr, _   = overlay_heatmap_on_frame(bgr_frame, heatmap)
                overlay_bgr      = _ensure_uint8_bgr(overlay_bgr)   # safety guard

                fname    = f"{video_id}_{label_str}_conf{prob:.2f}_{method_name}_f{fi}.png"
                out_path = output_dir / fname

                import cv2 as _cv2
                ok = _cv2.imwrite(str(out_path), overlay_bgr)
                if not ok:
                    print(f"[WARN] imwrite failed for {out_path}; "
                          f"shape={overlay_bgr.shape}, dtype={overlay_bgr.dtype}")

        print(f"[XAI overlay] saved {video_id} ({label_str}, prob={prob:.2f})")

    # Final cleanup
    del frames_by_idx, all_M_t_up
    import gc as _gc_xai
    _gc_xai.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[XAI overlays] Done. Outputs in {output_dir}")
