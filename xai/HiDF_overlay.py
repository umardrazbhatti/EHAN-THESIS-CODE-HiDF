"""
xai/overlay.py — Per-video heatmap overlay helpers used by scripts/evaluate.py.

Functions
---------
overlay_heatmap_on_frame  — resize heatmap, apply JET colormap, blend onto frame
draw_suspicion_box        — find largest high-attention contour, draw rect + label
write_video_summary       — save overlay MP4, middle-frame PNG, plain-English TXT
"""

import os
import cv2
import numpy as np


# 3×3 coarse face-region grid (row=top→bottom, col=left→right)
_REGION_GRID = {
    (0, 0): "forehead-left",   (0, 1): "forehead",        (0, 2): "forehead-right",
    (1, 0): "left eye",        (1, 1): "nose bridge",      (1, 2): "right eye",
    (2, 0): "left jaw/mouth",  (2, 1): "mouth/chin",       (2, 2): "right jaw/mouth",
}


def _region_from_centroid(heatmap: np.ndarray, threshold: float = 0.30) -> str:
    """Return coarse 3×3 grid label for the centroid of the thresholded heatmap."""
    h, w = heatmap.shape
    binary = (heatmap >= np.quantile(heatmap, 1.0 - threshold)).astype(np.float32)
    ys, xs = np.where(binary > 0)
    if len(ys) == 0:
        cy, cx = h // 2, w // 2
    else:
        cy, cx = int(ys.mean()), int(xs.mean())
    row = min(int(cy / h * 3), 2)
    col = min(int(cx / w * 3), 2)
    return _REGION_GRID.get((row, col), "face")


def overlay_heatmap_on_frame(
    frame_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Resize heatmap to frame size, apply COLORMAP_JET, blend onto frame_rgb.

    Parameters
    ----------
    frame_rgb : H×W×3 uint8 RGB image
    heatmap   : 2D float array (any spatial size)
    alpha     : blend weight for the heatmap overlay

    Returns
    -------
    blended : H×W×3 uint8 RGB image with heatmap overlay
    """
    H, W = frame_rgb.shape[:2]
    attn = cv2.resize(heatmap.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    a_min, a_max = attn.min(), attn.max()
    attn_norm = (attn - a_min) / (a_max - a_min + 1e-8)
    heatmap_u8  = (attn_norm * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(frame_rgb, 1.0 - alpha, heatmap_rgb, alpha, 0)
    return blended


def draw_suspicion_box(
    frame_rgb: np.ndarray,
    heatmap: np.ndarray,
    top_pct: float = 0.30,
) -> np.ndarray:
    """
    Threshold heatmap at its top top_pct quantile, find largest contour,
    draw rectangle + "Suspicion: high" label on frame_rgb (in-place copy).

    Returns a copy of frame_rgb with the rectangle drawn.
    """
    out = frame_rgb.copy()
    H, W = out.shape[:2]
    attn = cv2.resize(heatmap.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    thresh = np.quantile(attn, 1.0 - top_pct)
    binary = ((attn >= thresh) * 255).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 80, 0), 2)
        label_y = max(y - 6, 14)
        cv2.putText(out, "Suspicion: high", (x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 80, 0), 1)
    return out


def write_video_summary(
    out_dir: str,
    video_id: str,
    heatmap_stack: np.ndarray,
    pred_prob: float,
    frames_rgb: list = None,
    fps: int = 5,
) -> dict:
    """
    Save three artefacts for one test video:

      {video_id}_overlay.mp4  — frames with heatmap overlay (uses blank frames if
                                 frames_rgb is None)
      {video_id}_boxed.png    — middle frame with suspicion box drawn
      {video_id}_summary.txt  — plain-English description

    Parameters
    ----------
    out_dir        : directory to write into
    video_id       : string identifier
    heatmap_stack  : (T, H, W) float32 numpy array of attention maps
    pred_prob      : scalar sigmoid probability (fake=1)
    frames_rgb     : optional list of T  H×W×3 uint8 RGB frames; blank if None
    fps            : MP4 frame rate

    Returns
    -------
    dict with keys: mp4_path, png_path, txt_path
    """
    os.makedirs(out_dir, exist_ok=True)
    T, H, W = heatmap_stack.shape
    verdict    = "FAKE" if pred_prob > 0.5 else "REAL"
    confidence = abs(pred_prob - 0.5) * 2.0

    if frames_rgb is None:
        frames_rgb = [np.zeros((H, W, 3), dtype=np.uint8)] * T

    # ── MP4 overlay ───────────────────────────────────────────────────────────
    mp4_path = os.path.join(out_dir, f"{video_id}_overlay.mp4")
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(mp4_path, fourcc, fps, (W, H))
    for t in range(T):
        frame = cv2.resize(frames_rgb[t], (W, H))
        blended = overlay_heatmap_on_frame(frame, heatmap_stack[t])
        writer.write(cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
    writer.release()

    # ── Middle-frame PNG with suspicion box ───────────────────────────────────
    mid_t    = T // 2
    mid_rgb  = cv2.resize(frames_rgb[mid_t], (W, H))
    boxed    = draw_suspicion_box(
        overlay_heatmap_on_frame(mid_rgb, heatmap_stack[mid_t]),
        heatmap_stack[mid_t],
    )
    png_path = os.path.join(out_dir, f"{video_id}_boxed.png")
    cv2.imwrite(png_path, cv2.cvtColor(boxed, cv2.COLOR_RGB2BGR))

    # ── Plain-English summary TXT ─────────────────────────────────────────────
    frame_scores  = [float(heatmap_stack[t].max()) for t in range(T)]
    peak_t        = int(np.argmax(frame_scores))
    high_ts       = sorted(range(T), key=lambda t: frame_scores[t], reverse=True)[:3]
    frame_range   = f"frames {min(high_ts)+1}–{max(high_ts)+1}"
    mean_map      = heatmap_stack.mean(0)
    region        = _region_from_centroid(mean_map)

    summary = (
        f"Predicted {verdict} (confidence {confidence:.2f}, prob={pred_prob:.3f}).\n"
        f"Highest attention on {frame_range}, concentrated in the {region} region.\n"
        f"Peak attention frame: t={peak_t+1}/{T}  (score={frame_scores[peak_t]:.3f}).\n"
    )
    txt_path = os.path.join(out_dir, f"{video_id}_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary)

    return {"mp4_path": mp4_path, "png_path": png_path, "txt_path": txt_path}
