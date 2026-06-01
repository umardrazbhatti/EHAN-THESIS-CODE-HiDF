"""
utils/visualization.py — Explanation visualization utilities.

Functions
---------
overlay_heatmap_on_frame   — blend attention heatmap onto a BGR frame
get_region_label           — human-readable centroid label for a saliency map
generate_explanation_text  — multi-line plain-English explanation string
save_annotated_frame_strip — PNG strip of annotated frames + text panel
save_explanation_video     — MP4 with per-frame overlay and info panel
"""

import os
import cv2
import numpy as np


# ── Browser-compatible codec helper ──────────────────────────────────────────

_FOURCC = None   # cached after first call


def _get_fourcc() -> int:
    """
    Return the best available VideoWriter fourcc for browser-playable MP4.

    Tries H.264 (avc1) first — required for IPython.display.Video inline
    playback in Kaggle notebooks.  Falls back to mp4v if avc1 is unavailable
    (e.g. OpenCV built without x264).

    The result is cached in _FOURCC so the codec test runs only once per
    process.
    """
    global _FOURCC
    if _FOURCC is not None:
        return _FOURCC
    test_path = "/tmp/_codec_test.mp4"
    try:
        w = cv2.VideoWriter(
            test_path,
            cv2.VideoWriter_fourcc(*"avc1"),
            5, (16, 16),
        )
        if w.isOpened():
            w.release()
            _FOURCC = cv2.VideoWriter_fourcc(*"avc1")
            return _FOURCC
    except Exception:
        pass
    _FOURCC = cv2.VideoWriter_fourcc(*"mp4v")
    return _FOURCC


# ── Top-K bounding box helper (CHANGE 10a) ───────────────────────────────────

def _topk_bbox(heatmap: np.ndarray, percentile: int = 95):
    """
    Return (y0, x0, y1, x1) bounding box of pixels at or above the given
    percentile of the heatmap.  Returns None if no pixels qualify.

    Parameters
    ----------
    heatmap    : 2D numpy float array (any spatial size)
    percentile : intensity percentile threshold (default 95 → top 5%)

    Returns
    -------
    (y0, x0, y1, x1) int tuple, or None
    """
    thr = np.percentile(heatmap, percentile)
    ys, xs = np.where(heatmap >= thr)
    if len(ys) == 0:
        return None
    return (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max()))


# ── overlay_heatmap_on_frame ──────────────────────────────────────────────────

def overlay_heatmap_on_frame(
    frame_bgr: np.ndarray,
    attention_map: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
):
    """
    Blend an attention heatmap onto a BGR frame.

    Parameters
    ----------
    frame_bgr     : H×W×3 uint8 BGR image
    attention_map : 2D float array (any spatial size)
    alpha         : blend weight for the heatmap
    colormap      : OpenCV colormap constant

    Returns
    -------
    overlay_bgr        : H×W×3 uint8 — blended image with bounding rect
    normalized_attn    : H×W float32 in [0, 1] — resized+normalised map
    """
    H, W = frame_bgr.shape[:2]

    # Resize to frame dimensions
    attn_resized = cv2.resize(
        attention_map.astype(np.float32), (W, H),
        interpolation=cv2.INTER_LINEAR,
    )

    # Min-max normalise to [0, 1]
    a_min, a_max = attn_resized.min(), attn_resized.max()
    attn_norm = (attn_resized - a_min) / (a_max - a_min + 1e-8)

    # Apply colormap and blend
    heatmap_u8  = (attn_norm * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(heatmap_u8, colormap)
    overlay     = cv2.addWeighted(frame_bgr, 1 - alpha, heatmap_bgr, alpha, 0)

    # Find largest contour in threshold=0.6 binary map; draw green bounding rect
    binary    = (attn_norm >= 0.6).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text_y = max(y - 5, 12)
        cv2.putText(
            overlay, "High Attention", (x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1,
        )

    return overlay, attn_norm



# ── get_region_label ──────────────────────────────────────────────────────────
#
# Face region map (proportional, maps to 224×224 reference):
#   Row bands  (as fraction of map height):
#     0.000–0.268  → "forehead / upper face"     (rows   0– 60 of 224)
#     0.268–0.491  → "eye / brow region"          (rows  60–110 of 224)
#     0.491–0.670  → "nose / mid-face"            (rows 110–150 of 224)
#     0.670–0.804  → "mouth / philtrum"           (rows 150–180 of 224)
#     0.804–1.000  → "chin / jaw"                 (rows 180–224 of 224)
#
#   Col bands  (as fraction of map width):
#     0.000–0.268  → "left side"                  (cols   0– 60 of 224)
#     0.268–0.714  → "central"                    (cols  60–160 of 224)
#     0.714–1.000  → "right side"                 (cols 160–224 of 224)

_ROW_BANDS = [
    (0.000, 0.268, "forehead / upper face"),
    (0.268, 0.491, "eye / brow region"),
    (0.491, 0.670, "nose / mid-face"),
    (0.670, 0.804, "mouth / philtrum"),
    (0.804, 1.001, "chin / jaw"),
]

_COL_BANDS = [
    (0.000, 0.268, "left"),
    (0.268, 0.714, "central"),
    (0.714, 1.001, "right"),
]


def get_region_label(attn_map: np.ndarray) -> str:
    """
    Return a human-readable face-region label for the peak of the attention map.

    Uses a 5-row × 3-column proportional grid calibrated to a 224×224 face crop:
      rows: forehead/upper-face · eye/brow · nose/mid-face · mouth/philtrum · chin/jaw
      cols: left · central · right

    Appends "(peak at row=R, col=C)" for verifiability.

    Parameters
    ----------
    attn_map : 2D numpy float array (any spatial size)

    Returns
    -------
    str  e.g. "central nose / mid-face region (peak at row=80, col=140)"
    """
    peak_idx = int(np.argmax(attn_map))
    r, c = np.unravel_index(peak_idx, attn_map.shape)
    H, W = attn_map.shape

    row_frac = r / max(H - 1, 1)
    col_frac = c / max(W - 1, 1)

    row_label = "nose / mid-face"   # sensible fallback
    for lo, hi, label in _ROW_BANDS:
        if lo <= row_frac < hi:
            row_label = label
            break

    col_label = "central"
    for lo, hi, label in _COL_BANDS:
        if lo <= col_frac < hi:
            col_label = label
            break

    return f"{col_label} {row_label} region (peak at row={r}, col={c})"


# ── Manipulation-aware artifact bullets ───────────────────────────────────────

_MANIPULATION_ARTIFACTS = {
    "Deepfakes":       [
        "Face-swap blending boundaries, identity blend seams",
        "Identity inconsistency between source and target",
        "GAN frequency fingerprints in shallow texture layers",
    ],
    "Face2Face":       [
        "Expression-source mismatch, unnatural jaw / mouth motion",
        "Reenactment boundary misalignment around mouth corners",
        "Temporal expression discontinuity between frames",
    ],
    "FaceSwap":        [
        "Landmark-driven warp seams along jaw and hairline",
        "Geometric distortion at face boundary",
        "Texture discontinuity where warped region meets background",
    ],
    "FaceShifter":     [
        "Identity-preserving blending residue, attribute swap artifacts",
        "Subtle colour mismatch between swapped and original skin tones",
        "High-frequency edge ghosting at occlusion boundaries",
    ],
    "NeuralTextures":  [
        "Mouth-region neural rendering residue, lip-sync mismatch",
        "Texture hallucination artefacts in the lip / teeth area",
        "Temporal flickering of synthesised mouth pixels",
    ],
}

_DEFAULT_ARTIFACTS = [
    "Blending boundary artifacts at face-swap seams",
    "Unnatural skin texture or colour inconsistencies",
    "Identity inconsistencies introduced by face manipulation",
]


# ── generate_explanation_text ─────────────────────────────────────────────────

def generate_explanation_text(
    verdict: str,
    confidence: float,
    prob: float,
    attention_scores: list,
    attention_maps: list,
    batch_inter_sample_sim: float = 0.0,
    active_manipulation: str = "",
) -> str:
    """
    Build a multi-line plain-English explanation string.

    Parameters
    ----------
    verdict                 : "FAKE" or "REAL"
    confidence              : float 0–1  (abs(prob - 0.5) * 2)
    prob                    : float  raw sigmoid output
    attention_scores        : list of T floats — per-frame scalar attention values
    attention_maps          : list of T 2-D numpy arrays
    batch_inter_sample_sim  : mean cosine sim across the evaluation batch (for collapse check)
    active_manipulation     : EAHNConfig.active_manipulation string (e.g. "Face2Face");
                              controls the manipulation-specific artifact bullets.

    Returns
    -------
    str
    """
    T = len(attention_scores)
    M_t_up = np.stack(attention_maps) if T > 0 else np.zeros((1, 7, 7))

    # --- Collapse diagnostics (computed before text generation) ---

    # 1. Spatially uniform within frames? → mean per-frame std < 0.01
    spatial_std_per_frame = [float(m.std()) for m in M_t_up]
    is_spatially_uniform  = float(np.mean(spatial_std_per_frame)) < 0.01

    # 2. Temporally frozen? → cosine sim between t=0 and t=T-1 > 0.99
    if T > 1:
        flat0     = M_t_up[0].flatten() / (np.linalg.norm(M_t_up[0]) + 1e-8)
        flat_last = M_t_up[-1].flatten() / (np.linalg.norm(M_t_up[-1]) + 1e-8)
        is_temporally_frozen = float(np.dot(flat0, flat_last)) > 0.99
    else:
        is_temporally_frozen = False

    # 3. Class-agnostic? → batch-mean inter-sample cosine sim > 0.95
    is_class_agnostic = batch_inter_sample_sim > 0.95

    sorted_frames = sorted(range(T), key=lambda i: attention_scores[i], reverse=True)
    top3          = sorted_frames[:3]

    lines = [
        f"VERDICT: This video is likely {verdict} (confidence: {confidence:.0%}).",
        "",
        "EXPLANATION:",
    ]

    # Choose message based on collapse diagnostics
    if is_spatially_uniform and is_temporally_frozen and is_class_agnostic:
        lines.append(
            "  ⚠ EXPLANATION COLLAPSE DETECTED — heatmap is identical across frames "
            "AND across samples. Re-train with stronger inter-sample diversity loss."
        )
    elif is_spatially_uniform:
        lines.append(
            "  • Attention is spatially uniform within frames (no localised focus). "
            "The explanation head may need stronger diversity regularisation."
        )
    elif is_temporally_frozen:
        lines.append(
            "  • Attention map is nearly identical across all frames (temporally frozen). "
            "Consider reducing lambda2 (temporal consistency weight)."
        )
    elif is_class_agnostic:
        lines.append(
            "  • Attention maps are very similar across samples in this batch "
            "(class-agnostic). The diversity loss may need increasing."
        )
    else:
        top3_labels = ", ".join(str(f + 1) for f in top3)
        lines.append(f"  • Attention was highest in frames {top3_labels}.")

    # Region label from peak of mean attention map across all frames
    mean_attn = np.mean(M_t_up, axis=0)
    region    = get_region_label(mean_attn)
    lines.append(f"  • The primary area of concern is the {region}.")

    if verdict == "FAKE":
        # Manipulation-aware artifact bullets (Task 1.2)
        artifacts = _MANIPULATION_ARTIFACTS.get(active_manipulation, _DEFAULT_ARTIFACTS)
        lines.append("  • High attention in this area may indicate:")
        for art in artifacts:
            lines.append(f"      - {art}")
    else:
        lines.append("  • No strong manipulation artifacts were detected.")
        lines.append(
            "    Facial regions show consistent texture and identity across frames."
        )

    lines.append("")
    lines.append("ATTENTION SCORES PER FRAME:")
    for i, score in enumerate(attention_scores):
        filled = int(score * 20)
        bar    = "█" * filled + "░" * (20 - filled)
        # Task 1.3: 4 decimal places for verifiability (was .3f → .4f)
        lines.append(f"  Frame {i + 1:02d}: [{bar}]  {score:.4f}")

    return "\n".join(lines)


# ── save_annotated_frame_strip ────────────────────────────────────────────────

def save_annotated_frame_strip(
    frames_bgr: list,
    attention_maps: list,
    attention_scores: list,
    verdict: str,
    prob: float,
    output_path: str,
    sample_id: str,
    batch_inter_sample_sim: float = 0.0,
    active_manipulation: str = "",
) -> str:
    """
    Save a horizontal strip of up to 8 annotated frames plus a text panel.

    Parameters
    ----------
    frames_bgr          : list of T  H×W×3 uint8 BGR arrays
    attention_maps      : list of T  2-D float arrays
    attention_scores    : list of T  floats
    verdict             : "FAKE" or "REAL"
    prob                : raw sigmoid probability
    output_path         : destination .png path
    sample_id           : string identifier used in labels
    active_manipulation : EAHNConfig.active_manipulation (e.g. "Face2Face")

    Returns
    -------
    output_path : str
    """
    from PIL import Image as PILImage, ImageDraw, ImageFont

    T        = len(frames_bgr)
    n_select = min(T, 8)
    sel_idx  = np.linspace(0, T - 1, n_select, dtype=int)

    # ── Row 1: heatmap overlaid on frame + top-5% red bbox (CHANGE 10a) ───────
    annotated_frames   = []
    # ── Row 2: pure heatmap (no underlying frame)      (CHANGE 10b) ───────────
    raw_heatmap_frames = []

    for idx in sel_idx:
        frame = cv2.resize(frames_bgr[idx], (224, 224))
        overlay, attn_norm = overlay_heatmap_on_frame(frame, attention_maps[idx])

        # Frame label
        label = f"F{idx + 1:02d}  attn:{attention_scores[idx]:.2f}"
        cv2.putText(overlay, label, (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

        # Top-5% bounding box in red (thin, on top of green contour from overlay)
        bbox = _topk_bbox(attn_norm, percentile=95)
        if bbox is not None:
            y0, x0, y1, x1 = bbox
            cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 255), 1)

        annotated_frames.append(overlay)

        # Pure heatmap tile
        attn_r  = cv2.resize(attention_maps[idx].astype(np.float32), (224, 224))
        a_min, a_max = attn_r.min(), attn_r.max()
        attn_u8 = ((attn_r - a_min) / (a_max - a_min + 1e-8) * 255).astype(np.uint8)
        raw_hm  = cv2.applyColorMap(attn_u8, cv2.COLORMAP_JET)
        raw_heatmap_frames.append(raw_hm)

    strip_row1 = np.hstack(annotated_frames)       # (224, n_select*224, 3)
    strip_row2 = np.hstack(raw_heatmap_frames)     # (224, n_select*224, 3)
    strip_w    = strip_row1.shape[1]

    # ── Concentration stats for text panel (CHANGE 10c) ───────────────────────
    if attention_maps:
        all_peaks = [
            np.unravel_index(np.argmax(m), m.shape) for m in attention_maps
        ]
        if all_peaks:
            mode_loc       = max(set(all_peaks), key=all_peaks.count)
            peak_mode_share = all_peaks.count(mode_loc) / len(all_peaks)
        else:
            peak_mode_share = 0.0
        mean_std = float(np.mean([m.std() for m in attention_maps]))
    else:
        peak_mode_share = 0.0
        mean_std        = 0.0

    # Build explanation text and render onto a dark PIL panel
    confidence = prob if prob >= 0.5 else (1.0 - prob)
    text       = generate_explanation_text(
        verdict, confidence, prob, attention_scores, attention_maps,
        batch_inter_sample_sim=batch_inter_sample_sim,
        active_manipulation=active_manipulation,
    )
    # Append concentration stats line
    text += (
        f"\nAttention concentration: "
        f"peak_mode_share={peak_mode_share:.2f}, mt_std={mean_std:.4f}"
    )
    text_lines = text.split("\n")
    line_h     = 17
    top_margin = 10
    left_margin = 10
    panel_h    = len(text_lines) * line_h + 20

    panel_pil = PILImage.new("RGB", (strip_w, panel_h), (20, 20, 20))
    draw      = ImageDraw.Draw(panel_pil)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 13
        )
    except Exception:
        font = ImageFont.load_default()

    verdict_color = (255, 80, 80) if verdict == "FAKE" else (80, 255, 80)
    other_color   = (220, 220, 220)

    for i, line in enumerate(text_lines):
        y     = top_margin + i * line_h
        color = verdict_color if i == 0 else other_color
        draw.text((left_margin, y), line, fill=color, font=font)

    # Convert panel to BGR numpy and stack: row1 (overlay) + row2 (raw heatmap) + text panel
    panel_bgr   = cv2.cvtColor(np.array(panel_pil), cv2.COLOR_RGB2BGR)
    final_image = np.vstack([strip_row1, strip_row2, panel_bgr])

    # Save image
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, final_image)

    # Save companion text file
    txt_path = output_path.replace(".png", "_explanation.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)

    return output_path


# ── save_explanation_video ────────────────────────────────────────────────────

def save_explanation_video(
    frames_bgr: list,
    attention_maps: list,
    attention_scores: list,
    verdict: str,
    prob: float,
    output_path: str,
    fps: int = 5,
) -> None:
    """
    Save an annotated explanation video (224×304 px per frame: 224 frame + 80 panel).

    Parameters
    ----------
    frames_bgr       : list of T  H×W×3 uint8 BGR arrays
    attention_maps   : list of T  2-D float arrays
    attention_scores : list of T  floats
    verdict          : "FAKE" or "REAL"
    prob             : raw sigmoid probability
    output_path      : destination .mp4 path
    fps              : frames per second
    """
    T          = len(frames_bgr)
    confidence = prob if prob >= 0.5 else (1.0 - prob)
    verdict_color_bgr = (80, 80, 255) if verdict == "FAKE" else (80, 255, 80)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (224, 224 + 80))
    if not writer.isOpened():
        print(f"[WARN] VideoWriter failed to open for {output_path}; skipping")
        return

    for t in range(T):
        frame   = cv2.resize(frames_bgr[t], (224, 224))
        overlay, _ = overlay_heatmap_on_frame(frame, attention_maps[t])

        # Info panel: 80px tall, 224px wide, dark background (20, 20, 20)
        panel = np.full((80, 224, 3), 20, dtype=np.uint8)

        # Line 1 — verdict + confidence
        cv2.putText(
            panel, f"{verdict} ({confidence:.0%} conf)", (6, 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, verdict_color_bgr, 1,
        )
        # Line 2 — frame index + region
        region = get_region_label(attention_maps[t])
        cv2.putText(
            panel, f"Frame {t + 1:02d}/{T} | Region: {region}", (6, 34),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1,
        )
        # Line 3 — attention score
        cv2.putText(
            panel, f"Attn: {attention_scores[t]:.3f}", (6, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1,
        )
        # Attention bar: x=8 to x=8+(224-80) full width outline; filled portion
        bar_max_w = 224 - 80
        bar_w     = int(attention_scores[t] * bar_max_w)
        cv2.rectangle(panel, (8, 58), (8 + bar_max_w, 70), (100, 100, 100), 1)
        if bar_w > 0:
            cv2.rectangle(panel, (8, 58), (8 + bar_w, 70), (100, 200, 255), -1)

        combined = np.vstack([overlay, panel])   # (304, 224, 3)
        writer.write(combined)

    writer.release()
