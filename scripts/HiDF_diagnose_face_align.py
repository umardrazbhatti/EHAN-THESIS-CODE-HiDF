"""
scripts/diagnose_face_align.py — Phase 10 face-alignment diagnostic.

Probes 3 known sample videos + 20 random training videos (10 real / 10 fake,
seed=0). For each video:
  - Records original frame H, W
  - Runs standalone MTCNN for confidence + bbox diagnostic numbers
  - Runs the production FaceAligner (unchanged) to obtain the actual crop
  - Saves aligned_<id>.png and original_<id>.png to --output_dir

Side-effect-free: no cache writes (cache_dir=None), no model loads,
no checkpoint touches. All noisy stderr from decord / MTCNN is suppressed
so stdout stays clean for copy-paste.

Usage (on Kaggle):
    python scripts/diagnose_face_align.py \\
        --data_root /kaggle/input/.../ffpp_data \\
        --output_dir /kaggle/working/diag
"""

import argparse
import os
import random
import sys
import warnings

import cv2
import numpy as np


# ── stderr noise suppression ─────────────────────────────────────────────────

warnings.filterwarnings("ignore")


# ── video I/O helpers ─────────────────────────────────────────────────────────

def _read_frame0_bgr(video_path: str):
    """
    Read the first decodable frame of a video.
    Returns (frame_bgr, orig_H, orig_W) or (None, 0, 0) on failure.
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None, 0, 0
    H, W = frame.shape[:2]
    return frame, H, W


def _read_n_frames_rgb(video_path: str, n: int = 16) -> list:
    """
    Uniformly sample up to n frames from a video; return as RGB numpy arrays.
    Mirrors DeepfakeDataset._read_frames so the crop we pass to FaceAligner
    is the same as what training uses.
    """
    cap = cv2.VideoCapture(video_path)
    total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    target = set(np.linspace(0, total - 1, min(n, total), dtype=int).tolist())
    frames: list = []
    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fi in target:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        fi += 1
        if len(frames) >= n:
            break
    cap.release()
    if not frames:
        frames = [np.zeros((224, 224, 3), dtype=np.uint8)]
    return frames


# ── dataset discovery ─────────────────────────────────────────────────────────

_FF_METHODS = [
    "Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"
]


def _collect_random_videos(
    data_root: str,
    n_real: int = 10,
    n_fake: int = 10,
    seed: int = 0,
) -> list:
    """
    Return a balanced list of (video_path, label) tuples from FF++ c23.
    Fixed seed=0 for reproducibility.
    """
    from pathlib import Path
    root = Path(data_root)

    real_dir = root / "original_sequences" / "youtube" / "c23" / "videos"
    real_all = sorted(real_dir.glob("*.mp4")) if real_dir.exists() else []

    fake_all: list = []
    for method in _FF_METHODS:
        d = root / "manipulated_sequences" / method / "c23" / "videos"
        if d.exists():
            fake_all.extend(sorted(d.glob("*.mp4")))

    rng = random.Random(seed)
    real_sample = rng.sample(real_all, min(n_real, len(real_all)))
    fake_sample = rng.sample(fake_all, min(n_fake, len(fake_all)))

    result = [(str(p), 0) for p in real_sample] + [(str(p), 1) for p in fake_sample]
    rng.shuffle(result)
    return result


# ── per-video probe ───────────────────────────────────────────────────────────

_MARGIN = 0.30          # must match FaceAligner default
_CONF_THRESH = 0.90     # MTCNN confidence threshold


def _probe_video(
    video_path: str,
    output_dir: str,
    mtcnn,           # standalone MTCNN instance (or None)
    face_aligner,    # production FaceAligner instance
) -> dict:
    """Run the full diagnostic on one video. Returns a result dict."""
    video_id = os.path.splitext(os.path.basename(video_path))[0]

    # ── Read original frame 0 ─────────────────────────────────────────────
    frame0_bgr, orig_H, orig_W = _read_frame0_bgr(video_path)
    if frame0_bgr is None:
        return {
            "video_id":       video_id,
            "orig_HxW":       "READ_ERR",
            "mtcnn_ok":       False,
            "mtcnn_conf":     None,
            "bbox_xyxy":      None,
            "bbox_area_frac": 1.0,
            "crop_method":    "center_crop_fallback",
        }

    # Save original frame (BGR, PNG)
    cv2.imwrite(os.path.join(output_dir, f"original_{video_id}.png"), frame0_bgr)

    # ── Standalone MTCNN for diagnostic numbers ───────────────────────────
    frame0_rgb = cv2.cvtColor(frame0_bgr, cv2.COLOR_BGR2RGB)
    mtcnn_ok       = False
    mtcnn_conf     = None
    bbox_xyxy      = None
    bbox_area_frac = 1.0   # worst-case default
    crop_method    = "center_crop_fallback"

    if mtcnn is not None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                boxes, probs = mtcnn.detect(frame0_rgb)

            if (
                boxes is not None
                and len(boxes) > 0
                and probs is not None
                and probs[0] is not None
            ):
                conf = float(probs[0])
                if conf >= _CONF_THRESH:
                    mtcnn_ok   = True
                    mtcnn_conf = conf

                    # Apply the same margin FaceAligner uses
                    x1r, y1r, x2r, y2r = map(int, boxes[0])
                    bw = x2r - x1r
                    bh = y2r - y1r
                    mx = int(bw * _MARGIN)
                    my = int(bh * _MARGIN)
                    x1m = max(0, x1r - mx)
                    y1m = max(0, y1r - my)
                    x2m = min(orig_W, x2r + mx)
                    y2m = min(orig_H, y2r + my)

                    bbox_xyxy      = (x1m, y1m, x2m, y2m)
                    area           = (x2m - x1m) * (y2m - y1m)
                    bbox_area_frac = area / max(orig_W * orig_H, 1)
                    crop_method    = "mtcnn"
        except Exception:
            pass  # leave as center_crop_fallback

    # For center_crop fallback: report the actual 224×224 center-crop bbox
    # so bbox_area_frac reflects what FaceAligner truly extracts.
    if not mtcnn_ok:
        size = 224
        cx1 = max(0, (orig_W - size) // 2)
        cy1 = max(0, (orig_H - size) // 2)
        cx2 = min(orig_W, cx1 + size)
        cy2 = min(orig_H, cy1 + size)
        bbox_xyxy      = (cx1, cy1, cx2, cy2)
        area           = (cx2 - cx1) * (cy2 - cy1)
        bbox_area_frac = area / max(orig_W * orig_H, 1)

    # ── Production FaceAligner (actual crop training would use) ───────────
    frames_rgb     = _read_n_frames_rgb(video_path, n=16)
    aligned_frames = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # cache_dir=None → no cache writes; video_id used only for cache key
            aligned_frames = face_aligner.align_frames(frames_rgb, video_id)
    except Exception as e:
        print(f"[WARN] FaceAligner failed on {video_id}: {e}", file=sys.stderr)

    # Save aligned frame 0 (convert RGB→BGR for cv2.imwrite)
    if aligned_frames:
        aligned_bgr = cv2.cvtColor(
            np.asarray(aligned_frames[0], dtype=np.uint8), cv2.COLOR_RGB2BGR
        )
        cv2.imwrite(os.path.join(output_dir, f"aligned_{video_id}.png"), aligned_bgr)

    return {
        "video_id":       video_id,
        "orig_HxW":       f"{orig_H}x{orig_W}",
        "mtcnn_ok":       mtcnn_ok,
        "mtcnn_conf":     mtcnn_conf,
        "bbox_xyxy":      bbox_xyxy,
        "bbox_area_frac": bbox_area_frac,
        "crop_method":    crop_method,
    }


# ── table / summary output ────────────────────────────────────────────────────

def _fmt_conf(v) -> str:
    return f"{v:.3f}" if v is not None else "n/a"


def _fmt_bbox(b) -> str:
    if b is None:
        return "n/a"
    return f"({b[0]},{b[1]},{b[2]},{b[3]})"


def _print_table(results: list) -> None:
    hdr = (
        f"{'video_id':<26}| {'orig_HxW':<12}| {'mtcnn_ok':<9}| "
        f"{'mtcnn_conf':<11}| {'bbox_xyxy_post_margin':<26}| "
        f"{'bbox_area_frac':<15}| crop_method"
    )
    sep = (
        "-" * 26 + "|" + "-" * 13 + "|" + "-" * 10 + "|" +
        "-" * 12 + "|" + "-" * 27 + "|" + "-" * 16 + "|" + "-" * 20
    )
    print("===== FACE ALIGN DIAGNOSTIC =====")
    print(hdr)
    print(sep)
    for r in results:
        ok_str = "yes" if r["mtcnn_ok"] else "no"
        print(
            f"{r['video_id']:<26}| {r['orig_HxW']:<12}| {ok_str:<9}| "
            f"{_fmt_conf(r['mtcnn_conf']):<11}| {_fmt_bbox(r['bbox_xyxy']):<26}| "
            f"{r['bbox_area_frac']:<15.3f}| {r['crop_method']}"
        )


def _print_summary(results: list) -> None:
    total      = len(results)
    n_ok       = sum(1 for r in results if r["mtcnn_ok"])
    n_fail     = total - n_ok
    n_fallback = sum(1 for r in results if r["crop_method"] == "center_crop_fallback")

    mtcnn_fracs = [r["bbox_area_frac"] for r in results if r["mtcnn_ok"]]
    all_fracs   = [r["bbox_area_frac"] for r in results]

    median_frac = float(np.median(mtcnn_fracs)) if mtcnn_fracs else float("nan")
    max_frac    = max(all_fracs) if all_fracs else float("nan")
    n_large     = sum(1 for f in all_fracs if f > 0.80)

    print("")
    print("===== SUMMARY =====")
    print(f"total_probed        : {total}")
    print(f"mtcnn_success       : {n_ok}")
    print(f"mtcnn_failed        : {n_fail}")
    print(f"center_crop_fallback: {n_fallback}")
    print(
        f"median bbox_area_frac (mtcnn samples) : "
        f"{median_frac:.2f}"
    )
    print(f"max    bbox_area_frac                  : {max_frac:.2f}")
    print(f"n videos with bbox_area_frac > 0.80    : {n_large}")
    print("===== END =====")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 10 face-alignment diagnostic for the EAHN deepfake project. "
            "Probes 3 known videos + 20 random FF++ videos; reports MTCNN "
            "success rates and bbox area fractions; saves aligned/original PNGs."
        )
    )
    parser.add_argument(
        "--data_root", required=True,
        help="Path to the ffpp_data root directory "
             "(contains original_sequences/ and manipulated_sequences/).",
    )
    parser.add_argument(
        "--output_dir", default="/kaggle/working/diag",
        help="Directory to write aligned_*.png and original_*.png files "
             "(default: /kaggle/working/diag).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Add repo root to sys.path so local imports work regardless of cwd
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    # ── Load FaceAligner (production class — unchanged) ───────────────────
    from data.HiDF_face_align import FaceAligner
    # cache_dir=None → no cache writes; no side effects on training pipeline
    face_aligner = FaceAligner(margin=_MARGIN, cache_dir=None, device="cpu")

    # ── Load standalone MTCNN for diagnostic numbers ──────────────────────
    mtcnn = None
    try:
        from facenet_pytorch import MTCNN as _MTCNN
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mtcnn = _MTCNN(
                keep_all=False,
                device="cpu",
                select_largest=True,
                post_process=False,
            )
    except Exception as e:
        print(
            f"[WARN] Could not instantiate standalone MTCNN: {e}\n"
            "       MTCNN columns will show n/a; crop_method will be "
            "inferred from FaceAligner output.",
            file=sys.stderr,
        )

    # ── Three specific known sample videos ────────────────────────────────
    specific_paths = [
        os.path.join(
            args.data_root,
            "manipulated_sequences/Face2Face/c23/videos/314_347.mp4",
        ),
        os.path.join(
            args.data_root,
            "manipulated_sequences/Face2Face/c23/videos/834_852.mp4",
        ),
        os.path.join(
            args.data_root,
            "original_sequences/youtube/c23/videos/454.mp4",
        ),
    ]

    # ── 20 random videos (10 real / 10 fake, seed=0) ──────────────────────
    random_pairs = _collect_random_videos(
        args.data_root, n_real=10, n_fake=10, seed=0
    )

    # Combine: specific three first, then random sample
    all_paths = [(p, None) for p in specific_paths] + random_pairs

    # ── Probe each video ──────────────────────────────────────────────────
    results = []
    for video_path, _label in all_paths:
        if not os.path.exists(video_path):
            vid = os.path.splitext(os.path.basename(video_path))[0]
            print(
                f"[SKIP] File not found: {video_path}",
                file=sys.stderr,
            )
            results.append({
                "video_id":       vid,
                "orig_HxW":       "NOT_FOUND",
                "mtcnn_ok":       False,
                "mtcnn_conf":     None,
                "bbox_xyxy":      None,
                "bbox_area_frac": 1.0,
                "crop_method":    "center_crop_fallback",
            })
            continue

        r = _probe_video(video_path, args.output_dir, mtcnn, face_aligner)
        results.append(r)

    # ── Print structured table + summary ──────────────────────────────────
    _print_table(results)
    _print_summary(results)


if __name__ == "__main__":
    main()
