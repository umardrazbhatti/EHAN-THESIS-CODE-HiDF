"""
data/face_align.py — MTCNN face detector with tracking-based crop and disk cache.
Detects on frame 0; re-detects every 5 frames; falls back to centre-crop.

v2 patch — disk guard:
  np.save is now guarded by a free-space check. If free disk space on the
  cache volume falls below CACHE_MIN_FREE_GB (default 3.0 GB), the cache
  entry is silently skipped. This prevents the OSError crash from disk-full.

v3 patch — suppress warning spam:
  Instead of emitting a warnings.warn() per sample (which floods stdout with
  hundreds of lines mid-epoch), we print a single summary line every
  CACHE_SKIP_REPORT_EVERY skips. The skip counter resets each epoch via
  FaceAligner.reset_skip_counter().
"""

import cv2
import numpy as np
import os
import shutil
import warnings

try:
    from facenet_pytorch import MTCNN
    _MTCNN_AVAILABLE = True
except ImportError:
    _MTCNN_AVAILABLE = False

# Minimum free space (GB) required before writing a cache entry.
CACHE_MIN_FREE_GB = 3.0

# Print a summary warning every N skipped cache writes (not one per sample).
CACHE_SKIP_REPORT_EVERY = 50


def _has_disk_space(path: str, min_free_gb: float = CACHE_MIN_FREE_GB) -> bool:
    """Return True if the volume containing *path* has >= min_free_gb free."""
    try:
        usage = shutil.disk_usage(path)
        return (usage.free / 1e9) >= min_free_gb
    except Exception:
        return False


class FaceAligner:
    def __init__(self, margin: float = 0.30, cache_dir: str = None,
                 device: str = "cpu"):
        self.margin    = margin
        self.cache_dir = cache_dir
        self.device    = device
        self._skip_count = 0   # total cache writes skipped due to low disk
        if _MTCNN_AVAILABLE:
            self.mtcnn = MTCNN(
                keep_all=False,
                device="cpu",
                select_largest=True,
                post_process=False,
            )
        else:
            self.mtcnn = None

    def reset_skip_counter(self):
        """Call at the start of each epoch to reset the skip-warning counter."""
        if self._skip_count > 0:
            print(f"[FaceAligner] {self._skip_count} cache writes skipped last epoch "
                  f"(disk below {CACHE_MIN_FREE_GB} GB).")
        self._skip_count = 0

    def align_frames(self, frames: list, video_id: str,
                     output_size: int = 224) -> list:
        """
        Detect face on first frame, use box for tracking, re-detect every 5 frames.
        Returns list of (output_size×output_size×3) uint8 arrays.
        """
        if self.cache_dir:
            cache_key  = f"{video_id.replace('/', '_')}_T{len(frames)}"
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.npy")
            if os.path.exists(cache_path):
                return list(np.load(cache_path, allow_pickle=False))

        if self.mtcnn is None:
            return self._center_crop_all(frames, output_size)

        first = frames[0] if isinstance(frames[0], np.ndarray) else np.array(frames[0])
        try:
            boxes, _ = self.mtcnn.detect(first)
        except Exception as e:
            warnings.warn(f"[FaceAligner] MTCNN failed ({e}), using centre crop.")
            boxes = None
        if boxes is None or len(boxes) == 0:
            return self._center_crop_all(frames, output_size)

        box_ref = boxes[0].copy()
        aligned = []
        for i, img in enumerate(frames):
            if i > 0 and i % 5 == 0:
                try:
                    new_boxes, _ = self.mtcnn.detect(img)
                except Exception:
                    new_boxes = None
                if new_boxes is not None and len(new_boxes) > 0:
                    box_ref = new_boxes[0]
            x1, y1, x2, y2 = map(int, box_ref)
            w, h = x2 - x1, y2 - y1
            mx = int(w * self.margin)
            my = int(h * self.margin)
            y1 = max(0, y1 - my);  y2 = min(img.shape[0], y2 + my)
            x1 = max(0, x1 - mx);  x2 = min(img.shape[1], x2 + mx)
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                crop = img
            crop = cv2.resize(crop, (output_size, output_size))
            aligned.append(crop)

        # ── Disk-guarded cache write ───────────────────────────────────────────
        if self.cache_dir:
            check_dir = self.cache_dir if os.path.exists(self.cache_dir) else \
                        os.path.dirname(self.cache_dir) or "/"
            if _has_disk_space(check_dir):
                os.makedirs(self.cache_dir, exist_ok=True)
                try:
                    np.save(cache_path, np.array(aligned, dtype=np.uint8))
                except OSError:
                    self._skip_count += 1
            else:
                self._skip_count += 1
                # Print summary every N skips instead of once per sample
                if self._skip_count % CACHE_SKIP_REPORT_EVERY == 1:
                    print(f"[FaceAligner] cache skipped (disk < {CACHE_MIN_FREE_GB} GB) "
                          f"— {self._skip_count} times so far this epoch.")

        return aligned

    # ── helpers ───────────────────────────────────────────────────────────────

    def _center_crop_all(self, frames: list, size: int) -> list:
        out = []
        for f in frames:
            h, w = f.shape[:2]
            y1 = max(0, (h - size) // 2)
            x1 = max(0, (w - size) // 2)
            crop = f[y1:y1 + size, x1:x1 + size]
            if crop.shape[0] != size or crop.shape[1] != size:
                crop = cv2.resize(crop, (size, size))
            out.append(crop)
        return out

import cv2
import numpy as np
import os
import shutil
import warnings

try:
    from facenet_pytorch import MTCNN
    _MTCNN_AVAILABLE = True
except ImportError:
    _MTCNN_AVAILABLE = False

# Minimum free space (GB) required before writing a cache entry.
# Set to 3.0 so there is always headroom for checkpoints / plots.
CACHE_MIN_FREE_GB = 3.0


def _has_disk_space(path: str, min_free_gb: float = CACHE_MIN_FREE_GB) -> bool:
    """Return True if the volume containing *path* has >= min_free_gb free."""
    try:
        usage = shutil.disk_usage(path)
        free_gb = usage.free / 1e9
        return free_gb >= min_free_gb
    except Exception:
        return False  # if we can't check, be conservative and skip the write


class FaceAligner:
    def __init__(self, margin: float = 0.30, cache_dir: str = None,
                 device: str = "cpu"):
        self.margin = margin
        self.cache_dir = cache_dir
        self.device = device
        if _MTCNN_AVAILABLE:
            self.mtcnn = MTCNN(
                keep_all=False,
                device="cpu",          # always CPU — compatible with any GPU/CUDA version
                select_largest=True,
                post_process=False,
            )
        else:
            self.mtcnn = None

    def align_frames(self, frames: list, video_id: str,
                     output_size: int = 224) -> list:
        """
        Detect face on first frame, use box for tracking, re-detect every 5 frames.
        Returns list of (output_size×output_size×3) uint8 arrays.
        """
        if self.cache_dir:
            cache_key  = f"{video_id.replace('/', '_')}_T{len(frames)}"
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.npy")
            if os.path.exists(cache_path):
                return list(np.load(cache_path, allow_pickle=False))

        if self.mtcnn is None:
            return self._center_crop_all(frames, output_size)

        first = frames[0] if isinstance(frames[0], np.ndarray) else np.array(frames[0])
        try:
            boxes, _ = self.mtcnn.detect(first)
        except Exception as e:
            warnings.warn(f"[FaceAligner] MTCNN failed ({e}), using centre crop.")
            boxes = None
        if boxes is None or len(boxes) == 0:
            return self._center_crop_all(frames, output_size)

        box_ref = boxes[0].copy()
        aligned = []
        for i, img in enumerate(frames):
            if i > 0 and i % 5 == 0:
                try:
                    new_boxes, _ = self.mtcnn.detect(img)
                except Exception:
                    new_boxes = None
                if new_boxes is not None and len(new_boxes) > 0:
                    box_ref = new_boxes[0]
            x1, y1, x2, y2 = map(int, box_ref)
            w, h = x2 - x1, y2 - y1
            mx = int(w * self.margin)
            my = int(h * self.margin)
            y1 = max(0, y1 - my);  y2 = min(img.shape[0], y2 + my)
            x1 = max(0, x1 - mx);  x2 = min(img.shape[1], x2 + mx)
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                crop = img
            crop = cv2.resize(crop, (output_size, output_size))
            aligned.append(crop)

        # ── Disk-guarded cache write ───────────────────────────────────────────
        if self.cache_dir:
            check_dir = self.cache_dir if os.path.exists(self.cache_dir) else \
                        os.path.dirname(self.cache_dir) or "/"
            if _has_disk_space(check_dir):
                os.makedirs(self.cache_dir, exist_ok=True)
                try:
                    np.save(cache_path, np.array(aligned, dtype=np.uint8))
                except OSError as e:
                    warnings.warn(f"[FaceAligner] cache write failed ({e}), continuing.")
            else:
                warnings.warn(
                    f"[FaceAligner] disk space below {CACHE_MIN_FREE_GB} GB — "
                    f"skipping cache write for {video_id}."
                )

        return aligned

    # ── helpers ───────────────────────────────────────────────────────────────

    def _center_crop_all(self, frames: list, size: int) -> list:
        out = []
        for f in frames:
            h, w = f.shape[:2]
            y1 = max(0, (h - size) // 2)
            x1 = max(0, (w - size) // 2)
            crop = f[y1:y1 + size, x1:x1 + size]
            if crop.shape[0] != size or crop.shape[1] != size:
                crop = cv2.resize(crop, (size, size))
            out.append(crop)
        return out
