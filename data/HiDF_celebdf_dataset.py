"""
data/celebdf_dataset.py — Celeb-DF v2 test-only dataset loader.

Parses <root>/List_of_testing_videos.txt with line format:
    "<0|1> <relative/path/to/video.mp4>"
NOTE: In Celeb-DF v2, 1 = REAL, 0 = FAKE (opposite of our convention).
This loader flips the label on read so downstream code sees our convention
(0 = real, 1 = fake).

Expected test split (per Celeb-DF v2 official release):
    178 real + 340 fake = 518 videos.
Log the actual counts at construction time.
"""

import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

try:
    from decord import VideoReader, cpu as decord_cpu
    _DECORD_AVAILABLE = True
except ImportError:
    _DECORD_AVAILABLE = False
    warnings.warn(
        "decord not available; falling back to OpenCV for video reading in CelebDFv2TestDataset."
    )


class CelebDFv2TestDataset(Dataset):
    """
    Celeb-DF v2 test set.

    Args:
        root        : path to the Celeb-DF v2 dataset root (contains List_of_testing_videos.txt)
        num_frames  : number of frames to sample per video
        frame_size  : resize each frame to (frame_size, frame_size)
        face_aligner: FaceAligner instance (reused from DeepfakeDataset, not rebuilt)
        transform   : torchvision transform to apply to each PIL frame
        cache_dir   : optional face-alignment cache directory
    """

    def __init__(
        self,
        root: str,
        num_frames: int,
        frame_size: int,
        face_aligner,
        transform,
        cache_dir: str | None = None,
    ):
        self.root         = Path(root)
        self.num_frames   = num_frames
        self.frame_size   = frame_size
        self.face_aligner = face_aligner
        self.transform    = transform
        self.cache_dir    = cache_dir

        test_list_path = self.root / "List_of_testing_videos.txt"
        if not test_list_path.exists():
            raise FileNotFoundError(
                f"Celeb-DF v2 test list not found at {test_list_path}. "
                f"This file is required. Check that the dataset root is correct."
            )

        self.samples = []
        with open(test_list_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) < 2:
                    continue
                celebdf_label = int(parts[0])          # 1=real, 0=fake in their convention
                rel_path      = parts[1]
                our_label     = 0 if celebdf_label == 1 else 1   # FLIP: their 1=real → our 0=real
                full_path     = self.root / rel_path
                self.samples.append({
                    "video_path": str(full_path),
                    "label":      our_label,
                })

        n_real = sum(1 for s in self.samples if s["label"] == 0)
        n_fake = sum(1 for s in self.samples if s["label"] == 1)
        print(
            f"[Celeb-DF v2] test split loaded: "
            f"real={n_real} fake={n_fake} total={len(self.samples)}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample     = self.samples[idx]
        video_path = sample["video_path"]
        label      = sample["label"]

        frames_np = self._read_frames(video_path)

        # Face alignment (reuses shared FaceAligner with its cache)
        import os
        video_id  = os.path.splitext(os.path.basename(video_path))[0]
        frames_np = self.face_aligner.align_frames(frames_np, video_id)

        frames_tensor = torch.stack(
            [self.transform(Image.fromarray(f)) for f in frames_np]
        )  # (T, 3, H, W)

        return {
            "frames": frames_tensor,
            "label":  label,
            "meta":   {"video_path": video_path, "frame_indices": []},
        }

    def _read_frames(self, video_path: str) -> list:
        """
        Uniformly samples self.num_frames frames from a video file.
        Tries decord first; falls back to OpenCV.
        Returns list of (H, W, 3) uint8 RGB arrays.
        """
        T = self.num_frames

        if _DECORD_AVAILABLE:
            try:
                vr      = VideoReader(video_path, ctx=decord_cpu(0))
                total   = len(vr)
                indices = np.linspace(0, total - 1, T, dtype=int).tolist()
                batch   = vr.get_batch(indices).asnumpy()
                return [batch[i] for i in range(T)]
            except Exception as exc:
                warnings.warn(
                    f"decord failed on '{video_path}': {exc}. Using OpenCV fallback."
                )

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
