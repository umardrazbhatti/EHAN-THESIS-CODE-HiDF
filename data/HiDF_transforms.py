"""
data/transforms.py
==================
Provides get_transforms(mode, frame_size) used by DeepfakeDataset.__getitem__.

Train (6-7-26 cross-domain upgrade):
    Standard resize + horizontal flip + STRONGER colour jitter + light Gaussian
    blur + JPEG-compression simulation + small RandomErasing + ImageNet
    normalisation.

    Rationale (CelebDF generalization, fake_acc 0.03 → target ≥ 0.40):
      HiDF and CelebDF use different cameras, codecs, lighting, and
      synthesis pipelines.  Mild train-time augmentation (the previous 0.05
      ColorJitter pipeline) lets the model memorise HiDF-specific compression
      and colour fingerprints, so it sees CelebDF inputs as out-of-distribution
      and predicts "real" for nearly every fake.

      The expanded augmentation simulates cross-domain shifts WITHOUT erasing
      manipulation artifacts:
        - Stronger ColorJitter (0.15)            → camera / lighting variance
        - GaussianBlur p=0.15, sigma≤1.2         → mild compression artefact
        - RandomApply JPEG quality 40-80 p=0.30  → codec variance (CRITICAL —
                                                   CelebDF is c23 H.264, HiDF
                                                   uses a different codec)
        - RandomHorizontalFlip p=0.5             → standard symmetry
        - RandomErasing p=0.20 scale 0.02-0.06   → occlusion robustness;
                                                   patches too small to wipe
                                                   manipulation artifacts.
      All augmentations are CLASS-SYMMETRIC (applied with same probability to
      real and fake).  Per the existing project history, asymmetric heavy
      augmentation creates "blur = real" / "noise = fake" shortcuts that
      destroy the classifier; we deliberately avoid that.

Val / Test:
    Resize + ImageNet normalisation only — deterministic, matches inference.
"""

import io
import random

import torch
from PIL import Image
from torchvision import transforms


# ImageNet statistics — used for EfficientNet-B4 pre-trained weights
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────────────────────────────────────
# Custom transforms (PIL-level) for codec-variance simulation
# ─────────────────────────────────────────────────────────────────────────────

class RandomJPEGCompression:
    """
    Re-encode the PIL image through JPEG at a random quality in [q_lo, q_hi].
    Simulates the compression mismatch between HiDF (training) and CelebDF
    (deployment).  Class-symmetric and applied with probability p.
    """
    def __init__(self, p: float = 0.30, q_lo: int = 40, q_hi: int = 80):
        self.p    = float(p)
        self.q_lo = int(q_lo)
        self.q_hi = int(q_hi)

    def __call__(self, img):
        if random.random() > self.p:
            return img
        if not isinstance(img, Image.Image):
            return img
        if img.mode != "RGB":
            img = img.convert("RGB")
        q   = random.randint(self.q_lo, self.q_hi)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    def __repr__(self):
        return (f"{self.__class__.__name__}(p={self.p}, "
                f"q_lo={self.q_lo}, q_hi={self.q_hi})")


# ─────────────────────────────────────────────────────────────────────────────
# Public transform builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_train_pipeline(frame_size: int = 224):
    """
    Cross-domain-robust train pipeline (6-8-26 v2 — moderated).

    Background:
      The 6-7-26 v1 pipeline (ColorJitter=0.15, GaussianBlur p=0.15,
      RandomErasing p=0.20) moved CelebDF fake_acc from 0.03 → 0.37 — a real
      win — but also crushed the spatial explanation signal in M_t (insertion
      AUC 0.531 → 0.249, k1 ratio 7.38x → 1.00x). The architectural fix in
      models/HiDF_eahn.py (temporal_gate bottleneck) now handles the frame
      ranking structurally, so we don't need augmentation to be as aggressive.

    v2 changes vs v1:
      - ColorJitter 0.15 → 0.08      (still ~60% stronger than original 0.05)
      - GaussianBlur p=0.15 → 0.10   (lighter — was destroying spatial peaks)
      - RandomErasing p=0.20 → 0.10  (lighter — was eroding facial features)
      - RandomJPEGCompression KEPT at p=0.30, q=40–80 (the main cross-domain
        lever — codec mismatch HiDF vs CelebDF c23 H.264 — kept full strength
        because JPEG compression is what bridges the actual domain gap)
    """
    return transforms.Compose([
        transforms.Resize((frame_size, frame_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.08,
            contrast=0.08,
            saturation=0.08,
            hue=0.03,
        ),
        RandomJPEGCompression(p=0.30, q_lo=40, q_hi=80),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))],
            p=0.10,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
        transforms.RandomErasing(
            p=0.10, scale=(0.02, 0.05), ratio=(0.3, 3.3), value=0.0,
        ),
    ])


def get_heavy_transforms(frame_size: int = 224):
    """
    Minority-class augmentation — uses the same cross-domain pipeline as the
    standard training transform.  Kept identical (Fix 1) to prevent
    augmentation-artifact shortcut learning where the model treats a
    transformation as a class cue.
    """
    return _build_train_pipeline(frame_size)


def get_real_aug_transforms(frame_size: int = 224):
    """
    Real-video training augmentation: standard pipeline plus low-probability
    RandomGrayscale.

    Applied only to real (label=0) training samples to break per-video camera /
    compression identity shortcuts.  HiDF has ~3,500 real videos each with a
    distinctive camera noise + colour calibration fingerprint; the model can
    memorise these early (real_acc → 1.0 by epoch 2) and coast on that signal
    rather than learning fake-specific artifacts, causing fake_acc to stagnate.

    Why real-only and not all samples:
      Applying grayscale to fakes would erase chroma-channel manipulation cues
      that some methods leak.  Grayscale on reals is safe because reals have
      no manipulation cues to begin with.

    Why low probability (p=0.1):
      High-probability grayscale applied consistently to reals would itself
      become a class-conditional shortcut ("grayscale = real").  At p=0.1 the
      augmentation is unpredictable and cannot be used as a cue.
    """
    return transforms.Compose([
        transforms.Resize((frame_size, frame_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.08,
            contrast=0.08,
            saturation=0.08,
            hue=0.03,
        ),
        transforms.RandomGrayscale(p=0.1),
        RandomJPEGCompression(p=0.30, q_lo=40, q_hi=80),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))],
            p=0.10,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
        transforms.RandomErasing(
            p=0.10, scale=(0.02, 0.05), ratio=(0.3, 3.3), value=0.0,
        ),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Phase 27: DANN synthetic domain transforms
# ─────────────────────────────────────────────────────────────────────────────
# Four augmentation pipelines, each meant to look like a different "domain" so
# the DANN domain classifier has a real per-sample target to learn.  All four
# end in the same ToTensor + Normalize so feature statistics stay aligned.
#
#   D0  minimal       resize + hflip + normalize       (clean baseline)
#   D1  heavy JPEG    + RandomJPEGCompression p=1.0, q=30-50  (codec shift)
#   D2  noise         + Gaussian additive noise sigma 0.05-0.10 (sensor shift)
#   D3  blur          + GaussianBlur sigma 1.0-2.0     (resolution shift)
#
# Class-symmetric (same prob for real/fake), so the domain label cannot leak
# class information.

class _AddGaussianNoise:
    """Add per-pixel Gaussian noise to a normalised tensor (after ToTensor+Norm).

    sigma is drawn uniformly from [sigma_lo, sigma_hi] per call.  Always
    applied (no `p` short-circuit) so the noise level *is* the domain marker.
    """
    def __init__(self, sigma_lo: float = 0.05, sigma_hi: float = 0.10):
        self.sigma_lo = float(sigma_lo)
        self.sigma_hi = float(sigma_hi)

    def __call__(self, x):
        sigma = random.uniform(self.sigma_lo, self.sigma_hi)
        return x + torch.randn_like(x) * sigma

    def __repr__(self):
        return f"_AddGaussianNoise(sigma=[{self.sigma_lo},{self.sigma_hi}])"


def _domain_0_clean(frame_size: int = 224):
    return transforms.Compose([
        transforms.Resize((frame_size, frame_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])


def _domain_1_heavy_jpeg(frame_size: int = 224):
    return transforms.Compose([
        transforms.Resize((frame_size, frame_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        # Always JPEG-compress; quality 30-50 = visibly degraded codec
        RandomJPEGCompression(p=1.0, q_lo=30, q_hi=50),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])


def _domain_2_noise(frame_size: int = 224):
    return transforms.Compose([
        transforms.Resize((frame_size, frame_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
        _AddGaussianNoise(sigma_lo=0.05, sigma_hi=0.10),
    ])


def _domain_3_blur(frame_size: int = 224):
    return transforms.Compose([
        transforms.Resize((frame_size, frame_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        # Always blur with kernel 5 and sigma 1.0-2.0 (stronger than the
        # standard p=0.1 sigma 0.1-1.0 used in _build_train_pipeline).
        transforms.GaussianBlur(kernel_size=5, sigma=(1.0, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])


def get_domain_transform(domain_id: int, frame_size: int = 224):
    """Return the domain-id-conditioned train transform.

    Used in data/HiDF_datasets.py when Phase-27 DANN is enabled:
        domain_id = random.randint(0, num_domains-1)
        aug       = get_domain_transform(domain_id, frame_size)

    Falls back to the standard train pipeline for any unknown domain_id
    (defensive — should not happen in normal use).
    """
    if domain_id == 0:
        return _domain_0_clean(frame_size)
    if domain_id == 1:
        return _domain_1_heavy_jpeg(frame_size)
    if domain_id == 2:
        return _domain_2_noise(frame_size)
    if domain_id == 3:
        return _domain_3_blur(frame_size)
    return _build_train_pipeline(frame_size)


def get_transforms(mode: str, frame_size: int = 224):
    """
    Return a torchvision transform pipeline for the given split.

    Parameters
    ----------
    mode : str
        One of 'train', 'val', or 'test'.
    frame_size : int
        Target spatial resolution (height == width). Default 224.

    Returns
    -------
    torchvision.transforms.Compose
        A callable that accepts a PIL Image and returns a normalised float32 tensor
        of shape (3, frame_size, frame_size).
    """
    if mode == "train":
        t = _build_train_pipeline(frame_size)
        print(f"[get_transforms] train pipeline: {t}")
        return t
    # val and test: deterministic resize + normalise only
    return transforms.Compose([
        transforms.Resize((frame_size, frame_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])
