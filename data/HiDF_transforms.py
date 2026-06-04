"""
data/transforms.py
==================
Provides get_transforms(mode, frame_size) used by DeepfakeDataset.__getitem__.

Train : random horizontal flip + colour jitter + Gaussian blur + ImageNet normalisation
Val / Test : centre-crop resize + ImageNet normalisation only
"""

from torchvision import transforms


# ImageNet statistics — used for EfficientNet-B4 pre-trained weights
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


def get_heavy_transforms(frame_size: int = 224):
    """
    Minority-class augmentation — kept identical to the standard training
    pipeline (Fix 1) to prevent augmentation-artifact shortcut learning.
    """
    return transforms.Compose([
        transforms.Resize((frame_size, frame_size)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ColorJitter(
            brightness=0.05,
            contrast=0.05,
            saturation=0.05,
            hue=0.02,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])


def get_real_aug_transforms(frame_size: int = 224):
    """
    Real-video training augmentation: standard pipeline plus low-probability
    RandomGrayscale and GaussianBlur.

    Applied only to real (label=0) training samples to break per-video camera /
    compression identity shortcuts. HiDF has ~3,500 real videos each with a
    distinctive camera noise + colour calibration fingerprint; the model can
    memorise these early (real_acc → 1.0 by epoch 2) and coast on that signal
    rather than learning fake-specific artifacts, causing fake_acc to stagnate.

    Why real-only and not all samples:
      Applying the same blur/grayscale to fakes would erase the facial boundary
      artifacts that are the primary fake signal, counteracting the goal.

    Why low probability (p=0.1):
      High-probability grayscale or blur applied consistently to reals would
      itself become a class-conditional shortcut ("blurry = real"). At p=0.1
      the augmentation is unpredictable and cannot be used as a cue.
    """
    return transforms.Compose([
        transforms.Resize((frame_size, frame_size)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ColorJitter(
            brightness=0.05,
            contrast=0.05,
            saturation=0.05,
            hue=0.02,
        ),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])


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
        t = transforms.Compose([
            transforms.Resize((frame_size, frame_size)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ColorJitter(
                brightness=0.05,
                contrast=0.05,
                saturation=0.05,
                hue=0.02,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=_MEAN, std=_STD),
        ])
        print(f"[get_transforms] train pipeline: {t}")
        return t
    else:
        # val and test: deterministic resize + normalise only
        return transforms.Compose([
            transforms.Resize((frame_size, frame_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_MEAN, std=_STD),
        ])
