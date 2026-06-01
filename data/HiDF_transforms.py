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
