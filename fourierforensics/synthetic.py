"""Seeded procedural scenes and controlled processing operations."""

from io import BytesIO

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def scene(seed: int = 0, size: int = 256) -> np.ndarray:
    """Smooth random scene plus independent noise; deliberately simple controls."""
    rng = np.random.default_rng(seed)
    base = gaussian_filter(rng.random((size, size)), 8)
    base = 0.2 + 0.6 * (base - base.min()) / np.ptp(base)
    rgb = np.stack([base, 0.9 * base + 0.03, 0.8 * base + 0.08], axis=-1)
    return np.uint8(np.clip(rgb + rng.normal(0, 0.008, rgb.shape), 0, 1) * 255)


def jpeg_bytes(image: np.ndarray, quality: int = 90) -> bytes:
    """Encode with an aligned 8x8 grid and no chroma subsampling."""
    stream = BytesIO()
    Image.fromarray(image).save(stream, format="JPEG", quality=quality, subsampling=0)
    return stream.getvalue()


def decode(data: bytes) -> np.ndarray:
    with Image.open(BytesIO(data)) as im:
        return np.array(im.convert("RGB"))


def variants(seed: int = 0, size: int = 256) -> dict[str, tuple[np.ndarray, bytes | None]]:
    a = scene(seed, size)
    resized = cv2.resize(a, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    matrix = cv2.getRotationMatrix2D((size / 2, size / 2), 17, 1)
    rotated = cv2.warpAffine(a, matrix, (size, size), borderMode=cv2.BORDER_REFLECT)
    margin = size // 6
    rotated = rotated[margin:-margin, margin:-margin]
    spliced = a.copy()
    sl = np.s_[size // 4 : size // 2 + size // 4, size // 4 : size // 2 + size // 4]
    rng = np.random.default_rng(seed + 100000)
    spliced[sl] = np.uint8(np.clip(a[sl].astype(float) + rng.normal(0, 18, a[sl].shape), 0, 255))
    single = jpeg_bytes(a)
    double = jpeg_bytes(decode(jpeg_bytes(a, 65)), 90)
    return {
        "original": (a, None),
        "single_jpeg": (decode(single), single),
        "resize": (resized, None),
        "rotation": (rotated, None),
        "splice": (spliced, None),
        "double_jpeg": (decode(double), double),
    }
