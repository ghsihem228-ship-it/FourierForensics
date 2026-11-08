"""Fourier utilities; frequencies are measured in cycles per pixel."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def luminance(image: ArrayLike) -> NDArray[np.float64]:
    """Convert uint8/uint16 or finite [0,1] float grayscale/RGB to luminance."""
    a = np.asarray(image)
    if a.ndim not in (2, 3) or (a.ndim == 3 and a.shape[2] != 3):
        raise ValueError("Expected grayscale (H,W) or RGB (H,W,3)")
    if min(a.shape[:2]) < 2:
        raise ValueError("Image sides must have at least two pixels")
    if a.dtype in (np.dtype("uint8"), np.dtype("uint16")):
        a = a.astype(float) / np.iinfo(a.dtype).max
    elif np.issubdtype(a.dtype, np.floating):
        a = a.astype(float)
    else:
        raise ValueError("Use uint8, uint16, or floating point input")
    if not np.isfinite(a).all() or a.min() < 0 or a.max() > 1:
        raise ValueError("Image values must be finite and in [0,1]")
    return a if a.ndim == 2 else a @ np.array([0.2126, 0.7152, 0.0722])


def fft2_centered(
    image: ArrayLike, *, window: bool = False, remove_mean: bool = False
) -> NDArray[np.complex128]:
    """Return fftshift(fft2(I)); optional separable Hann reduces edge leakage."""
    a = luminance(image)
    if remove_mean:
        a = a - a.mean()
    if window:
        a = a * np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    return np.fft.fftshift(np.fft.fft2(a))


def log_magnitude(image: ArrayLike) -> NDArray[np.float64]:
    """Display spectrum log(1 + |F|)."""
    return np.log1p(np.abs(fft2_centered(image)))


def radial_average(spectrum: ArrayLike, bins: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Average a centered magnitude/power spectrum in frequency annuli.

    Returns bin centers and means; empty annuli are NaN. Power must be
    supplied explicitly as abs(F)**2 when a power profile is desired.
    """
    a = np.asarray(spectrum)
    if a.ndim != 2 or np.iscomplexobj(a) or not np.isfinite(a).all() or (a < 0).any():
        raise ValueError("Expected finite nonnegative 2D magnitude or power")
    if not isinstance(bins, int) or bins < 1:
        raise ValueError("bins must be a positive integer")
    fy = np.fft.fftshift(np.fft.fftfreq(a.shape[0]))
    fx = np.fft.fftshift(np.fft.fftfreq(a.shape[1]))
    r = np.hypot(fy[:, None], fx[None, :])
    edges = np.linspace(0, np.sqrt(0.5) + 1e-12, bins + 1)
    ids = np.minimum(np.digitize(r.ravel(), edges) - 1, bins - 1)
    counts = np.bincount(ids, minlength=bins)
    sums = np.bincount(ids, weights=a.ravel(), minlength=bins)
    mean = np.divide(sums, counts, out=np.full(bins, np.nan), where=counts > 0)
    return (edges[:-1] + edges[1:]) / 2, mean
