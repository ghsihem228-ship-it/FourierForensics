"""Periodic peaks in locally normalized high-pass residual energy."""

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter, median_filter

from ..common import DetectorResult, highpass


def detect(image: np.ndarray) -> DetectorResult:
    r = highpass(image)[4:-4, 4:-4]
    energy = r * r / (gaussian_filter(r * r, 3) + 1e-10)
    window = np.outer(np.hanning(r.shape[0]), np.hanning(r.shape[1]))
    power = abs(np.fft.fftshift(np.fft.fft2(window * (energy - energy.mean())))) ** 2
    log = np.log1p(power)
    prominence = log - median_filter(log, size=9)
    fy = np.fft.fftshift(np.fft.fftfreq(r.shape[0]))
    fx = np.fft.fftshift(np.fft.fftfreq(r.shape[1]))
    radius = np.hypot(fy[:, None], fx[None, :])
    mask = (radius > 0.06) & (radius < 0.68)
    values = prominence[mask]
    median = np.median(values)
    scale = max(1.4826 * np.median(abs(values - median)), 0.1)
    z = (prominence - median) / scale
    peaks = np.where(mask & (z == maximum_filter(z, 3)), z, -np.inf)
    y, x = np.unravel_index(np.argmax(peaks), peaks.shape)
    strength = float(peaks[y, x])
    score = float(np.clip((strength - 4) / 4, 0, 1))
    return DetectorResult(
        "resampling",
        score,
        f"Strongest residual-energy peak: {strength:.2f} robust deviations. Repeating texture can also create peaks.",
        diagnostics={
            "spectrum": log,
            "peak_z": strength,
            "frequency": [float(fy[y]), float(fx[x])],
        },
    )
