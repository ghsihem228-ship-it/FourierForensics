"""Robust block noise estimates and a spatial anomaly map."""

import numpy as np
from scipy.ndimage import uniform_filter

from ..common import DetectorResult, highpass


def detect(image: np.ndarray, block_size: int = 32) -> DetectorResult:
    """Estimate variance using MAD(R)/(0.67449 sqrt(20)).

    The factor 20 is the squared norm of the Laplacian kernel for iid noise.
    Equal partitions avoid tiny remainder blocks on odd image dimensions.
    """
    if block_size < 8:
        raise ValueError("block_size must be at least 8")
    r = highpass(image)
    ys = np.linspace(0, r.shape[0], max(1, r.shape[0] // block_size) + 1, dtype=int)
    xs = np.linspace(0, r.shape[1], max(1, r.shape[1] // block_size) + 1, dtype=int)
    variances = np.empty((len(ys) - 1, len(xs) - 1))
    for y in range(len(ys) - 1):
        for x in range(len(xs) - 1):
            tile = r[ys[y] + 1 : ys[y + 1] - 1, xs[x] + 1 : xs[x + 1] - 1]
            sigma = np.median(abs(tile - np.median(tile))) / (0.67448975 * np.sqrt(20))
            variances[y, x] = sigma * sigma
    log = np.log(variances + 1e-8)
    median = np.median(log)
    scale = max(1.4826 * np.median(abs(log - median)), np.log(2))
    anomalies = np.clip((abs(log - median) / scale - 1) / 3, 0, 1)
    heatmap = np.zeros(r.shape)
    for y in range(len(ys) - 1):
        for x in range(len(xs) - 1):
            heatmap[ys[y] : ys[y + 1], xs[x] : xs[x + 1]] = anomalies[y, x]
    k = max(1, int(np.ceil(anomalies.size * 0.1)))
    score = float(np.sort(anomalies.ravel())[-k:].mean())
    local = np.maximum(uniform_filter(r * r, 7) - uniform_filter(r, 7) ** 2, 0) / 20
    return DetectorResult(
        "noise",
        score,
        "Block noise differs from the image median. Texture, denoising and lighting can also cause differences.",
        diagnostics={"heatmap": heatmap, "block_variance": variances, "local_variance": local},
    )
