"""JPEG diagnostics from decoded pixels; not exact compressed coefficients."""

from io import BytesIO

import numpy as np
from PIL import Image
from scipy.fft import dctn

from ..common import DetectorResult
from ..fft import luminance


def ela(image: np.ndarray, quality: int = 90) -> np.ndarray:
    """Mean absolute error after JPEG recompression, on a fixed [0,1] scale."""
    from ..synthetic import decode, jpeg_bytes

    luminance(image)
    a = np.asarray(image)
    if a.dtype != np.uint8:
        a = np.uint8(np.round(a.astype(float) / (65535 if a.dtype == np.uint16 else 1) * 255))
    if not 1 <= quality <= 100:
        raise ValueError("JPEG quality must be in [1,100]")
    if a.ndim == 2:
        a = np.repeat(a[:, :, None], 3, axis=2)
    recompressed = decode(jpeg_bytes(a, quality))
    return np.mean(abs(a.astype(float) - recompressed.astype(float)), axis=2) / 255


def _cdf(x: np.ndarray, scale: float) -> np.ndarray:
    return 0.5 + 0.5 * np.sign(x) * (-np.expm1(-abs(x) / scale))


def _probabilities(q1: float, q2: float, scale: float) -> np.ndarray:
    # Integrate a Laplace density over primary quantization cells, then
    # reassign their mass through the second rounding operation.
    k = np.arange(-4096, 4097)
    mass = _cdf((k + 0.5) * q1, scale) - _cdf((k - 0.5) * q1, scale)
    n = np.round(k * q1 / q2).astype(int)
    idx = np.where(abs(n) <= 40, n + 40, 81)
    p = np.bincount(idx, weights=mass, minlength=82)
    p[:81] = np.convolve(p[:81], [0.03, 0.94, 0.03], mode="same")
    p = np.maximum(p, 1e-12)
    return p / p.sum()


def detect(image: np.ndarray, source_bytes: bytes | None = None) -> DetectorResult:
    a = luminance(image)
    if min(a.shape) < 32:
        raise ValueError("Detectors require at least 32 pixels per side")
    unavailable = lambda reason: DetectorResult("jpeg", 0, reason, available=False)
    if source_bytes is None:
        return unavailable("JPEG histogram analysis requires original JPEG bytes.")
    try:
        with Image.open(BytesIO(source_bytes)) as im:
            if im.format != "JPEG" or not getattr(im, "quantization", None):
                return unavailable("Source is not a supported JPEG.")
            source = np.array(im.convert("RGB"))
            if source.shape[:2] != a.shape or not np.allclose(luminance(source), a, atol=1 / 255):
                raise ValueError("JPEG source bytes do not match analyzed pixels")
            table_id = im.layer[0][2]
            q = np.array(im.quantization[table_id]).reshape(8, 8)
    except (OSError, KeyError) as exc:
        return unavailable(f"Unsupported JPEG metadata: {exc}")
    h, w = (a.shape[0] // 8) * 8, (a.shape[1] // 8) * 8
    blocks = (255 * a[:h, :w] - 128).reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3)
    coeff = dctn(blocks, axes=(-2, -1), norm="ortho").reshape(-1, 8, 8)
    if len(coeff) < 64:
        return unavailable("At least 64 complete JPEG blocks are required.")
    gains, histograms = [], []
    for u, v in [(0, 1), (1, 0), (1, 1), (0, 2), (2, 0), (1, 2), (2, 1), (2, 2)]:
        q2 = float(q[u, v])
        values = np.round(coeff[:, u, v] / q2).astype(int)
        if np.count_nonzero(values) < 32:
            continue
        histogram = np.bincount(np.where(abs(values) <= 40, values + 40, 81), minlength=82)
        scale0 = max(float(np.mean(abs(coeff[:, u, v]))), q2 / 2)
        scales = scale0 * np.array([0.6, 0.8, 1, 1.3, 1.7])
        single = max(float(histogram @ np.log(_probabilities(q2, q2, b))) for b in scales)
        candidates = [
            q1
            for q1 in list(range(1, 41)) + [48, 64, 80]
            if 0.25 <= q1 / q2 <= 6 and abs(q1 / q2 - 1) > 0.1
        ]
        double = max(
            (
                float(histogram @ np.log(_probabilities(q1, q2, b)))
                for q1 in candidates
                for b in scales
            ),
            default=single,
        )
        gains.append(max(0, (double - single - 0.5 * np.log(len(values))) / len(values)))
        histograms.append({"mode": [u, v], "counts": histogram.tolist()})
    if len(gains) < 3:
        return unavailable("Too few informative AC modes for JPEG scoring.")
    score = float(np.clip(np.median(gains) / 0.25, 0, 1))
    return DetectorResult(
        "jpeg",
        score,
        "Decoded DCT histograms compared with single and aligned double quantization models. Quality pair and grid alignment affect sensitivity.",
        diagnostics={"gains": gains, "histograms": histograms},
    )
