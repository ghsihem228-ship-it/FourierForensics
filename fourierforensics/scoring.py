"""Explicit, uncalibrated fusion of independent artifact clues."""

from dataclasses import dataclass

import numpy as np

from .common import DetectorResult
from .detectors import jpeg, noise, resampling
from .fft import fft2_centered, log_magnitude, radial_average

WEIGHTS = {"resampling": 0.40, "noise": 0.35, "jpeg": 0.25}


def combine(results: list[DetectorResult]) -> float:
    """0.75 max(score) + 0.25 weighted mean over available detectors."""
    active = [r for r in results if r.available]
    if not active:
        return 0.0
    weights = np.array([WEIGHTS[r.name] for r in active])
    values = np.array([r.score for r in active])
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Detector scores must be finite and in [0,1]")
    return float(0.75 * values.max() + 0.25 * np.average(values, weights=weights))


@dataclass
class Analysis:
    score: float
    detectors: list[DetectorResult]
    spectrum: np.ndarray
    radial_frequency: np.ndarray
    radial_power: np.ndarray

    @property
    def heatmap(self) -> np.ndarray:
        return self.detectors[1].diagnostics["heatmap"]

    def summary(self) -> dict:
        return {
            "score": self.score,
            "interpretation": "Artifact suspicion, not probability of forgery",
            "detectors": [d.summary() for d in self.detectors],
            "weights": WEIGHTS,
        }


def analyze(image: np.ndarray, source_bytes: bytes | None = None) -> Analysis:
    results = [resampling.detect(image), noise.detect(image), jpeg.detect(image, source_bytes)]
    frequency, power = radial_average(abs(fft2_centered(image, window=True, remove_mean=True)) ** 2)
    return Analysis(combine(results), results, log_magnitude(image), frequency, power)
