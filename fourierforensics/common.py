"""Shared detector contract and high-pass filtering."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.ndimage import convolve

from .fft import luminance


@dataclass
class DetectorResult:
    name: str
    score: float
    explanation: str
    available: bool = True
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict:
        """JSON-safe scalar summary; large maps stay in the analysis object."""
        return {
            "name": self.name,
            "score": float(self.score),
            "available": self.available,
            "explanation": self.explanation,
        }


def highpass(image: np.ndarray) -> np.ndarray:
    """Laplacian R = 4I - I_up - I_down - I_left - I_right."""
    a = luminance(image)
    if min(a.shape) < 32:
        raise ValueError("Detectors require at least 32 pixels per side")
    return convolve(a, np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]), mode="reflect")
