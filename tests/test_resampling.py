import numpy as np
import pytest

from fourierforensics.detectors.resampling import detect
from fourierforensics.synthetic import variants


def test_interpolation_evidence():
    samples = variants(12)
    original = detect(samples["original"][0])
    resized = detect(samples["resize"][0])
    assert resized.score > original.score
    assert resized.diagnostics["peak_z"] > original.diagnostics["peak_z"]
    assert 0 <= detect(samples["rotation"][0]).score <= 1


def test_constant_and_small():
    assert detect(np.zeros((64, 64))).score == 0
    with pytest.raises(ValueError):
        detect(np.zeros((16, 16)))
