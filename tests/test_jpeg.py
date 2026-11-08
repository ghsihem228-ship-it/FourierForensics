import numpy as np

from fourierforensics.detectors.jpeg import detect, ela
from fourierforensics.synthetic import scene, variants


def test_single_double_controls():
    samples = variants(14)
    single = detect(*samples["single_jpeg"])
    double = detect(*samples["double_jpeg"])
    assert single.available and double.available
    assert double.score > single.score
    assert double.score > 0.5


def test_unavailable_and_ela():
    assert not detect(scene()).available
    assert not detect(np.zeros((64, 64, 3), dtype=np.uint8), b"invalid").available
    assert np.max(ela(np.zeros((64, 64, 3), dtype=np.uint8))) == 0
    assert ela(scene()).mean() > 0
