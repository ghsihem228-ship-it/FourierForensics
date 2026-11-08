import numpy as np
import pytest

from fourierforensics.detectors.noise import detect
from fourierforensics.synthetic import variants


def test_splice_localization():
    samples = variants(13)
    a = detect(samples["original"][0])
    b = detect(samples["splice"][0])
    assert b.score > a.score
    heat = b.diagnostics["heatmap"]
    assert heat[64:192, 64:192].mean() > heat[:64].mean() + 0.3


def test_known_variance_and_odd_size():
    a = 0.5 + np.random.default_rng(4).normal(0, 0.02, (257, 259))
    r = detect(a)
    assert np.median(r.diagnostics["block_variance"]) == pytest.approx(0.02**2, rel=0.2)
    assert r.diagnostics["heatmap"].shape == a.shape
    assert detect(np.zeros((64, 64))).score == 0
