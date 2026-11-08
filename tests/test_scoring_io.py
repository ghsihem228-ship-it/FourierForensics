from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from fourierforensics.common import DetectorResult
from fourierforensics.io import load_image
from fourierforensics.scoring import analyze, combine
from fourierforensics.synthetic import variants


def test_fusion_missing_evidence():
    results = [
        DetectorResult("resampling", 0.8, ""),
        DetectorResult("noise", 0.2, ""),
        DetectorResult("jpeg", 1, "", available=False),
    ]
    assert combine(results) == pytest.approx(0.75 * 0.8 + 0.25 * (0.4 * 0.8 + 0.35 * 0.2) / 0.75)
    assert combine([]) == 0


def test_pipeline_and_source_bytes():
    a, data = variants(15)["single_jpeg"]
    loaded = load_image(data)
    np.testing.assert_array_equal(loaded.pixels, a)
    result = analyze(a, data)
    assert result.heatmap.shape == a.shape[:2]
    assert 0 <= result.score <= 1
    assert loaded.source_bytes == data


def test_loader_errors_and_transparency():
    with pytest.raises(ValueError):
        load_image(b"not an image")
    stream = BytesIO()
    Image.new("RGBA", (64, 64), (0, 0, 0, 0)).save(stream, format="PNG")
    loaded = load_image(stream.getvalue())
    assert loaded.warnings
    assert loaded.pixels.min() == 255
