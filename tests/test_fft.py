import numpy as np
import pytest

from fourierforensics.fft import fft2_centered, luminance, radial_average


@pytest.mark.parametrize("shape", [(64, 64), (65, 71)])
def test_inverse_parseval_and_dc(shape):
    a = np.random.default_rng(1).random(shape)
    f = fft2_centered(a)
    np.testing.assert_allclose(np.fft.ifft2(np.fft.ifftshift(f)).real, a, atol=1e-14)
    assert np.sum(abs(f) ** 2) / a.size == pytest.approx(np.sum(a * a))
    assert abs(f[shape[0] // 2, shape[1] // 2]) == pytest.approx(a.sum())


def test_known_sinusoid():
    a = np.tile(0.5 + 0.4 * np.cos(2 * np.pi * 8 * np.arange(64) / 64), (64, 1))
    f = abs(fft2_centered(a, remove_mean=True))
    assert set(np.argpartition(f.ravel(), -2)[-2:]) == {32 * 64 + 24, 32 * 64 + 40}
    _, means = radial_average(np.ones((64, 64)))
    np.testing.assert_allclose(means[np.isfinite(means)], 1)


@pytest.mark.parametrize(
    "a", [np.zeros((4, 4, 4)), np.array([[np.nan] * 4] * 4), np.ones((4, 4)) * 2]
)
def test_invalid(a):
    with pytest.raises(ValueError):
        luminance(a)


def test_rgb_convention():
    a = np.zeros((4, 4, 3), dtype=np.uint8)
    a[:, :, 0] = 255
    np.testing.assert_allclose(luminance(a), 0.2126)
