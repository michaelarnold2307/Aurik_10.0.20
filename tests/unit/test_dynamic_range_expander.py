import numpy as np
import pytest

from backend.core.dsp.dynamic_range_expander import DynamicRangeExpander


@pytest.mark.unit
def test_dynamic_range_expander_mono_finite_and_shape() -> None:
    sr = 48000
    t = np.linspace(0.0, 1.0, sr, endpoint=False)
    audio = (0.2 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float64)

    out = DynamicRangeExpander().process(audio, sr)
    assert out.shape == audio.shape
    assert np.isfinite(out).all()
    assert np.max(np.abs(out)) <= 1.0


def test_dynamic_range_expander_stereo_supported() -> None:
    sr = 48000
    t = np.linspace(0.0, 1.0, sr, endpoint=False)
    left = 0.25 * np.sin(2.0 * np.pi * 220.0 * t)
    right = 0.20 * np.sin(2.0 * np.pi * 880.0 * t)
    audio = np.stack([left, right], axis=0).astype(np.float64)

    out = DynamicRangeExpander().process(audio, sr)
    assert out.shape == audio.shape
    assert np.isfinite(out).all()
    assert np.max(np.abs(out)) <= 1.0


def test_dynamic_range_expander_handles_nan_inf() -> None:
    sr = 48000
    audio = np.zeros(sr, dtype=np.float64)
    audio[5] = np.nan
    audio[6] = np.inf
    audio[7] = -np.inf

    out = DynamicRangeExpander().process(audio, sr)
    assert np.isfinite(out).all()
