import numpy as np
import pytest

from backend.core.dsp.auto_eq import AutoEQ


@pytest.mark.unit
def test_auto_eq_process_mono_shape_and_finite() -> None:
    sr = 48000
    t = np.linspace(0.0, 1.0, sr, endpoint=False)
    audio = (0.3 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float64)

    out = AutoEQ().process(audio, sr)
    assert out.shape == audio.shape
    assert np.isfinite(out).all()
    assert np.max(np.abs(out)) <= 1.0


def test_auto_eq_process_stereo_shape_and_finite() -> None:
    sr = 48000
    t = np.linspace(0.0, 1.0, sr, endpoint=False)
    left = 0.25 * np.sin(2.0 * np.pi * 220.0 * t)
    right = 0.20 * np.sin(2.0 * np.pi * 880.0 * t)
    audio = np.stack([left, right], axis=0).astype(np.float64)

    out = AutoEQ().process(audio, sr)
    assert out.shape == audio.shape
    assert np.isfinite(out).all()
    assert np.max(np.abs(out)) <= 1.0


def test_auto_eq_replaces_nan_and_inf() -> None:
    sr = 48000
    audio = np.zeros(sr, dtype=np.float64)
    audio[10] = np.nan
    audio[20] = np.inf
    audio[30] = -np.inf

    out = AutoEQ().process(audio, sr)
    assert np.isfinite(out).all()
    assert np.max(np.abs(out)) <= 1.0
