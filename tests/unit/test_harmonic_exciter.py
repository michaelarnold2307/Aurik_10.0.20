import numpy as np
import pytest

from backend.core.dsp.harmonic_exciter import HarmonicExciter, HarmonicExciterStudio


@pytest.mark.unit
def test_harmonic_exciter_mono_shape_finite_and_clip() -> None:
    sr = 48000
    t = np.linspace(0.0, 1.0, sr, endpoint=False)
    audio = (0.3 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float64)

    out = HarmonicExciter().process(audio, sr)
    assert out.shape == audio.shape
    assert np.isfinite(out).all()
    assert np.max(np.abs(out)) <= 1.0


def test_harmonic_exciter_stereo_supported() -> None:
    sr = 48000
    t = np.linspace(0.0, 1.0, sr, endpoint=False)
    left = 0.2 * np.sin(2.0 * np.pi * 330.0 * t)
    right = 0.15 * np.sin(2.0 * np.pi * 660.0 * t)
    audio = np.stack([left, right], axis=0).astype(np.float64)

    out = HarmonicExciter(amount=0.7, saturation=0.8).process(audio, sr)
    assert out.shape == audio.shape
    assert np.isfinite(out).all()
    assert np.max(np.abs(out)) <= 1.0


def test_harmonic_exciter_handles_nan_inf() -> None:
    sr = 48000
    audio = np.zeros(sr, dtype=np.float64)
    audio[1] = np.nan
    audio[2] = np.inf
    audio[3] = -np.inf

    out = HarmonicExciter().process(audio, sr)
    assert np.isfinite(out).all()


def test_harmonic_exciter_studio_finite_clip() -> None:
    sr = 48000
    t = np.linspace(0.0, 1.0, sr, endpoint=False)
    audio = (0.95 * np.sin(2.0 * np.pi * 1000.0 * t)).astype(np.float64)

    out = HarmonicExciterStudio(amount=0.9).process(audio, sr)
    assert out.shape == audio.shape
    assert np.isfinite(out).all()
    assert np.max(np.abs(out)) <= 1.0
