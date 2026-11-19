import numpy as np
import pytest

from backend.core.dsp.analysis_and_quality import RMSEnergy, SpectralCentroid, SpectralRolloff, ZeroCrossingRate


@pytest.mark.unit
def test_spectral_centroid_basic():
    audio = np.ones(1024)
    sr = 44100
    centroid = SpectralCentroid().process(audio, sr)
    assert centroid.shape == (1,)
    assert centroid[0] == 0.0


def test_spectral_centroid_sinus():
    sr = 44100
    t = np.linspace(0, 1, sr, endpoint=False)
    audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz Sinus
    centroid = SpectralCentroid().process(audio, sr)
    assert centroid.shape == (1,)
    assert centroid[0] > 0


def test_spectral_rolloff_basic():
    audio = np.ones(1024)
    sr = 44100
    rolloff = SpectralRolloff().process(audio, sr)
    assert rolloff.shape == (1,)
    assert rolloff[0] == 0.0


def test_spectral_rolloff_sinus():
    sr = 44100
    t = np.linspace(0, 1, sr, endpoint=False)
    audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz Sinus
    rolloff = SpectralRolloff().process(audio, sr)
    assert rolloff.shape == (1,)
    assert rolloff[0] > 0


def test_rms_energy_basic():
    audio = np.ones(1024)
    rms = RMSEnergy().process(audio)
    assert rms.shape == (1,)
    assert np.isclose(rms[0], 1.0)


def test_zero_crossing_rate_basic():
    audio = np.zeros(1024)
    zcr = ZeroCrossingRate().process(audio)
    assert zcr.shape == (1,)
    assert zcr[0] == 0.0
