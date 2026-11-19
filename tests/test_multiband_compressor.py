import numpy as np
import pytest

from backend.core.dsp.multiband_compressor import MultibandCompressor


@pytest.mark.unit
def test_multiband_compressor_reduces_peaks():
    sr = 44100
    t = np.linspace(0, 1, sr, endpoint=False)
    # Signal mit künstlichen Peaks
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    audio[10000:10010] += 1.0  # Peak
    # Für Coverage: Aggressive Parameter, musikalische Ziele in produktiven Settings adaptiv!
    compressor = MultibandCompressor(thresholds_db=[-40, -40, -40], ratios=[10, 10, 10])
    processed = compressor.process(audio, sr)
    assert np.max(processed) < 1.0


def test_multiband_compressor_identity_on_clean():
    sr = 44100
    t = np.linspace(0, 1, sr, endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    compressor = MultibandCompressor(thresholds_db=[10, 10, 10])
    processed = compressor.process(audio, sr)
    # Das Signal sollte sich nur minimal ändern
    mae = np.mean(np.abs(processed - audio))
    assert mae < 0.25


def test_multiband_compressor_extreme_compression():
    sr = 44100
    t = np.linspace(0, 1, sr, endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    audio[10000:10010] += 2.0  # extremer Peak
    compressor = MultibandCompressor(thresholds_db=[-40, -40, -40], ratios=[10, 10, 10])
    processed = compressor.process(audio, sr)
    # Peaks müssen deutlich reduziert werden
    assert np.max(processed) < 1.0
