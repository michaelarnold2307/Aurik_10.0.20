import numpy as np
import pytest

from backend.core.dsp.automatic_declipper import AutomaticDeclipper


@pytest.mark.unit
def test_declipper_removes_clipping():
    sr = 44100
    t = np.linspace(0, 1, sr, endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    audio[10000:10010] = 1.0  # Clipping
    declipper = AutomaticDeclipper(clip_threshold=0.98)
    processed = declipper.declip(audio, sr)
    # Prüfe, ob die Clipping-Stellen reduziert wurden
    assert np.max(np.abs(processed[10000:10010])) < 1.0


def test_declipper_identity_on_clean():
    sr = 44100
    t = np.linspace(0, 1, sr, endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    declipper = AutomaticDeclipper()
    processed = declipper.declip(audio, sr)
    # Das Signal sollte sich nur minimal ändern
    mae = np.mean(np.abs(processed - audio))
    assert mae < 0.01


def test_declipper_extreme_clipping():
    sr = 44100
    t = np.linspace(0, 1, sr, endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    audio[10000:10020] = 2.0  # Extremes Clipping
    declipper = AutomaticDeclipper(clip_threshold=0.98)
    processed = declipper.declip(audio, sr)
    # Die extremen Clipping-Stellen müssen deutlich reduziert werden
    assert np.max(np.abs(processed[10000:10020])) < 1.0
