import numpy as np
import pytest

from plugins.demucs_v4_plugin import DemucsV4Plugin

pytestmark = [pytest.mark.ml, pytest.mark.slow]


def test_demucs_v4_plugin_aurik90():
    sr = 48000
    t = np.linspace(0, 1, sr)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    plugin = DemucsV4Plugin()
    stems = plugin.process(audio, sr=sr)
    assert isinstance(stems, dict)
