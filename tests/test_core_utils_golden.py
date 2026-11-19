import numpy as np

from backend.core.core_utils import audio_stats


# Golden Sample: Referenzsignal (z.B. Sinus)
def golden_sample():
    t = np.linspace(0, 1, 48000, endpoint=False)
    return 0.5 * np.sin(2 * np.pi * 440 * t)


def test_audio_stats_golden():
    ref = golden_sample()
    stats = audio_stats(ref)
    assert np.isclose(stats["peak"], 0.5, atol=1e-3)
    assert np.isfinite(stats["rms"])
    assert np.isfinite(stats["loudness"])
