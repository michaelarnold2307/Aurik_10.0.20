import numpy as np


def test_mert_plugin_desktop():
    """Smoke-Test für das kanonische MERT-Plugin (Naturalness-Analyse)."""
    from plugins.mert_plugin import get_mert_plugin

    plugin = get_mert_plugin()
    sr = 24000
    t = np.linspace(0, 1, sr, endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    result = plugin.analyze(audio, sr)

    assert result is not None
    assert 0.0 <= float(result.naturalness_score) <= 1.0
    assert result.analysis_frames >= 0
