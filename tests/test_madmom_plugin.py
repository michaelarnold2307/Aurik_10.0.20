import numpy as np


def test_beats_plugin_desktop():
    """Smoke-Test für den kanonischen BEATs-Tagger (Nachfolger des legacy madmom-Setups)."""
    from plugins.beats_plugin import get_beats_plugin

    plugin = get_beats_plugin()
    sr = 48000
    t = np.linspace(0, 1, sr, endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    result = plugin.get_tags(audio, sr, top_k=5)

    assert result is not None
    assert isinstance(result.tags, dict)
    assert len(result.top_k) <= 5
