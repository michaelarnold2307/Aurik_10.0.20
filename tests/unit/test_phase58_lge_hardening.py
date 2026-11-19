import types

import numpy as np
import pytest


@pytest.mark.unit
def test_phase58_load_retry_then_passthrough(monkeypatch):
    from backend.core.phases.phase_58_lyrics_guided_enhancement import Phase58LyricsGuidedEnhancement

    phase = Phase58LyricsGuidedEnhancement()
    audio = np.random.default_rng(1).uniform(-0.1, 0.1, 48_000).astype(np.float32)

    calls = {"get": 0}

    def _raise_get():
        calls["get"] += 1
        raise RuntimeError("lge unavailable")

    fake_module = types.SimpleNamespace(get_lyrics_guided_enhancement=_raise_get)
    monkeypatch.setitem(__import__("sys").modules, "backend.core.lyrics_guided_enhancement", fake_module)

    result = phase.process(audio, 48_000, vocal_probability=0.9, strength=1.0)

    assert result.success is True
    assert np.allclose(result.audio, audio, atol=1e-7)
    assert calls["get"] == 2
    assert result.metadata.get("retry_attempted") is True
    assert int(result.metadata.get("load_attempts", 0)) == 2


def test_phase58_enhance_retry_second_attempt_success(monkeypatch):
    from backend.core.phases.phase_58_lyrics_guided_enhancement import Phase58LyricsGuidedEnhancement

    phase = Phase58LyricsGuidedEnhancement()
    audio = np.random.default_rng(2).uniform(-0.1, 0.1, 48_000).astype(np.float32)

    class _FakeLGE:
        def __init__(self):
            self.calls = 0

        def enhance(self, audio_in, sample_rate, strength: float = 1.0):
            self.calls += 1
            if self.calls == 1:
                raise ValueError("transient failure")
            transcription = types.SimpleNamespace(words=["a", "b", "c"])
            return np.asarray(audio_in, dtype=np.float32) * 0.95, transcription

    fake_lge = _FakeLGE()

    def _get_lge():
        return fake_lge

    fake_module = types.SimpleNamespace(get_lyrics_guided_enhancement=_get_lge)
    monkeypatch.setitem(__import__("sys").modules, "backend.core.lyrics_guided_enhancement", fake_module)

    result = phase.process(audio, 48_000, vocal_probability=0.9, strength=1.0)

    assert result.success is True
    assert fake_lge.calls == 2
    assert result.metadata.get("lge_active") is True
    assert int(result.metadata.get("n_phoneme_segments", 0)) == 3
