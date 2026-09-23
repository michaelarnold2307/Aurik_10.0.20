"""tests/unit/test_phase_53_semantic_audio.py — §v10.700 I1."""

import numpy as np
import pytest

from backend.core.phases.phase_53_semantic_audio import SemanticAudioPhase


@pytest.fixture
def phase():
    return SemanticAudioPhase()


@pytest.fixture
def audio():
    rng = np.random.RandomState(42)
    t = np.linspace(0, 1, 48000, endpoint=False, dtype=np.float32)
    return (np.sin(2 * np.pi * 440 * t) * 0.5 + rng.randn(48000) * 0.01).astype(np.float32)


def test_returns_ndarray(phase, audio):
    result = phase.process(audio, sample_rate=48000, material_type="vinyl")
    assert isinstance(result.audio, np.ndarray)


def test_no_nan_inf(phase, audio):
    result = phase.process(audio, sample_rate=48000, material_type="vinyl")
    assert np.isfinite(result.audio).all()


def test_not_silent(phase, audio):
    result = phase.process(audio, sample_rate=48000, material_type="vinyl")
    assert float(np.sqrt(np.mean(result.audio**2))) > 1e-10


def test_length_preserved(phase, audio):
    result = phase.process(audio, sample_rate=48000, material_type="vinyl")
    assert len(result.audio) == len(audio)


class TestBeatSyncedTimeConstants:
    """§2.69c Beat-Sync für Dynamik-Zeitkonstanten (modusabhängig)."""

    def test_restoration_never_shorter_than_base(self) -> None:
        from backend.core.audio_utils import resolve_beat_synced_time_ms

        # 120 BPM → 1 Beat = 500 ms; Base 200 ms rastet auf 500 ms ein.
        assert resolve_beat_synced_time_ms(200.0, 120.0) == 500.0
        # Base 800 ms bleibt (nie kürzer als Material-Default).
        assert resolve_beat_synced_time_ms(800.0, 120.0) == 800.0

    def test_studio_tight_and_punchy(self) -> None:
        from backend.core.audio_utils import resolve_beat_synced_time_ms

        # 120 BPM: min(200×0.75, 500) = 150 ms.
        assert resolve_beat_synced_time_ms(200.0, 120.0, is_studio=True) == 150.0
        # Untergrenze 10 ms greift.
        assert resolve_beat_synced_time_ms(5.0, 300.0, is_studio=True, beats=0.25) == 10.0

    def test_invalid_tempo_passthrough(self) -> None:
        from backend.core.audio_utils import resolve_beat_synced_time_ms

        assert resolve_beat_synced_time_ms(200.0, None) == 200.0
        assert resolve_beat_synced_time_ms(200.0, 500.0) == 200.0
        assert resolve_beat_synced_time_ms(200.0, 10.0) == 200.0
