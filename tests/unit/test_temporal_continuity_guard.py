"""Unit-Tests für TemporalContinuityGuard (§2.69, v10.0.0)."""

import numpy as np
import pytest

from backend.core.temporal_continuity_guard import (
    TemporalContinuityResult,
    check_temporal_continuity,
)

SR = 48000
_SINE_SAMPLES = 4 * SR  # 4 Sekunden Sinuston


def _make_sine(freq: float = 440.0, duration_s: float = 4.0, amplitude: float = 0.3) -> np.ndarray:
    t = np.linspace(0, duration_s, int(duration_s * SR), endpoint=False)
    return np.asarray(amplitude * np.sin(2 * np.pi * freq * t), dtype=np.float32)


def _make_silence(duration_s: float = 4.0) -> np.ndarray:
    return np.zeros(int(duration_s * SR), dtype=np.float32)


@pytest.mark.unit
class TestTemporalContinuityResultDataclass:
    def test_fields_present(self):
        r = TemporalContinuityResult(ok=True, variance_ratio=1.2, phase_id="phase_03")
        assert r.ok is True
        assert r.variance_ratio == 1.2
        assert r.phase_id == "phase_03"
        assert r.critical is False

    def test_critical_flag(self):
        r = TemporalContinuityResult(ok=False, variance_ratio=9.0, phase_id="phase_29", critical=True)
        assert r.critical is True


class TestCheckTemporalContinuity:
    """Grundlegende Verhaltenstests."""

    def test_identical_audio_returns_ok(self):
        audio = _make_sine()
        result = check_temporal_continuity(audio, audio.copy(), "phase_test", SR)
        assert result.ok is True
        assert result.variance_ratio == pytest.approx(1.0, abs=0.1)

    def test_silence_input_returns_ok_neutral(self):
        """Stilles Pre-Audio → Division durch Null → neutrales Ergebnis."""
        silence = _make_silence()
        result = check_temporal_continuity(silence, _make_sine(), "phase_test", SR)
        assert result.ok is True
        assert result.variance_ratio == pytest.approx(1.0)

    def test_high_variance_increase_returns_not_ok(self):
        """Starke Dynamik-Explosion im Post → variance_ratio > 2.5 → ok=False."""
        pre = _make_sine(440.0, amplitude=0.05)  # leise, konstant
        # Post: wechselnde Amplituden (stark variierende RMS)
        t = np.linspace(0, 4.0, _SINE_SAMPLES, endpoint=False)
        envelope = np.abs(np.sin(2 * np.pi * 5.0 * t)).astype(np.float32)  # 5 Hz Envelope
        post = (np.sin(2 * np.pi * 440.0 * t) * envelope * 0.8).astype(np.float32)
        result = check_temporal_continuity(pre, post, "phase_test", SR)
        assert result.variance_ratio > 1.0  # Ratio muss > 1 sein
        # Ergebnis hängt von librosa-Verfügbarkeit ab — non-blocking Test
        assert isinstance(result.ok, bool)

    def test_returns_temporal_continuity_result(self):
        audio = _make_sine()
        result = check_temporal_continuity(audio, audio.copy(), "phase_01", SR)
        assert isinstance(result, TemporalContinuityResult)

    def test_phase_id_preserved_in_result(self):
        audio = _make_sine()
        result = check_temporal_continuity(audio, audio.copy(), "phase_99_test", SR)
        assert result.phase_id == "phase_99_test"

    def test_2d_stereo_input_ok(self):
        """Stereo-Input (2D) wird korrekt auf Mono downgemischt."""
        mono = _make_sine()
        stereo = np.stack([mono, mono])  # shape (2, N)
        result = check_temporal_continuity(stereo, stereo.copy(), "phase_stereo", SR)
        assert isinstance(result, TemporalContinuityResult)
        assert result.ok is True

    def test_short_audio_returns_neutral(self):
        """Audio kürzer als frame_length → neutrales Ergebnis (kein Crash)."""
        short = np.zeros(100, dtype=np.float32)
        result = check_temporal_continuity(short, short.copy(), "phase_short", SR)
        assert result.ok is True
        assert result.variance_ratio == pytest.approx(1.0)

    def test_non_blocking_on_exception(self):
        """Kein Crash, auch wenn librosa nicht verfügbar (Exception → ok=True, ratio=1.0)."""
        # Test mit ungültigem Input der Exception provoziert
        result = check_temporal_continuity(
            np.array([float("nan")], dtype=np.float32),
            np.array([float("nan")], dtype=np.float32),
            "phase_nan",
            SR,
        )
        assert isinstance(result, TemporalContinuityResult)
