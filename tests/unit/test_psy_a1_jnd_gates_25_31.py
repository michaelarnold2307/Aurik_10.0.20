"""Unit-Tests für die JND-gestützten PSY-A1-Gates (Hörordnung §8b) in
phase_25 (Azimut) und phase_31 (Speed-Pitch) — Maßnahme SOTA-PSY-A1-Rest.

Die Schwellen sind jetzt Hör-JND-basiert (hearing_jnd) statt fester
Sample-Konstanten. Verhaltens-Invariante: Das Bestandsverhalten bei 48 kHz
bleibt erhalten (ITD-JND 30 µs × 3,5 ≈ 5 Samples @ 48 kHz).
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.core.dsp.hearing_jnd import jnd
from backend.core.phases.phase_25_azimuth_correction import AzimuthCorrectionPhaseV2
from backend.core.phases.phase_31_speed_pitch_correction import SpeedPitchCorrectionPhase


def _stereo_48k() -> np.ndarray:
    sr = 48000
    t = np.arange(sr, dtype=np.float32) / sr
    left = 0.22 * np.sin(2 * np.pi * 440.0 * t)
    right = 0.20 * np.sin(2 * np.pi * 441.0 * t)
    return np.stack([left, right], axis=1).astype(np.float32)


def _mk_phase25(monkeypatch, shift: float) -> AzimuthCorrectionPhaseV2:
    phase = AzimuthCorrectionPhaseV2()

    def _split_multiband(audio, sample_rate):
        return [audio.copy(), audio.copy(), audio.copy()]

    def _analyze_band_azimuth(band_audio, sample_rate, band_index):
        return type("AzErr", (), {"phase_shift_samples": shift, "confidence": 1.0})()

    def _correct_band_azimuth_timevarying(band_audio, sample_rate, azimuth_error, band_index):
        return (band_audio * 0.15).astype(np.float32)

    def _recombine_multiband(bands):
        return np.mean(np.stack(bands, axis=0), axis=0).astype(np.float32)

    monkeypatch.setattr(phase, "_split_multiband", _split_multiband)
    monkeypatch.setattr(phase, "_analyze_band_azimuth", _analyze_band_azimuth)
    monkeypatch.setattr(phase, "_correct_band_azimuth_timevarying", _correct_band_azimuth_timevarying)
    monkeypatch.setattr(phase, "_recombine_multiband", _recombine_multiband)
    monkeypatch.setattr(phase, "_restore_hf_content", lambda c, a, s, h: c)
    monkeypatch.setattr(phase, "_measure_hf_loss", lambda l, r, s: 0.0)
    return phase


@pytest.mark.unit
class TestPhase25JndGate:
    def test_jnd_table_provides_expected_threshold(self) -> None:
        """ITD-JND ist 30 µs (hearing_jnd) — Basis der Azimut-Schwelle."""
        assert jnd("itd_tone") == pytest.approx(30e-6, rel=1e-9)
        assert jnd("level_tone_1khz") > 0.0

    def test_subjnd_shift_is_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """5 Samples @ 48 kHz = 104 µs < 3,5 × ITD-JND (105 µs) → Skip (wie bisher)."""
        phase = _mk_phase25(monkeypatch, shift=5.0)
        src = _stereo_48k()
        result = phase.process(src, 48000, "tape", strength=1.0)
        assert result.audio.shape == src.shape
        assert np.array_equal(result.audio, src)  # unverändert
        assert result.metadata.get("reason") == "below_threshold"
        assert result.metrics.get("threshold_phase_shift") == pytest.approx(5.04, abs=0.05)
        # HF-Schwelle fällt nicht unter die Pegel-JND (max-Floor).
        assert result.metrics.get("threshold_hf_loss_db") == pytest.approx(2.3, abs=0.05)

    def test_above_jnd_shift_is_corrected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """6 Samples = 125 µs > JND-Schwelle → Korrektur greift (wie bisher)."""
        phase = _mk_phase25(monkeypatch, shift=6.0)
        src = _stereo_48k()
        result = phase.process(src, 48000, "tape", strength=1.0)
        assert not np.array_equal(result.audio, src)
        assert result.metrics.get("phase_shift_before_samples") == pytest.approx(6.0)

    def test_hf_loss_alone_triggers_correction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Sekundärkriterium: HF-Verlust über 2,3 dB korrigiert auch ohne Phasen-Shift."""
        phase = _mk_phase25(monkeypatch, shift=0.0)
        monkeypatch.setattr(phase, "_measure_hf_loss", lambda l, r, s: 3.1)
        src = _stereo_48k()
        result = phase.process(src, 48000, "tape", strength=1.0)
        assert not np.array_equal(result.audio, src)


@pytest.mark.unit
class TestPhase31JndThreshold:
    def test_skip_reason_is_jnd_documented(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Unter-Schwellen-Pfad nennt die JND-gestützte Schwelle (§PSY-A8)."""
        phase = SpeedPitchCorrectionPhase(sample_rate=48000)
        monkeypatch.setattr(phase, "_detect_pitch_pyin", lambda audio, params: (440.0, 1.0))
        monkeypatch.setattr(phase, "_compute_tuning_offset", lambda a, s, rp, dp: (0.0, 1.0))
        t = np.linspace(0, 1, 48000, endpoint=False, dtype=np.float32)
        src = (np.sin(2 * np.pi * 440.0 * t) * 0.3).astype(np.float32)
        result = phase.process(src, sample_rate=48000, material_type="vinyl", quality_mode="fast")
        reason = str(result.modifications.get("reason", ""))
        assert "JND-gestützt" in reason
        # max(0,3 %, Frequenz-JND 0,2 %) = 0,30 % — Bestandsverhalten bleibt.
        assert "0.30%" in reason
        assert np.array_equal(result.audio, src)
