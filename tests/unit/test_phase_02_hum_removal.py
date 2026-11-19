"""tests/unit/test_phase_02_hum_removal.py — §v10.700 I1: Unit-Tests für Phase 02 (Hum Removal)."""

import numpy as np
import pytest

from backend.core.phases.phase_02_hum_removal import HumRemovalPhase


@pytest.fixture
def phase():
    return HumRemovalPhase()


@pytest.fixture
def clean_audio():
    t = np.linspace(0, 1, 48000, endpoint=False, dtype=np.float32)
    return (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)


@pytest.fixture
def hum_audio(clean_audio):
    t = np.linspace(0, 1, 48000, endpoint=False, dtype=np.float32)
    hum = 0.1 * np.sin(2 * np.pi * 50 * t) + 0.03 * np.sin(2 * np.pi * 150 * t)
    return (clean_audio + hum).astype(np.float32)


class TestHumRemovalPhase:
    def test_process_returns_ndarray(self, phase, hum_audio):
        result = phase.process(hum_audio, sample_rate=48000, material_type="vinyl")
        assert isinstance(result.audio, np.ndarray)

    def test_no_nan_inf(self, phase, hum_audio):
        result = phase.process(hum_audio, sample_rate=48000, material_type="vinyl")
        assert np.isfinite(result.audio).all()

    def test_not_silent(self, phase, clean_audio):
        result = phase.process(clean_audio, sample_rate=48000, material_type="vinyl")
        rms = float(np.sqrt(np.mean(result.audio**2)))
        assert rms > 1e-10, f"Output is silent (RMS={rms:.2e})"

    def test_reduces_hum(self, phase, clean_audio, hum_audio):
        """Hum Removal sollte 50-Hz-Anteil reduzieren."""
        result = phase.process(hum_audio, sample_rate=48000, material_type="vinyl")
        # Output sollte höhere Korrelation mit Clean haben als Input
        corr_in = float(np.corrcoef(clean_audio, hum_audio)[0, 1])
        corr_out = float(np.corrcoef(clean_audio, result.audio[: len(clean_audio)])[0, 1])
        assert corr_out > 0.80, f"Hum not reduced (corr_out={corr_out:.4f}, corr_in={corr_in:.4f})"

    def test_channels_first_stereo_output_layout_is_preserved(self, phase, hum_audio):
        stereo = np.stack([hum_audio, hum_audio * 0.8], axis=0).astype(np.float32)

        result = phase.process(stereo, sample_rate=48000, material_type="vinyl")

        assert result.audio.shape == stereo.shape
        assert np.isfinite(result.audio).all()

    @pytest.mark.parametrize("shape", [(1, 48000), (48000, 1)])
    def test_two_dimensional_mono_uses_mono_processing_and_preserves_layout(self, phase, monkeypatch, shape):
        mono = np.zeros(shape, dtype=np.float32)
        detected_shapes = []
        applied_shapes = []

        def fake_detect(audio, _params):
            detected_shapes.append(audio.shape)
            return [50]

        def fake_apply(audio, _harmonics, _params):
            applied_shapes.append(audio.shape)
            return audio.copy(), {"reduction_db": 0.0, "harmonics_removed": 0}

        monkeypatch.setattr(phase, "_detect_multi_fundamental", fake_detect)
        monkeypatch.setattr(phase, "_track_harmonics", lambda *_args: [])
        monkeypatch.setattr(phase, "_apply_adaptive_comb", fake_apply)

        result = phase.process(mono, sample_rate=48000, material_type="unknown")

        assert detected_shapes == [(48000,)]
        assert applied_shapes == [(48000,)]
        assert result.audio.shape == mono.shape

    @pytest.mark.parametrize("shape", [(3, 48000), (48000, 3), (2, 48000, 1)])
    def test_unsupported_channel_layout_raises(self, phase, shape):
        with pytest.raises(ValueError, match="erwartet Mono oder Stereo"):
            phase.process(np.zeros(shape, dtype=np.float32), sample_rate=48000)
