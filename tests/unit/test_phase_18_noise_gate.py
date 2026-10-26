"""tests/unit/test_phase_18_noise_gate.py — §v10.700 I1."""

import numpy as np
import pytest

from backend.core.phases.phase_18_noise_gate import NoiseGate


@pytest.fixture
def phase():
    return NoiseGate()


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


def test_panns_vad_offset_field_present_and_neutral_without_kwarg(phase, audio):
    """KP-4 report-only: ohne PANNs-Evidenz bleibt der Offset neutral (0.0) und
    das produktive Gate-Verhalten (Bestandsfelder) bleibt unverändert vorhanden."""
    result = phase.process(audio, sample_rate=48000, material_type="vinyl")
    assert "panns_vad_offset_db" in result.metadata
    # Ohne panns_singing/panns_tags → neutraler Zeugen-Offset 0 dB.
    assert result.metadata["panns_vad_offset_db"] == 0.0


def test_panns_vad_offset_scaled_with_panns_singing(phase, audio):
    """KP-4 report-only: der PANNs-Singing-Offset wird skaliert ins Meta geschrieben,
    OHNE das Gate audio zu verändern (neutraler Bestands-Audio-Output)."""
    # Grep-Verifikation: das Feld wird gesetzt, aber kein Flag geflippt.
    result = phase.process(
        audio,
        sample_rate=48000,
        material_type="vinyl",
        panns_singing=0.6,
    )
    assert result.metadata["panns_vad_offset_db"] == -9.0
    assert isinstance(result.audio, np.ndarray)
    assert np.isfinite(result.audio).all()
