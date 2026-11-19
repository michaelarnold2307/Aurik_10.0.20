"""tests/unit/test_phase_44_guitar_enhancement.py — §v10.700 I1."""

import numpy as np
import pytest

from backend.core.phases.phase_44_guitar_enhancement import GuitarEnhancementPhase


@pytest.fixture
def phase():
    return GuitarEnhancementPhase()


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


def test_psy_a4_tempering_multiplies_strength(phase, audio, monkeypatch):
    """§SOTA-PSY-A4: Der Equal-Loudness-Faktor multipliziert die Effektiv-Stärke (Wiring-Beweis)."""
    from backend.core import fletcher_munson_curves as _fmc

    monkeypatch.setattr(
        _fmc,
        "equal_loudness_strength_factor",
        lambda f, target_phon=60, reference_freq_hz=1000.0, min_factor=0.5: np.full(
            np.asarray(f, dtype=float).shape, 1.0
        ),
    )
    ref = phase.process(audio, sample_rate=48000, material_type="vinyl", strength=1.0)
    eff_ref = float(ref.metadata.get("effective_strength", 0.0))

    monkeypatch.setattr(
        _fmc,
        "equal_loudness_strength_factor",
        lambda f, target_phon=60, reference_freq_hz=1000.0, min_factor=0.5: np.full(
            np.asarray(f, dtype=float).shape, 0.7
        ),
    )
    tempered = phase.process(audio, sample_rate=48000, material_type="vinyl", strength=1.0)
    eff_tempered = float(tempered.metadata.get("effective_strength", 0.0))

    assert eff_tempered == pytest.approx(eff_ref * 0.7, rel=1e-4)
