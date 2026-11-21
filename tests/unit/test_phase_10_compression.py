"""tests/unit/test_phase_10_compression.py — §v10.700 I1."""

import numpy as np
import pytest

from backend.core.defect_scanner import MaterialType
from backend.core.phases.phase_10_compression import CompressionPhase


@pytest.fixture
def phase():
    return CompressionPhase()


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


def test_phase_10_profiles_cover_every_material_type():
    materials = set(MaterialType)
    assert set(CompressionPhase.COMPRESSION_PARAMS) == materials
    assert set(CompressionPhase.PARALLEL_BLEND) == materials
    assert set(CompressionPhase.DETECTION_MODE) == materials


def test_mp3_high_profile_processes_and_is_reported_as_unvalidated(phase, audio):
    result = phase.process(audio, sample_rate=48000, material_type="mp3_high")

    assert result.success
    assert result.metadata["processing"] == "applied"
    assert result.metadata["material_profile_status"] == "interpolated_candidate_not_ab_tested"
    assert np.isfinite(result.audio).all()


def test_uncalibrated_material_skips_without_changing_audio(phase, audio):
    result = phase.process(audio, sample_rate=48000, material_type="mp3_low")

    assert result.success
    assert result.metadata["processing"] == "skipped_uncalibrated_material"
    np.testing.assert_array_equal(result.audio, audio)


def test_processing_does_not_mutate_shared_material_parameters(phase, audio):
    original = {band: values.copy() for band, values in CompressionPhase.COMPRESSION_PARAMS[MaterialType.VINYL].items()}

    phase.process(audio, sample_rate=48000, material_type="vinyl", tempo_bpm=120)

    assert CompressionPhase.COMPRESSION_PARAMS[MaterialType.VINYL] == original
