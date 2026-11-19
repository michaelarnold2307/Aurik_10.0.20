import pytest

"""Phase 24 short-context STFT safety tests.

Ensures tonal/atonal repair paths stay stable for short before/after windows
and do not hit scipy noverlap/nperseg errors.
"""

import numpy as np

from backend.core.phases.phase_24_dropout_repair import DropoutRepairPhase


@pytest.mark.unit
def test_phase24_repair_tonal_short_context_finite():
    phase = DropoutRepairPhase()
    phase.sample_rate = 48000

    before = np.linspace(-0.1, 0.1, 120, dtype=np.float64)
    after = np.linspace(0.1, -0.1, 96, dtype=np.float64)
    gap_length = 180

    repaired = phase._repair_tonal(before, after, gap_length)

    assert repaired.shape[0] == gap_length
    assert np.isfinite(repaired).all()
    assert np.max(np.abs(repaired)) <= 1.0 + 1e-12


def test_phase24_repair_atonal_short_context_finite():
    phase = DropoutRepairPhase()
    phase.sample_rate = 48000

    rng = np.random.default_rng(42)
    before = rng.normal(0.0, 0.05, 140).astype(np.float64)
    after = rng.normal(0.0, 0.05, 110).astype(np.float64)
    gap_length = 220

    repaired = phase._repair_atonal(before, after, gap_length)

    assert repaired.shape[0] == gap_length
    assert np.isfinite(repaired).all()
    assert np.max(np.abs(repaired)) <= 1.0 + 1e-12


def test_phase24_content_adaptive_strength_keeps_full_compensation():
    """§G188 (GEBOTE.md): keine festen Inhalts-/Material-Dämpfungen — auch bei
    Vocal/Vinyl wird der erkannte Dropout voll kompensiert (Guards schützen)."""
    phase = DropoutRepairPhase()
    phase._current_material = "vinyl"
    phase._current_panns_tags = {"Singing voice": 0.8}

    strength = phase._content_adaptive_repair_strength(0.95, "tonal", 60.0)

    assert abs(strength - 0.95) < 1e-9


def test_phase24_content_adaptive_strength_keeps_atonal_nonvocal_base():
    phase = DropoutRepairPhase()
    phase._current_material = "tape"
    phase._current_panns_tags = {"Singing voice": 0.05}

    strength = phase._content_adaptive_repair_strength(0.90, "atonal", 80.0)

    assert abs(strength - 0.90) < 1e-9
