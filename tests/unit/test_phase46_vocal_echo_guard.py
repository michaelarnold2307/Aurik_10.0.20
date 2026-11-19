from __future__ import annotations

import pytest

"""Regressiontests fuer Phase 46 Vocal-Echo-Guard in Restoration."""


import numpy as np

from backend.core.phases.phase_46_spatial_enhancement import SpatialEnhancementPhase


def _make_stereo_signal(sr: int = 48000, seconds: float = 1.0) -> np.ndarray:
    n = int(sr * seconds)
    t = np.arange(n, dtype=np.float32) / float(sr)
    # Vokal-nahe Grundstruktur: Grundton + Formantnahe Obertone
    mono = 0.20 * np.sin(2.0 * np.pi * 220.0 * t) + 0.05 * np.sin(2.0 * np.pi * 880.0 * t)
    # Leicht unterschiedliche Kanaele (realistische Stereoaufnahme)
    left = mono
    right = 0.98 * mono
    return np.column_stack([left, right]).astype(np.float32)


@pytest.mark.unit
def test_phase46_vocal_safe_mode_for_vocals_in_restoration() -> None:
    phase = SpatialEnhancementPhase()
    audio = _make_stereo_signal()

    result = phase.process(
        audio,
        sample_rate=48000,
        strength=1.0,
        quality_mode="restoration",
        panns_singing=0.75,
    )

    assert result.success
    assert result.metadata.get("algorithm") == "vocal_echo_guard_safe"
    assert result.metadata.get("dry_wet") == 0.0
    assert result.metadata.get("diffuse") is False


def test_phase46_not_bypassed_for_studio_mode() -> None:
    phase = SpatialEnhancementPhase()
    audio = _make_stereo_signal()

    result = phase.process(
        audio,
        sample_rate=48000,
        strength=1.0,
        quality_mode="studio_2026",
        panns_singing=0.75,
    )

    assert result.success
    assert result.metadata.get("algorithm") == "phase_46_default"
