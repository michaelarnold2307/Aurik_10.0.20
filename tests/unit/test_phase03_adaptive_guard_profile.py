from __future__ import annotations

import pytest

from backend.core.phases.phase_03_denoise import DenoisePhase


@pytest.mark.unit
def test_adaptive_guard_profile_keys_and_bounds() -> None:
    profile = DenoisePhase._compute_adaptive_guard_profile(
        material_type="vinyl",
        quality_mode="quality",
        restorability_score=55.0,
    )

    assert set(profile.keys()) == {
        "quality_warning_threshold",
        "energy_min_ratio",
        "energy_target_ratio",
    }
    assert 0.55 <= profile["quality_warning_threshold"] <= 0.85
    assert 0.14 <= profile["energy_min_ratio"] <= 0.32
    assert 0.20 <= profile["energy_target_ratio"] <= 0.45
    assert profile["energy_target_ratio"] >= profile["energy_min_ratio"] + 0.02


def test_low_restorability_relaxes_quality_warning() -> None:
    low = DenoisePhase._compute_adaptive_guard_profile(
        material_type="vinyl",
        quality_mode="balanced",
        restorability_score=10.0,
    )
    high = DenoisePhase._compute_adaptive_guard_profile(
        material_type="vinyl",
        quality_mode="balanced",
        restorability_score=90.0,
    )

    assert low["quality_warning_threshold"] < high["quality_warning_threshold"]


def test_fast_mode_uses_stricter_energy_preservation_than_quality() -> None:
    fast = DenoisePhase._compute_adaptive_guard_profile(
        material_type="vinyl",
        quality_mode="fast",
        restorability_score=60.0,
    )
    quality = DenoisePhase._compute_adaptive_guard_profile(
        material_type="vinyl",
        quality_mode="quality",
        restorability_score=60.0,
    )

    assert fast["energy_min_ratio"] > quality["energy_min_ratio"]


def test_digital_material_is_more_conservative_than_analog() -> None:
    digital = DenoisePhase._compute_adaptive_guard_profile(
        material_type="cd_digital",
        quality_mode="balanced",
        restorability_score=60.0,
    )
    analog = DenoisePhase._compute_adaptive_guard_profile(
        material_type="tape",
        quality_mode="balanced",
        restorability_score=60.0,
    )

    assert digital["energy_min_ratio"] > analog["energy_min_ratio"]
    assert digital["quality_warning_threshold"] > analog["quality_warning_threshold"]
