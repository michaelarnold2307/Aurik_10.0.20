from __future__ import annotations

import pytest

from backend.core.phases.phase_49_advanced_dereverb import AdvancedDereverbPhase


@pytest.mark.unit
def test_wet_mix_guard_profile_keys_and_bounds() -> None:
    p = AdvancedDereverbPhase._adaptive_wet_mix_guard_profile(
        material_key="vinyl",
        quality_mode="quality",
        restorability_score=55.0,
    )

    assert set(p.keys()) == {
        "wet_curve_exp",
        "attenuation_guard_floor",
        "rescue_wet_floor",
        "scratch_guard_floor",
    }
    assert 0.95 <= p["wet_curve_exp"] <= 1.35
    assert 0.25 <= p["attenuation_guard_floor"] <= 0.45
    assert 0.15 <= p["rescue_wet_floor"] <= 0.30
    assert 0.18 <= p["scratch_guard_floor"] <= 0.35


def test_fast_mode_is_more_conservative_than_quality() -> None:
    fast = AdvancedDereverbPhase._adaptive_wet_mix_guard_profile(
        material_key="cd_digital",
        quality_mode="fast",
        restorability_score=60.0,
    )
    quality = AdvancedDereverbPhase._adaptive_wet_mix_guard_profile(
        material_key="cd_digital",
        quality_mode="quality",
        restorability_score=60.0,
    )

    assert fast["wet_curve_exp"] > quality["wet_curve_exp"]
    assert fast["attenuation_guard_floor"] > quality["attenuation_guard_floor"]
    assert fast["rescue_wet_floor"] > quality["rescue_wet_floor"]
    assert fast["scratch_guard_floor"] > quality["scratch_guard_floor"]


def test_low_restorability_relaxes_wet_curve_exponent() -> None:
    low = AdvancedDereverbPhase._adaptive_wet_mix_guard_profile(
        material_key="tape",
        quality_mode="balanced",
        restorability_score=10.0,
    )
    high = AdvancedDereverbPhase._adaptive_wet_mix_guard_profile(
        material_key="tape",
        quality_mode="balanced",
        restorability_score=90.0,
    )

    assert low["wet_curve_exp"] < high["wet_curve_exp"]
