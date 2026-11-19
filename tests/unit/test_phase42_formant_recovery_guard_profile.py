from __future__ import annotations

import pytest

from backend.core.defect_scanner import MaterialType
from backend.core.phases.phase_42_vocal_enhancement import VocalEnhancement as VocalEnhancementPhase


@pytest.mark.unit
def test_formant_recovery_guard_profile_keys_and_bounds() -> None:
    p = VocalEnhancementPhase._compute_formant_recovery_guard_profile(
        material_type=MaterialType.VINYL,
        quality_mode="quality",
        restorability_score=55.0,
    )

    assert set(p.keys()) == {"headroom_min_db", "correction_min_db", "eq_min_gain_db"}
    assert 0.25 <= p["headroom_min_db"] <= 0.80
    assert 0.10 <= p["correction_min_db"] <= 0.35
    assert 0.02 <= p["eq_min_gain_db"] <= 0.10


def test_low_restorability_makes_recovery_more_permissive() -> None:
    low = VocalEnhancementPhase._compute_formant_recovery_guard_profile(
        material_type=MaterialType.SHELLAC,
        quality_mode="quality",
        restorability_score=10.0,
    )
    high = VocalEnhancementPhase._compute_formant_recovery_guard_profile(
        material_type=MaterialType.SHELLAC,
        quality_mode="quality",
        restorability_score=90.0,
    )

    assert low["headroom_min_db"] < high["headroom_min_db"]
    assert low["correction_min_db"] < high["correction_min_db"]


def test_fast_mode_is_more_conservative_than_quality() -> None:
    fast = VocalEnhancementPhase._compute_formant_recovery_guard_profile(
        material_type=MaterialType.VINYL,
        quality_mode="fast",
        restorability_score=60.0,
    )
    quality = VocalEnhancementPhase._compute_formant_recovery_guard_profile(
        material_type=MaterialType.VINYL,
        quality_mode="quality",
        restorability_score=60.0,
    )

    assert fast["headroom_min_db"] > quality["headroom_min_db"]
    assert fast["correction_min_db"] > quality["correction_min_db"]
    assert fast["eq_min_gain_db"] > quality["eq_min_gain_db"]


def test_hard_analog_material_is_more_permissive_than_mid_analog() -> None:
    hard = VocalEnhancementPhase._compute_formant_recovery_guard_profile(
        material_type=MaterialType.SHELLAC,
        quality_mode="balanced",
        restorability_score=60.0,
    )
    mid = VocalEnhancementPhase._compute_formant_recovery_guard_profile(
        material_type=MaterialType.TAPE,
        quality_mode="balanced",
        restorability_score=60.0,
    )

    assert hard["headroom_min_db"] < mid["headroom_min_db"]
    assert hard["correction_min_db"] < mid["correction_min_db"]
