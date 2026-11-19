from __future__ import annotations

import pytest

from backend.core.defect_scanner import MaterialType
from backend.core.phases.phase_12_wow_flutter_fix import WowFlutterFix


@pytest.mark.unit
def test_profile_keys_present() -> None:
    p = WowFlutterFix._compute_adaptive_threshold_profile(
        material=MaterialType.VINYL,
        quality_mode="quality",
        restorability_score=55.0,
    )
    assert set(p.keys()) == {
        "detection_threshold",
        "yin_threshold",
        "ml_confidence_threshold",
        "min_confidence_for_correction",
    }


def test_profile_bounds_are_enforced() -> None:
    p = WowFlutterFix._compute_adaptive_threshold_profile(
        material=MaterialType.SHELLAC,
        quality_mode="fast",
        restorability_score=-999.0,
    )
    assert 0.20 <= p["detection_threshold"] <= 3.50
    assert 0.08 <= p["yin_threshold"] <= 0.25
    assert 0.45 <= p["ml_confidence_threshold"] <= 0.85
    assert 0.18 <= p["min_confidence_for_correction"] <= 0.60


def test_low_restorability_lowers_confidence_guard() -> None:
    low = WowFlutterFix._compute_adaptive_threshold_profile(
        material=MaterialType.VINYL,
        quality_mode="quality",
        restorability_score=10.0,
    )
    high = WowFlutterFix._compute_adaptive_threshold_profile(
        material=MaterialType.VINYL,
        quality_mode="quality",
        restorability_score=90.0,
    )
    assert low["min_confidence_for_correction"] < high["min_confidence_for_correction"]


def test_fast_mode_is_more_conservative_than_quality() -> None:
    fast = WowFlutterFix._compute_adaptive_threshold_profile(
        material=MaterialType.VINYL,
        quality_mode="fast",
        restorability_score=60.0,
    )
    quality = WowFlutterFix._compute_adaptive_threshold_profile(
        material=MaterialType.VINYL,
        quality_mode="quality",
        restorability_score=60.0,
    )
    assert fast["detection_threshold"] > quality["detection_threshold"]
    assert fast["ml_confidence_threshold"] > quality["ml_confidence_threshold"]


def test_tape_keeps_lower_base_confidence_guard() -> None:
    tape = WowFlutterFix._compute_adaptive_threshold_profile(
        material=MaterialType.TAPE,
        quality_mode="balanced",
        restorability_score=50.0,
    )
    vinyl = WowFlutterFix._compute_adaptive_threshold_profile(
        material=MaterialType.VINYL,
        quality_mode="balanced",
        restorability_score=50.0,
    )
    assert tape["min_confidence_for_correction"] < vinyl["min_confidence_for_correction"]
