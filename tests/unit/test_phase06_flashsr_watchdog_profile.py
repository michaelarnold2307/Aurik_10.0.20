from __future__ import annotations

import pytest

from backend.core.phases.phase_06_frequency_restoration import FrequencyRestorationPhase


@pytest.mark.unit
def test_watchdog_profile_keys_and_bounds() -> None:
    p = FrequencyRestorationPhase._compute_flashsr_watchdog_profile(
        quality_mode="quality",
        material_type="vinyl",
        restorability_score=55.0,
        audio_duration_s=120.0,
        default_min_duration_s=10.0,
    )

    assert set(p.keys()) == {
        "min_duration_s",
        "timeout_seconds",
        "timeout_mult",
        "timeout_min",
        "timeout_max",
    }
    assert 4.0 <= p["min_duration_s"] <= 20.0
    assert 2.0 <= p["timeout_mult"] <= 14.0
    assert p["timeout_min"] <= p["timeout_seconds"] <= p["timeout_max"]


def test_fast_mode_uses_shorter_timeout_and_higher_min_duration() -> None:
    fast = FrequencyRestorationPhase._compute_flashsr_watchdog_profile(
        quality_mode="fast",
        material_type="cd_digital",
        restorability_score=60.0,
        audio_duration_s=60.0,
        default_min_duration_s=10.0,
    )
    quality = FrequencyRestorationPhase._compute_flashsr_watchdog_profile(
        quality_mode="quality",
        material_type="cd_digital",
        restorability_score=60.0,
        audio_duration_s=60.0,
        default_min_duration_s=10.0,
    )

    assert fast["timeout_seconds"] < quality["timeout_seconds"]
    assert fast["min_duration_s"] > quality["min_duration_s"]


def test_low_restorability_relaxes_min_duration() -> None:
    low = FrequencyRestorationPhase._compute_flashsr_watchdog_profile(
        quality_mode="balanced",
        material_type="vinyl",
        restorability_score=10.0,
        audio_duration_s=30.0,
        default_min_duration_s=10.0,
    )
    high = FrequencyRestorationPhase._compute_flashsr_watchdog_profile(
        quality_mode="balanced",
        material_type="vinyl",
        restorability_score=90.0,
        audio_duration_s=30.0,
        default_min_duration_s=10.0,
    )

    assert low["min_duration_s"] < high["min_duration_s"]


def test_analog_material_gets_more_runtime_than_digital() -> None:
    analog = FrequencyRestorationPhase._compute_flashsr_watchdog_profile(
        quality_mode="quality",
        material_type="shellac",
        restorability_score=60.0,
        audio_duration_s=90.0,
        default_min_duration_s=10.0,
    )
    digital = FrequencyRestorationPhase._compute_flashsr_watchdog_profile(
        quality_mode="quality",
        material_type="cd_digital",
        restorability_score=60.0,
        audio_duration_s=90.0,
        default_min_duration_s=10.0,
    )

    assert analog["timeout_seconds"] >= digital["timeout_seconds"]
