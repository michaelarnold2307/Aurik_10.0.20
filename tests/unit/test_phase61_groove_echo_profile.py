import pytest

"""Unit-Tests: GrooveEchoCancellationPhase._compute_groove_echo_profile() (§2.56)."""

import numpy as np

from backend.core.phases.phase_61_groove_echo_cancellation import GrooveEchoCancellationPhase


def _profile(material: str, qm: str | None = "balanced", rest: float = 50.0) -> dict:
    quality_mode: str = qm or "balanced"
    return GrooveEchoCancellationPhase._compute_groove_echo_profile(material, quality_mode, rest)


@pytest.mark.unit
def test_vinyl_more_sensitive_than_cd():
    vinyl = _profile("vinyl")
    cd = _profile("cd_digital")
    assert vinyl["min_groove_echo_score"] < cd["min_groove_echo_score"]


def test_quality_adjustment():
    base = _profile("vinyl", "balanced", 60.0)
    q = _profile("vinyl", "quality", 60.0)
    assert q["min_groove_echo_score"] < base["min_groove_echo_score"]
    assert q["spectral_subtraction_floor_db"] < base["spectral_subtraction_floor_db"]


def test_fast_adjustment():
    base = _profile("vinyl", "balanced", 60.0)
    fast = _profile("vinyl", "fast", 60.0)
    assert fast["min_groove_echo_score"] > base["min_groove_echo_score"]
    assert fast["spectral_subtraction_floor_db"] > base["spectral_subtraction_floor_db"]


def test_low_restorability_adjustment():
    high_rest = _profile("vinyl", "balanced", 80.0)
    low_rest = _profile("vinyl", "balanced", 20.0)
    assert low_rest["min_groove_echo_score"] < high_rest["min_groove_echo_score"]


def test_profile_bounds():
    for material in ["vinyl", "shellac", "cd_digital", "unknown"]:
        for qm in ["balanced", "quality", "maximum", "fast", None]:
            p = _profile(material, qm, 35.0)
            assert 0.05 <= p["min_groove_echo_score"] <= 0.25
            assert -60.0 <= p["spectral_subtraction_floor_db"] <= -20.0


def test_process_metadata_contains_profile():
    phase = GrooveEchoCancellationPhase()
    audio = np.random.uniform(-0.2, 0.2, 48000).astype(np.float32)

    result = phase.process(
        audio,
        sample_rate=48000,
        quality_mode="quality",
        restorability_score=35.0,
        material_type="vinyl",
        defect_scores={"groove_echo": 0.2},
        strength=0.5,
    )

    assert result.success
    assert "groove_echo_profile" in result.metadata
    assert "min_groove_echo_score" in result.metadata
    assert "spectral_subtraction_floor_db" in result.metadata


def test_locality_profile_is_event_adaptive():
    profile, coverage = GrooveEchoCancellationPhase._build_locality_profile(
        n_samples=48000 * 4,
        sample_rate=48000,
        defect_locations={"groove_echo": [(0.80, 1.20), (2.60, 2.90)]},
        defect_event_metadata={"groove_echo": {"severity": 0.95, "confidence": 0.95}},
    )

    assert profile.shape == (48000 * 4,)
    assert 0.0 < coverage < 0.70
    strong_region = float(np.mean(profile[int(0.90 * 48000) : int(1.10 * 48000)]))
    later_region = float(np.mean(profile[int(2.65 * 48000) : int(2.85 * 48000)]))
    clean_region = float(np.mean(profile[int(1.80 * 48000) : int(2.10 * 48000)]))
    assert strong_region > 0.40
    assert later_region > 0.30
    assert clean_region < 0.10


def test_vibrato_zone_caps_locality_profile():
    free, _ = GrooveEchoCancellationPhase._build_locality_profile(
        n_samples=48000 * 3,
        sample_rate=48000,
        defect_locations={"groove_echo": [(1.20, 1.60)]},
        defect_event_metadata={"groove_echo": {"severity": 0.95, "confidence": 0.95}},
    )
    capped, _ = GrooveEchoCancellationPhase._build_locality_profile(
        n_samples=48000 * 3,
        sample_rate=48000,
        defect_locations={"groove_echo": [(1.20, 1.60)]},
        defect_event_metadata={"groove_echo": {"severity": 0.95, "confidence": 0.95}},
        protected_zones=[(1.10, 1.70, 0.20)],
    )

    free_strength = float(np.mean(free[int(1.25 * 48000) : int(1.55 * 48000)]))
    capped_strength = float(np.mean(capped[int(1.25 * 48000) : int(1.55 * 48000)]))
    assert capped_strength <= 0.21
    assert capped_strength < free_strength * 0.50
