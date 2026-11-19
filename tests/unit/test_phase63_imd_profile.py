import pytest

"""Unit-Tests: IntermodulationReductionPhase._compute_imd_profile() (§2.56)."""

import numpy as np

from backend.core.phases.phase_63_intermodulation_reduction import IntermodulationReductionPhase


def _profile(material: str, qm: str = "balanced", rest: float = 50.0) -> dict:
    return IntermodulationReductionPhase._compute_imd_profile(material, qm, rest)


@pytest.mark.unit
def test_vinyl_less_sensitive_than_cd():
    vinyl = _profile("vinyl")
    cd = _profile("cd_digital")
    assert vinyl["min_imd_score"] > cd["min_imd_score"]


def test_quality_adjustment():
    base = _profile("vinyl", "balanced", 60.0)
    q = _profile("vinyl", "quality", 60.0)
    assert q["min_imd_score"] < base["min_imd_score"]
    assert q["notch_width_hz"] < base["notch_width_hz"]


def test_fast_adjustment():
    base = _profile("vinyl", "balanced", 60.0)
    fast = _profile("vinyl", "fast", 60.0)
    assert fast["min_imd_score"] > base["min_imd_score"]
    assert fast["notch_width_hz"] > base["notch_width_hz"]


def test_low_restorability_adjustment():
    high_rest = _profile("vinyl", "balanced", 80.0)
    low_rest = _profile("vinyl", "balanced", 20.0)
    assert low_rest["min_imd_score"] < high_rest["min_imd_score"]


def test_profile_bounds():
    for material in ["shellac", "vinyl", "cd_digital", "unknown"]:
        p = _profile(material, "maximum", 10.0)
        assert 0.05 <= p["min_imd_score"] <= 0.30
        assert 20.0 <= p["notch_width_hz"] <= 120.0


def test_process_metadata_contains_profile():
    phase = IntermodulationReductionPhase()
    audio = np.random.uniform(-0.2, 0.2, 48000).astype(np.float32)

    result = phase.process(
        audio,
        sample_rate=48000,
        quality_mode="quality",
        restorability_score=35.0,
        material_type="vinyl",
        defect_scores={"intermodulation_distortion": 0.3},
        strength=0.5,
    )

    assert result.success
    assert "imd_profile" in result.metadata
    assert "imd_threshold" in result.metadata
    assert "notch_width_hz" in result.metadata


def test_locality_profile_is_event_strength_adaptive():
    sr = 48000
    profile, coverage = IntermodulationReductionPhase._build_locality_profile(
        n_samples=sr * 3,
        sample_rate=sr,
        defect_locations={"intermodulation_distortion": [(0.30, 0.90)], "digital_artifacts": [(1.70, 2.30)]},
        event_metadata={
            "intermodulation_distortion": {"severity": 0.95, "confidence": 0.95},
            "digital_artifacts": {"severity": 0.35, "confidence": 0.65},
        },
    )

    assert profile.shape == (sr * 3,)
    assert 0.0 < coverage < 0.50
    strong_region = float(np.mean(profile[int(0.45 * sr) : int(0.75 * sr)]))
    mild_region = float(np.mean(profile[int(1.85 * sr) : int(2.15 * sr)]))
    clean_region = float(np.mean(profile[int(1.10 * sr) : int(1.35 * sr)]))
    assert strong_region > mild_region * 1.5
    assert clean_region < 0.04


def test_vibrato_zone_caps_imd_profile():
    sr = 48000
    free, _ = IntermodulationReductionPhase._build_locality_profile(
        n_samples=sr * 3,
        sample_rate=sr,
        defect_locations={"intermodulation_distortion": [(1.20, 1.80)]},
        event_metadata={"intermodulation_distortion": {"severity": 0.95, "confidence": 0.95}},
    )
    capped, _ = IntermodulationReductionPhase._build_locality_profile(
        n_samples=sr * 3,
        sample_rate=sr,
        defect_locations={"intermodulation_distortion": [(1.20, 1.80)]},
        event_metadata={"intermodulation_distortion": {"severity": 0.95, "confidence": 0.95}},
        protected_zones=[(1.15, 1.85, 0.20)],
    )

    free_strength = float(np.mean(free[int(1.35 * sr) : int(1.65 * sr)]))
    capped_strength = float(np.mean(capped[int(1.35 * sr) : int(1.65 * sr)]))
    assert capped_strength <= 0.21
    assert capped_strength < free_strength * 0.55
