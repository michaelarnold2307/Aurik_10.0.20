import pytest

"""Unit-Tests: InnerGrooveDistortionRepairPhase._compute_igd_profile() (§2.56)."""

import numpy as np

from backend.core.phases.phase_60_inner_groove_distortion_repair import InnerGrooveDistortionRepairPhase, apply


def _profile(material: str, qm: str = "balanced", rest: float = 50.0) -> dict:
    return InnerGrooveDistortionRepairPhase._compute_igd_profile(material, qm, rest)


@pytest.mark.unit
class TestIgdProfileMaterial:
    def test_vinyl_lower_min_than_cd(self):
        v = _profile("vinyl")["min_igd_score"]
        cd = _profile("cd_digital")["min_igd_score"]
        assert v < cd

    def test_bounds(self):
        p = _profile("unknown")
        assert 0.05 <= p["min_igd_score"] <= 0.25
        assert 4 <= p["n_segments"] <= 14


class TestIgdProfileQualityMode:
    def test_quality_more_sensitive_and_more_segments(self):
        base = _profile("vinyl", "balanced", 60.0)
        q = _profile("vinyl", "quality", 60.0)
        assert q["min_igd_score"] < base["min_igd_score"]
        assert q["n_segments"] > base["n_segments"]

    def test_fast_less_sensitive_and_fewer_segments(self):
        base = _profile("vinyl", "balanced", 60.0)
        fast = _profile("vinyl", "fast", 60.0)
        assert fast["min_igd_score"] > base["min_igd_score"]
        assert fast["n_segments"] < base["n_segments"]


class TestIgdProfileRestorability:
    def test_low_restorability_more_sensitive(self):
        high_rest = _profile("vinyl", "balanced", 80.0)
        low_rest = _profile("vinyl", "balanced", 20.0)
        assert low_rest["min_igd_score"] < high_rest["min_igd_score"]
        assert low_rest["n_segments"] >= high_rest["n_segments"]


class TestIgdProfileIntegration:
    def test_metadata_contains_profile(self):
        phase = InnerGrooveDistortionRepairPhase()
        audio = np.random.uniform(-0.2, 0.2, 48000).astype(np.float32)

        result = phase.process(
            audio,
            sample_rate=48000,
            strength=0.5,
            quality_mode="quality",
            restorability_score=30.0,
            material_type="vinyl",
            defect_scores={"inner_groove_distortion": 0.2},
        )

        assert result.success
        assert "igd_profile" in result.metadata
        assert "min_igd_score" in result.metadata
        assert "n_segments" in result.metadata
        assert 0.05 <= result.metadata["min_igd_score"] <= 0.25
        assert 4 <= result.metadata["n_segments"] <= 14


class TestIgdSegmentOracle:
    def test_distorted_inner_segment_is_processed_more_than_clean_outer_segment(self):
        sr = 48000
        t = np.arange(sr * 2, dtype=np.float32) / float(sr)
        # §7.4c-L3 (2026-09-28): 220 Hz statt 440 Hz — das IGD-Band beginnt
        # seit dem Bass-Summton-Fix bei 400 Hz; ein 440-Hz-Grundton würde
        # im Außen-Segment als IGD-Energie zählen und den
        # Positions-Kontrast des Oracle verfälschen.
        audio = (0.20 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
        audio[sr:] += (0.08 * np.sin(2.0 * np.pi * 4200.0 * t[sr:])).astype(np.float32)

        out = apply(
            audio,
            sr,
            strength=0.8,
            defect_scores={"inner_groove_distortion": 0.8},
            n_segments=4,
        )

        outer_delta = float(np.sqrt(np.mean((out[: sr // 2] - audio[: sr // 2]) ** 2)))
        inner_delta = float(np.sqrt(np.mean((out[sr:] - audio[sr:]) ** 2)))
        assert inner_delta > outer_delta * 2.0

    def test_protected_zone_caps_inner_groove_processing(self):
        sr = 48000
        t = np.arange(sr * 2, dtype=np.float32) / float(sr)
        audio = (0.20 * np.sin(2.0 * np.pi * 440.0 * t) + 0.08 * np.sin(2.0 * np.pi * 4200.0 * t)).astype(np.float32)

        free = apply(
            audio,
            sr,
            strength=0.8,
            defect_scores={"inner_groove_distortion": 0.8},
            n_segments=4,
        )
        capped = apply(
            audio,
            sr,
            strength=0.8,
            defect_scores={"inner_groove_distortion": 0.8},
            n_segments=4,
            protected_zones=[(1.0, 2.0, 0.20)],
        )

        free_delta = float(np.sqrt(np.mean((free[sr:] - audio[sr:]) ** 2)))
        capped_delta = float(np.sqrt(np.mean((capped[sr:] - audio[sr:]) ** 2)))
        assert capped_delta < free_delta * 0.65
