import pytest

"""Unit tests for phase_19_de_esser._compute_de_esser_profile (§2.56)."""

import numpy as np

from backend.core.phases.phase_19_de_esser import DeEsserPhase


@pytest.mark.unit
class TestDeEsserProfile:
    def _p(self, material="vinyl", qm="balanced", rest=50.0):
        return DeEsserPhase._compute_de_esser_profile(material, qm, rest)

    def test_returns_required_keys(self):
        p = self._p()
        assert "lookahead_ms" in p

    def test_lookahead_in_bounds(self):
        for mat in ("vinyl", "shellac", "wax_cylinder", "cd_digital", "tape", "unknown"):
            for qm in ("fast", "balanced", "quality", "maximum"):
                p = self._p(mat, qm)
                assert 2.0 <= p["lookahead_ms"] <= 10.0, (
                    f"lookahead={p['lookahead_ms']} out of [2,10] for mat={mat} qm={qm}"
                )

    def test_quality_increases_lookahead(self):
        base = self._p("vinyl", "balanced")
        qual = self._p("vinyl", "quality")
        assert qual["lookahead_ms"] >= base["lookahead_ms"]

    def test_fast_decreases_lookahead(self):
        base = self._p("vinyl", "balanced")
        fast = self._p("vinyl", "fast")
        assert fast["lookahead_ms"] <= base["lookahead_ms"]

    def test_shellac_larger_lookahead_than_cd(self):
        shellac = self._p("shellac", "balanced")
        cd = self._p("cd_digital", "balanced")
        assert shellac["lookahead_ms"] >= cd["lookahead_ms"]

    def test_wax_cylinder_highest_base(self):
        wax = self._p("wax_cylinder", "balanced")
        cd = self._p("cd_digital", "balanced")
        assert wax["lookahead_ms"] >= cd["lookahead_ms"]

    def test_low_rest_increases_lookahead(self):
        high = self._p("vinyl", "balanced", 80.0)
        low = self._p("vinyl", "balanced", 20.0)
        assert low["lookahead_ms"] >= high["lookahead_ms"]

    def test_none_quality_mode(self):
        p = self._p("vinyl", None)
        assert 2.0 <= p["lookahead_ms"] <= 10.0

    def test_unknown_material(self):
        p = self._p("totally_unknown_xyz")
        assert 2.0 <= p["lookahead_ms"] <= 10.0


class TestDeEsserLocalityProfile:
    def test_sibilance_profile_is_event_strength_adaptive(self):
        sr = 48000
        profile, coverage = DeEsserPhase._build_sibilance_locality_profile(
            n_samples=sr * 2,
            sample_rate=sr,
            defect_locations={"sibilance": [(0.20, 0.38)], "vocal_harshness": [(1.20, 1.38)]},
            event_metadata={
                "sibilance": {"severity": 0.95, "confidence": 0.95},
                "vocal_harshness": {"severity": 0.30, "confidence": 0.65},
            },
        )

        assert profile.shape == (sr * 2,)
        assert 0.0 < coverage < 0.35
        strong_region = float(np.mean(profile[int(0.24 * sr) : int(0.34 * sr)]))
        mild_region = float(np.mean(profile[int(1.24 * sr) : int(1.34 * sr)]))
        clean_region = float(np.mean(profile[int(0.60 * sr) : int(0.90 * sr)]))
        assert strong_region > mild_region * 1.5
        assert clean_region < 0.04

    def test_vibrato_zone_caps_sibilance_profile(self):
        sr = 48000
        free, _ = DeEsserPhase._build_sibilance_locality_profile(
            n_samples=sr * 2,
            sample_rate=sr,
            defect_locations={"sibilance": [(1.10, 1.35)]},
            event_metadata={"sibilance": {"severity": 0.95, "confidence": 0.95}},
        )
        capped, _ = DeEsserPhase._build_sibilance_locality_profile(
            n_samples=sr * 2,
            sample_rate=sr,
            defect_locations={"sibilance": [(1.10, 1.35)]},
            event_metadata={"sibilance": {"severity": 0.95, "confidence": 0.95}},
            protected_zones=[(1.05, 1.40, 0.20)],
        )

        free_strength = float(np.mean(free[int(1.16 * sr) : int(1.30 * sr)]))
        capped_strength = float(np.mean(capped[int(1.16 * sr) : int(1.30 * sr)]))
        assert capped_strength <= 0.21
        assert capped_strength < free_strength * 0.35
