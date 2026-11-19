import pytest

"""Unit tests for phase_33_stereo_width_limiter._compute_width_limiter_profile (§2.56)."""

from backend.core.phases.phase_33_stereo_width_limiter import StereoWidthLimiterPhaseV2


@pytest.mark.unit
class TestWidthLimiterProfile:
    def _p(self, material="vinyl", qm="balanced", rest=50.0):
        return StereoWidthLimiterPhaseV2._compute_width_limiter_profile(material, qm, rest)

    def test_returns_required_keys(self):
        p = self._p()
        keys = {"attack_ms", "release_ms", "transient_threshold_percentile", "transient_width_preservation"}
        assert keys.issubset(p)

    def test_all_values_in_bounds(self):
        for mat in ("vinyl", "shellac", "cd_digital", "tape", "unknown"):
            for qm in ("fast", "balanced", "quality", "maximum"):
                p = self._p(mat, qm)
                assert 5.0 <= p["attack_ms"] <= 20.0
                assert 50.0 <= p["release_ms"] <= 200.0
                assert 70.0 <= p["transient_threshold_percentile"] <= 95.0
                assert 0.5 <= p["transient_width_preservation"] <= 0.9

    def test_shellac_slower_than_cd(self):
        shellac = self._p("shellac", "balanced")
        cd = self._p("cd_digital", "balanced")
        assert shellac["attack_ms"] >= cd["attack_ms"]
        assert shellac["release_ms"] >= cd["release_ms"]

    def test_shellac_more_preservation_than_cd(self):
        shellac = self._p("shellac", "balanced")
        cd = self._p("cd_digital", "balanced")
        assert shellac["transient_width_preservation"] >= cd["transient_width_preservation"]

    def test_quality_increases_timing(self):
        base = self._p("vinyl", "balanced")
        qual = self._p("vinyl", "quality")
        assert qual["attack_ms"] >= base["attack_ms"]
        assert qual["release_ms"] >= base["release_ms"]

    def test_fast_decreases_timing(self):
        base = self._p("vinyl", "balanced")
        fast = self._p("vinyl", "fast")
        assert fast["attack_ms"] <= base["attack_ms"]
        assert fast["release_ms"] <= base["release_ms"]

    def test_low_rest_lowers_threshold_percentile(self):
        high = self._p("vinyl", "balanced", 80.0)
        low = self._p("vinyl", "balanced", 20.0)
        assert low["transient_threshold_percentile"] <= high["transient_threshold_percentile"]

    def test_none_quality_mode(self):
        p = self._p("vinyl", None)
        assert 5.0 <= p["attack_ms"] <= 20.0

    def test_unknown_material(self):
        p = self._p("totally_unknown_xyz")
        assert 5.0 <= p["attack_ms"] <= 20.0
