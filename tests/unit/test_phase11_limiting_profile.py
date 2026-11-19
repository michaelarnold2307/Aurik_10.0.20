import pytest

"""Unit tests for phase_11_limiting._compute_limiting_profile (§2.56)."""

from backend.core.phases.phase_11_limiting import LimitingPhase


@pytest.mark.unit
class TestLimitingProfile:
    def _p(self, material="vinyl", qm="balanced", rest=50.0):
        return LimitingPhase._compute_limiting_profile(material, qm, rest)

    def test_returns_required_keys(self):
        p = self._p()
        assert "lookahead_ms" in p

    def test_lookahead_bounds(self):
        for mat in ("vinyl", "shellac", "cd_digital", "unknown", "tape"):
            p = self._p(mat)
            assert 5.0 <= p["lookahead_ms"] <= 20.0, f"Out of bounds for {mat}"

    def test_quality_increases_lookahead(self):
        base = self._p("vinyl", "balanced")
        qual = self._p("vinyl", "quality")
        assert qual["lookahead_ms"] >= base["lookahead_ms"]

    def test_fast_decreases_lookahead(self):
        base = self._p("vinyl", "balanced")
        fast = self._p("vinyl", "fast")
        assert fast["lookahead_ms"] <= base["lookahead_ms"]

    def test_shellac_longer_than_cd(self):
        shellac = self._p("shellac")
        cd = self._p("cd_digital")
        assert shellac["lookahead_ms"] >= cd["lookahead_ms"]

    def test_low_rest_increases_lookahead(self):
        high = self._p("vinyl", "balanced", 80.0)
        low = self._p("vinyl", "balanced", 20.0)
        assert low["lookahead_ms"] >= high["lookahead_ms"]

    def test_none_quality_mode(self):
        p = self._p("vinyl", None)
        assert p["lookahead_ms"] > 0

    def test_unknown_material(self):
        p = self._p("unknown_xyz")
        assert 5.0 <= p["lookahead_ms"] <= 20.0
