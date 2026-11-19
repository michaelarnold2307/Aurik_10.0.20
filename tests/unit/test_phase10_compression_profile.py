import pytest

"""Unit tests for phase_10_compression._compute_compression_profile (§2.56)."""

from backend.core.phases.phase_10_compression import CompressionPhase


@pytest.mark.unit
class TestCompressionProfile:
    def _p(self, material="vinyl", qm="balanced", rest=50.0):
        return CompressionPhase._compute_compression_profile(material, qm, rest)

    def test_returns_required_keys(self):
        p = self._p()
        for key in ("lookahead_ms", "rms_window_ms", "peak_window_ms"):
            assert key in p

    def test_lookahead_bounds(self):
        for mat in ("vinyl", "shellac", "cd_digital", "unknown"):
            p = self._p(mat)
            assert 2.0 <= p["lookahead_ms"] <= 10.0

    def test_rms_window_bounds(self):
        for mat in ("vinyl", "shellac", "cd_digital"):
            p = self._p(mat)
            assert 5.0 <= p["rms_window_ms"] <= 20.0

    def test_peak_window_bounds(self):
        for mat in ("vinyl", "cd_digital"):
            p = self._p(mat)
            assert 2.0 <= p["peak_window_ms"] <= 10.0

    def test_quality_increases_windows(self):
        base = self._p("vinyl", "balanced")
        qual = self._p("vinyl", "quality")
        assert qual["rms_window_ms"] >= base["rms_window_ms"]
        assert qual["peak_window_ms"] >= base["peak_window_ms"]

    def test_fast_reduces_windows(self):
        base = self._p("vinyl", "balanced")
        fast = self._p("vinyl", "fast")
        assert fast["rms_window_ms"] <= base["rms_window_ms"]

    def test_shellac_longer_lookahead_than_cd(self):
        shellac = self._p("shellac")
        cd = self._p("cd_digital")
        assert shellac["lookahead_ms"] >= cd["lookahead_ms"]

    def test_none_quality_mode(self):
        p = self._p("vinyl", None)
        assert p["lookahead_ms"] > 0

    def test_unknown_material(self):
        p = self._p("super_exotic_xyz")
        assert all(v > 0 for v in p.values())

    def test_low_rest_increases_lookahead(self):
        high = self._p("vinyl", "balanced", 80.0)
        low = self._p("vinyl", "balanced", 20.0)
        assert low["lookahead_ms"] >= high["lookahead_ms"]
