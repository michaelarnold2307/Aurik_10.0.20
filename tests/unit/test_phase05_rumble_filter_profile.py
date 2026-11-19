import pytest

"""Unit tests for phase_05_rumble_filter._compute_rumble_filter_profile (§2.56)."""

from backend.core.phases.phase_05_rumble_filter import RumbleFilterPhase


@pytest.mark.unit
class TestRumbleFilterProfile:
    def _p(self, material="vinyl", qm="balanced", rest=50.0):
        return RumbleFilterPhase._compute_rumble_filter_profile(material, qm, rest)

    def test_returns_required_keys(self):
        p = self._p()
        assert "max_rms_drop_db" in p
        assert "onset_hop" in p
        assert "onset_fft" in p

    def test_max_rms_drop_bounds(self):
        for mat in ("vinyl", "shellac", "cd_digital", "unknown"):
            p = self._p(mat)
            assert 0.5 <= p["max_rms_drop_db"] <= 3.5, f"Out of bounds for {mat}"

    def test_onset_hop_bounds(self):
        for mat in ("vinyl", "wax_cylinder", "cd_digital"):
            p = self._p(mat)
            assert 128 <= int(p["onset_hop"]) <= 1024

    def test_onset_fft_power_of_two(self):
        for mat in ("vinyl", "cd_digital", "shellac"):
            p = self._p(mat)
            fft = int(p["onset_fft"])
            assert fft >= 512 and fft <= 4096
            assert (fft & (fft - 1)) == 0, f"onset_fft {fft} not power-of-2"

    def test_quality_increases_drop_and_reduces_hop(self):
        base = self._p("vinyl", "balanced")
        qual = self._p("vinyl", "quality")
        assert qual["max_rms_drop_db"] >= base["max_rms_drop_db"]
        assert qual["onset_hop"] <= base["onset_hop"]

    def test_fast_mode_reduces_max_drop(self):
        base = self._p("vinyl", "balanced")
        fast = self._p("vinyl", "fast")
        assert fast["max_rms_drop_db"] <= base["max_rms_drop_db"]

    def test_low_rest_increases_drop(self):
        high = self._p("vinyl", "balanced", 80.0)
        low = self._p("vinyl", "balanced", 20.0)
        assert low["max_rms_drop_db"] >= high["max_rms_drop_db"]

    def test_shellac_larger_drop_than_cd(self):
        shellac = self._p("shellac")
        cd = self._p("cd_digital")
        assert shellac["max_rms_drop_db"] >= cd["max_rms_drop_db"]

    def test_none_quality_mode(self):
        p = self._p("vinyl", None)
        assert p["max_rms_drop_db"] > 0

    def test_unknown_material(self):
        p = self._p("totally_unknown_xyz")
        assert p["max_rms_drop_db"] > 0
