import pytest

"""Unit tests for phase_08_transient_preservation._compute_transient_profile (§2.56)."""

from backend.core.phases.phase_08_transient_preservation import TransientPreservationPhase


@pytest.mark.unit
class TestTransientProfile:
    def _p(self, material="vinyl", qm="balanced", rest=50.0):
        return TransientPreservationPhase._compute_transient_profile(material, qm, rest)

    def test_returns_required_keys(self):
        p = self._p()
        for key in ("onset_hop", "onset_fft", "superflux_w"):
            assert key in p

    def test_onset_hop_in_range(self):
        for mat in ("vinyl", "shellac", "cd_digital", "unknown"):
            p = self._p(mat)
            assert 128 <= int(p["onset_hop"]) <= 1024

    def test_onset_fft_power_of_two(self):
        for mat in ("vinyl", "cd_digital", "shellac"):
            p = self._p(mat)
            fft = int(p["onset_fft"])
            assert 512 <= fft <= 4096
            assert (fft & (fft - 1)) == 0

    def test_superflux_w_in_range(self):
        for mat in ("vinyl", "shellac", "cd_digital"):
            p = self._p(mat)
            assert 2 <= int(p["superflux_w"]) <= 5

    def test_quality_increases_fft(self):
        base = self._p("vinyl", "balanced")
        qual = self._p("vinyl", "quality")
        assert qual["onset_fft"] >= base["onset_fft"]

    def test_fast_reduces_fft(self):
        base = self._p("vinyl", "balanced")
        fast = self._p("vinyl", "fast")
        assert fast["onset_fft"] <= base["onset_fft"]

    def test_low_rest_reduces_hop(self):
        high = self._p("vinyl", "balanced", 80.0)
        low = self._p("vinyl", "balanced", 20.0)
        assert low["onset_hop"] <= high["onset_hop"]

    def test_none_quality_mode(self):
        p = self._p("vinyl", None)
        assert p["onset_hop"] > 0

    def test_unknown_material(self):
        p = self._p("something_exotic_xyz")
        assert all(v > 0 for v in p.values())
