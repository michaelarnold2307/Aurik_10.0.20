import pytest

"""Unit tests for phase_04_eq_correction._compute_eq_correction_profile (§2.56)."""

from backend.core.phases.phase_04_eq_correction import EQCorrectionPhase


@pytest.mark.unit
class TestEqCorrectionProfile:
    def _p(self, material="vinyl", qm="balanced", rest=50.0):
        return EQCorrectionPhase._compute_eq_correction_profile(material, qm, rest)

    def test_returns_required_keys(self):
        p = self._p()
        assert "analysis_fft_size" in p

    def test_fft_size_power_of_two(self):
        for mat in ("vinyl", "shellac", "wax_cylinder", "cd_digital", "unknown"):
            for qm in ("fast", "balanced", "quality"):
                p = self._p(mat, qm)
                fft = int(p["analysis_fft_size"])
                assert fft > 0 and (fft & (fft - 1)) == 0, f"fft_size={fft} not power-of-2 for mat={mat} qm={qm}"

    def test_fft_size_in_bounds(self):
        for mat in ("vinyl", "shellac", "cd_digital", "tape", "unknown"):
            p = self._p(mat)
            assert 1024 <= int(p["analysis_fft_size"]) <= 8192

    def test_quality_increases_fft(self):
        base = self._p("vinyl", "balanced")
        qual = self._p("vinyl", "quality")
        assert qual["analysis_fft_size"] >= base["analysis_fft_size"]

    def test_fast_decreases_fft(self):
        base = self._p("vinyl", "balanced")
        fast = self._p("vinyl", "fast")
        assert fast["analysis_fft_size"] <= base["analysis_fft_size"]

    def test_cd_larger_fft_than_wax(self):
        cd = self._p("cd_digital")
        wax = self._p("wax_cylinder")
        assert cd["analysis_fft_size"] >= wax["analysis_fft_size"]

    def test_none_quality_mode(self):
        p = self._p("vinyl", None)
        fft = int(p["analysis_fft_size"])
        assert fft > 0 and (fft & (fft - 1)) == 0

    def test_unknown_material(self):
        p = self._p("super_exotic_xyz")
        assert int(p["analysis_fft_size"]) > 0

    def test_restorability_has_no_effect_on_fft(self):
        """FFT size is not restorability-dependent (quality and material drive it)."""
        low = self._p("vinyl", "balanced", 10.0)
        high = self._p("vinyl", "balanced", 90.0)
        # Both should produce the same fft size since restorability doesn't affect it
        assert low["analysis_fft_size"] == high["analysis_fft_size"]
