import pytest

"""Unit tests for phase_09_crackle_removal._compute_crackle_removal_profile (§2.56)."""

from backend.core.phases.phase_09_crackle_removal import CrackleRemovalPhase


@pytest.mark.unit
class TestCrackleRemovalProfile:
    def _p(self, material="vinyl", qm="balanced", rest=50.0):
        return CrackleRemovalPhase._compute_crackle_removal_profile(material, qm, rest)

    def test_returns_required_keys(self):
        p = self._p()
        for key in ("stft_nperseg_model", "stft_nperseg_interp", "ar_order_texture"):
            assert key in p

    def test_ar_order_never_below_16(self):
        """§VERBOTEN: LPC/AR order must never be < 16."""
        for mat in ("vinyl", "shellac", "wax_cylinder", "cd_digital", "unknown"):
            for qm in ("fast", "balanced", "quality", "maximum"):
                for rest in (10.0, 50.0, 90.0):
                    p = self._p(mat, qm, rest)
                    assert int(p["ar_order_texture"]) >= 16, f"ar_order_texture < 16 for mat={mat} qm={qm} rest={rest}"

    def test_ar_order_ceiling(self):
        for mat in ("vinyl", "shellac"):
            p = self._p(mat)
            assert int(p["ar_order_texture"]) <= 32

    def test_stft_nperseg_model_power_of_two(self):
        for mat in ("vinyl", "cd_digital", "shellac"):
            p = self._p(mat)
            v = int(p["stft_nperseg_model"])
            assert 512 <= v <= 4096
            assert (v & (v - 1)) == 0

    def test_stft_nperseg_interp_in_range(self):
        for mat in ("vinyl", "cd_digital"):
            p = self._p(mat)
            v = int(p["stft_nperseg_interp"])
            assert 128 <= v <= 1024

    def test_quality_increases_nperseg_model(self):
        base = self._p("vinyl", "balanced")
        qual = self._p("vinyl", "quality")
        assert qual["stft_nperseg_model"] >= base["stft_nperseg_model"]

    def test_fast_decreases_nperseg_model(self):
        base = self._p("vinyl", "balanced")
        fast = self._p("vinyl", "fast")
        assert fast["stft_nperseg_model"] <= base["stft_nperseg_model"]

    def test_shellac_higher_ar_than_cd(self):
        shellac = self._p("shellac")
        cd = self._p("cd_digital")
        assert shellac["ar_order_texture"] >= cd["ar_order_texture"]

    def test_none_quality_mode(self):
        p = self._p("vinyl", None)
        assert int(p["ar_order_texture"]) >= 16

    def test_unknown_material(self):
        p = self._p("something_exotic_xyz")
        assert int(p["ar_order_texture"]) >= 16
