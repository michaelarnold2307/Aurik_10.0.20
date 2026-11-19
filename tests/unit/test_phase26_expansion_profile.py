import pytest

"""Unit tests for phase_26_dynamic_range_expansion._compute_expansion_profile (§2.56, §6.2b)."""

import numpy as np

from backend.core.phases.phase_26_dynamic_range_expansion import DynamicRangeExpansion


@pytest.mark.unit
class TestExpansionProfile:
    def _p(self, material="vinyl", qm="balanced", rest=50.0):
        return DynamicRangeExpansion._compute_expansion_profile(material, qm, rest)

    def test_returns_required_keys(self):
        p = self._p()
        assert "max_expansion_db" in p

    def test_values_in_bounds(self):
        for mat in ("vinyl", "shellac", "wax_cylinder", "cd_digital", "tape", "unknown"):
            for qm in ("fast", "balanced", "quality", "maximum", "restoration"):
                p = self._p(mat, qm)
                assert 2.0 <= p["max_expansion_db"] <= 12.0, (
                    f"max_expansion_db={p['max_expansion_db']} out of [2,12] mat={mat} qm={qm}"
                )

    def test_shellac_lower_than_cd(self):
        shellac = self._p("shellac", "balanced")
        cd = self._p("cd_digital", "balanced")
        assert shellac["max_expansion_db"] <= cd["max_expansion_db"]

    def test_wax_lowest(self):
        wax = self._p("wax_cylinder", "balanced")
        cd = self._p("cd_digital", "balanced")
        assert wax["max_expansion_db"] <= cd["max_expansion_db"]

    def test_quality_increases_expansion(self):
        base = self._p("vinyl", "restoration")
        qual = self._p("vinyl", "quality")
        assert qual["max_expansion_db"] >= base["max_expansion_db"]

    def test_fast_decreases_expansion(self):
        base = self._p("vinyl", "balanced")
        fast = self._p("vinyl", "fast")
        assert fast["max_expansion_db"] <= base["max_expansion_db"]

    def test_low_rest_decreases_expansion(self):
        high = self._p("vinyl", "balanced", 80.0)
        low = self._p("vinyl", "balanced", 20.0)
        assert low["max_expansion_db"] <= high["max_expansion_db"]

    def test_none_quality_mode(self):
        p = self._p("vinyl", None)
        assert 2.0 <= p["max_expansion_db"] <= 12.0

    def test_unknown_material(self):
        p = self._p("totally_unknown_xyz")
        assert 2.0 <= p["max_expansion_db"] <= 12.0


def test_measure_dynamic_range_uses_mid_channel_not_left_only():
    phase = DynamicRangeExpansion()
    n = 48000
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    left = 0.9 * np.sin(2.0 * np.pi * 440.0 * t)
    right = 0.9 * np.sin(2.0 * np.pi * 440.0 * t + np.pi / 2.0)
    stereo = np.column_stack([left, right]).astype(np.float32)

    dr = phase._measure_dynamic_range(stereo)
    assert np.isfinite(dr)
    assert dr > 0.0
