import pytest

"""Unit tests for phase_32_mono_to_stereo._compute_mono_to_stereo_profile (§2.56)."""

from backend.core.phases.phase_32_mono_to_stereo import MonoToStereoPhaseV2


@pytest.mark.unit
class TestMonoToStereoProfile:
    def _p(self, material="vinyl", qm="balanced", rest=50.0):
        return MonoToStereoPhaseV2._compute_mono_to_stereo_profile(material, qm, rest)

    def test_returns_required_keys(self):
        p = self._p()
        assert "mono_correlation_threshold" in p

    def test_values_in_bounds(self):
        for mat in ("vinyl", "shellac", "cd_digital", "tape", "unknown"):
            for qm in ("fast", "balanced", "quality", "maximum"):
                p = self._p(mat, qm)
                assert 0.80 <= p["mono_correlation_threshold"] <= 0.97, (
                    f"threshold={p['mono_correlation_threshold']} mat={mat} qm={qm}"
                )

    def test_shellac_lower_than_cd(self):
        shellac = self._p("shellac", "balanced")
        cd = self._p("cd_digital", "balanced")
        assert shellac["mono_correlation_threshold"] <= cd["mono_correlation_threshold"]

    def test_quality_increases_threshold(self):
        base = self._p("vinyl", "balanced")
        qual = self._p("vinyl", "quality")
        assert qual["mono_correlation_threshold"] >= base["mono_correlation_threshold"]

    def test_low_rest_decreases_threshold(self):
        high = self._p("vinyl", "balanced", 80.0)
        low = self._p("vinyl", "balanced", 20.0)
        assert low["mono_correlation_threshold"] <= high["mono_correlation_threshold"]

    def test_none_quality_mode(self):
        p = self._p("vinyl", None)
        assert 0.80 <= p["mono_correlation_threshold"] <= 0.97

    def test_unknown_material(self):
        p = self._p("totally_unknown_xyz")
        assert 0.80 <= p["mono_correlation_threshold"] <= 0.97
