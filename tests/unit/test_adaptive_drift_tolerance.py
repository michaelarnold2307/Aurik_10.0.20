import pytest

"""
tests/unit/test_adaptive_drift_tolerance.py — §2.54 compute_adaptive_drift_tolerance() E2E

Tests that drift tolerance is computed adaptively from 4 parameters:
  restorability_score, material_type, defect_severity_mean, n_active_phases
  - Always negative, clamped [-0.30, -0.02]
  - More degraded material → looser (more negative)
  - Lower restorability → looser
  - Higher defect severity → looser
  - More active phases → looser
"""


def _get_fn():
    from backend.core.cumulative_interaction_guard import compute_adaptive_drift_tolerance

    return compute_adaptive_drift_tolerance


@pytest.mark.unit
class TestBasicProperties:
    """Fundamental properties of adaptive drift tolerance."""

    def test_import(self):
        fn = _get_fn()
        assert callable(fn)

    def test_returns_float(self):
        fn = _get_fn()
        result = fn()
        assert isinstance(result, float)

    def test_always_negative(self):
        fn = _get_fn()
        for mat in ["cd_digital", "vinyl", "shellac", "tape", "mp3_low", "wax_cylinder"]:
            result = fn(material_type=mat)
            assert result < 0.0, f"{mat}: {result}"

    def test_clamped_lower_bound(self):
        """Never looser than -0.30."""
        fn = _get_fn()
        result = fn(restorability_score=0.0, material_type="wax_cylinder", defect_severity_mean=1.0, n_active_phases=50)
        assert result >= -0.30

    def test_clamped_upper_bound(self):
        """Never tighter than -0.02."""
        fn = _get_fn()
        result = fn(restorability_score=100.0, material_type="cd_digital", defect_severity_mean=0.0, n_active_phases=1)
        assert result <= -0.02


class TestMaterialAdaptivity:
    """Different materials get different base tolerances."""

    def test_cd_tightest(self):
        fn = _get_fn()
        cd = fn(material_type="cd_digital", restorability_score=50.0)
        vinyl = fn(material_type="vinyl", restorability_score=50.0)
        assert cd > vinyl  # cd is less negative (tighter)

    def test_shellac_looser_than_vinyl(self):
        fn = _get_fn()
        shellac = fn(material_type="shellac", restorability_score=50.0)
        vinyl = fn(material_type="vinyl", restorability_score=50.0)
        assert shellac < vinyl  # shellac is more negative (looser)

    def test_wax_cylinder_loosest(self):
        fn = _get_fn()
        wax = fn(material_type="wax_cylinder", restorability_score=50.0)
        cd = fn(material_type="cd_digital", restorability_score=50.0)
        assert wax < cd  # wax is much more negative

    def test_unknown_material_has_default(self):
        fn = _get_fn()
        result = fn(material_type="unknown")
        assert -0.30 <= result <= -0.02

    def test_unknown_key_falls_back(self):
        fn = _get_fn()
        result = fn(material_type="nonexistent_format")
        assert -0.30 <= result <= -0.02

    def test_all_known_materials(self):
        """All 15 material types should produce valid results."""
        fn = _get_fn()
        materials = [
            "cd_digital",
            "dat",
            "minidisc",
            "mp3_high",
            "mp3_low",
            "cassette",
            "tape",
            "reel_tape",
            "vinyl",
            "shellac",
            "wax_cylinder",
            "wire_recording",
            "optical_film",
            "radio_broadcast",
            "unknown",
        ]
        for mat in materials:
            result = fn(material_type=mat)
            assert -0.30 <= result <= -0.02, f"{mat}: {result}"


class TestRestorabilityFactor:
    """Lower restorability → more tolerance (looser)."""

    def test_low_restorability_looser(self):
        fn = _get_fn()
        low = fn(restorability_score=10.0, material_type="vinyl")
        high = fn(restorability_score=90.0, material_type="vinyl")
        assert low < high  # low restorability is more negative

    def test_zero_restorability_maximum_multiplicator(self):
        fn = _get_fn()
        zero = fn(restorability_score=0.0, material_type="cd_digital")
        hundred = fn(restorability_score=100.0, material_type="cd_digital")
        assert zero < hundred

    def test_monotonic_decrease(self):
        """Tolerance should monotonically tighten as restorability increases."""
        fn = _get_fn()
        prev = fn(restorability_score=0.0, material_type="tape")
        for r in range(10, 101, 10):
            curr = fn(restorability_score=float(r), material_type="tape")
            assert curr >= prev, f"Non-monotonic at restorability={r}"
            prev = curr


class TestSeverityFactor:
    """Higher defect severity → more tolerance."""

    def test_high_severity_looser(self):
        fn = _get_fn()
        low_sev = fn(defect_severity_mean=0.1, material_type="vinyl")
        high_sev = fn(defect_severity_mean=0.9, material_type="vinyl")
        assert high_sev < low_sev  # more negative = looser

    def test_zero_severity_no_extra_tolerance(self):
        fn = _get_fn()
        zero = fn(defect_severity_mean=0.0, material_type="cd_digital")
        small = fn(defect_severity_mean=0.05, material_type="cd_digital")
        assert small <= zero  # at least equal or looser

    def test_severity_clamped_at_1(self):
        """Severity > 1.0 should be clamped."""
        fn = _get_fn()
        at_1 = fn(defect_severity_mean=1.0, material_type="vinyl")
        at_5 = fn(defect_severity_mean=5.0, material_type="vinyl")
        assert abs(at_1 - at_5) < 0.01  # Should be clamped to same value


class TestPhaseCountFactor:
    """More active phases → more tolerance."""

    def test_more_phases_looser(self):
        fn = _get_fn()
        few = fn(n_active_phases=5, material_type="tape")
        many = fn(n_active_phases=25, material_type="tape")
        assert many < few  # more negative = looser

    def test_below_5_phases_no_extra(self):
        """At ≤ 5 phases, phase factor should be 1.0."""
        fn = _get_fn()
        at_1 = fn(n_active_phases=1, material_type="cd_digital", restorability_score=50.0, defect_severity_mean=0.3)
        at_5 = fn(n_active_phases=5, material_type="cd_digital", restorability_score=50.0, defect_severity_mean=0.3)
        assert abs(at_1 - at_5) < 0.001


class TestCombinedScenarios:
    """Realistic combined parameter sets."""

    def test_clean_cd_minimal_tolerance(self):
        fn = _get_fn()
        result = fn(restorability_score=95.0, material_type="cd_digital", defect_severity_mean=0.05, n_active_phases=3)
        # Should be close to -0.02 (tightest)
        assert result > -0.05

    def test_heavily_degraded_shellac_wide_tolerance(self):
        fn = _get_fn()
        result = fn(restorability_score=15.0, material_type="shellac", defect_severity_mean=0.8, n_active_phases=20)
        # Should be close to -0.30 (loosest)
        assert result < -0.15

    def test_moderate_vinyl(self):
        fn = _get_fn()
        result = fn(restorability_score=50.0, material_type="vinyl", defect_severity_mean=0.4, n_active_phases=12)
        # Should be in the middle range
        assert -0.25 < result < -0.05

    def test_mp3_low_with_many_defects(self):
        fn = _get_fn()
        result = fn(restorability_score=40.0, material_type="mp3_low", defect_severity_mean=0.6, n_active_phases=15)
        assert -0.30 <= result <= -0.02
