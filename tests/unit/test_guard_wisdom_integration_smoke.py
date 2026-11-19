"""Smoke test: GuardWisdom + CrossGuardCoordinator + GoalBudget integration.

Verifies the strength-modulation pattern used in _profiled_phase_call
during pipeline runs, without requiring the full pipeline.
"""

import pytest

from backend.core.goal_budget import GoalBudget
from backend.core.klang_guards import CrossGuardCoordinator, GuardWisdom


@pytest.mark.unit
class TestGuardWisdomIntegrationSmoke:
    def test_guard_wisdom_strength_drops_on_violations(self):
        """GuardWisdom.get_strength_mod() must drop below 1.0 after violations."""
        gw = GuardWisdom(material="cassette", genre="rock")
        assert gw.get_strength_mod() == 1.0

        # Record violations — strength must decrease
        gw.record("phase_03_denoise", "phase_quality", {"rms_ratio": 50.0}, verdict="violation")
        assert gw.get_strength_mod() == 0.85

        gw.record("phase_29_tape_hiss", "phase_quality", {"rms_ratio": 10.0}, verdict="violation")
        assert gw.get_strength_mod() == 0.70

        # OK records should not change strength
        gw.record("phase_01_click", "phase_quality", {"rms_ratio": 1.0}, verdict="ok")
        assert gw.get_strength_mod() == 0.70

    def test_strength_floor_respected(self):
        """Strength must never drop below 0.3."""
        gw = GuardWisdom(material="cassette", genre="rock")
        for i in range(50):
            gw.record(f"phase_{i}", "test", {}, verdict="violation")
        assert gw.get_strength_mod() >= 0.3

    def test_goal_budget_fraction_left(self):
        """GoalBudget.fraction_left must exist and return <= 1.0."""
        gb = GoalBudget(material_key="cassette")
        assert hasattr(gb, "fraction_left")
        # Use goals that actually exist in _DEFAULT_GOAL_BUDGET.
        for goal in ("waerme", "brillanz", "durchschlagskraft"):
            val = gb.fraction_left(goal)
            assert isinstance(val, float)
            assert 0.0 <= val <= 1.0

    def test_cross_guard_coordinator(self):
        """CrossGuardCoordinator must record and evaluate."""
        cg = CrossGuardCoordinator()
        # API: record(guard_name, phase_id, metrics)
        cg.record("dynamics_arc", "phase_36", {"lufs_drift": 3.0})
        result = cg.evaluate()
        assert "verdict" in result
        assert result["verdict"] in ("ok", "warning", "degraded")

    def test_guard_wisdom_in_context_pattern(self):
        """Verify the pattern used in _profiled_phase_call."""
        ctx = {}
        gw = GuardWisdom(material="tape", genre="")
        ctx["_guard_wisdom"] = gw

        # Simulate what _profiled_phase_call / unified_restorer_v3 does
        _gw = ctx.get("_guard_wisdom")
        assert _gw is not None
        sm = _gw.get_strength_mod()
        assert isinstance(sm, float)
        assert 0.3 <= sm <= 1.0

        # Simulate strength modulation
        kwargs = {"strength": 0.8}
        if sm < 1.0:
            kwargs["strength"] = float(kwargs["strength"]) * sm
        assert kwargs["strength"] == 0.8 * sm
