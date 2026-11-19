from __future__ import annotations

import pytest

"""Unit-Tests fuer Spec-Upgrade Gate (§2.81 / §MG-UPG)."""


from backend.core.song_goal_importance import ALL_GOAL_NAMES
from backend.core.spec_upgrade_gate import evaluate_spec_upgrade_candidate


def _scores(default: float = 0.8) -> dict[str, float]:
    return dict.fromkeys(ALL_GOAL_NAMES, default)


@pytest.mark.unit
class TestSpecUpgradeGate:
    """Prueft Promotion nur bei Safety + 15-Goal-Nicht-Regression."""

    def test_promote_when_one_goal_improves_and_rest_not_worse(self):
        baseline = _scores(0.8)
        candidate = _scores(0.8)
        candidate["natuerlichkeit"] = 0.82

        decision = evaluate_spec_upgrade_candidate(
            baseline,
            candidate,
            artifact_freedom=0.97,
            panns_singing=0.0,
        )

        assert decision.promoted is True
        assert decision.reason == "promote_to_spec"
        assert decision.improved_goals_count >= 1
        assert decision.non_degraded_goals_count == 15

    def test_reject_when_artifact_freedom_below_threshold(self):
        baseline = _scores(0.8)
        candidate = _scores(0.8)
        candidate["natuerlichkeit"] = 0.85

        decision = evaluate_spec_upgrade_candidate(
            baseline,
            candidate,
            artifact_freedom=0.94,
            panns_singing=0.0,
        )

        assert decision.promoted is False
        assert decision.safety_ok is False
        assert decision.reason == "safety_fail_artifact_freedom"

    def test_reject_when_more_than_one_goal_regresses(self):
        baseline = _scores(0.8)
        candidate = _scores(0.8)
        candidate["natuerlichkeit"] = 0.82
        candidate["brillanz"] = 0.75
        candidate["waerme"] = 0.76

        decision = evaluate_spec_upgrade_candidate(
            baseline,
            candidate,
            artifact_freedom=0.98,
            panns_singing=0.0,
        )

        assert decision.promoted is False
        assert decision.non_degraded_goals_count == 13
        assert decision.reason == "too_many_goal_regressions"

    def test_reject_vocal_upgrade_when_vqi_worsens(self):
        baseline = _scores(0.8)
        candidate = _scores(0.8)
        candidate["artikulation"] = 0.83

        decision = evaluate_spec_upgrade_candidate(
            baseline,
            candidate,
            artifact_freedom=0.97,
            panns_singing=0.6,
            vqi_before=0.88,
            vqi_after=0.86,
        )

        assert decision.promoted is False
        assert decision.vqi_ok is False
        assert decision.reason == "vqi_regression_or_missing"

    def test_reject_vocal_upgrade_when_vqi_missing(self):
        baseline = _scores(0.8)
        candidate = _scores(0.8)
        candidate["artikulation"] = 0.84

        decision = evaluate_spec_upgrade_candidate(
            baseline,
            candidate,
            artifact_freedom=0.98,
            panns_singing=0.5,
            vqi_before=None,
            vqi_after=0.9,
        )

        assert decision.promoted is False
        assert decision.reason == "vqi_regression_or_missing"
