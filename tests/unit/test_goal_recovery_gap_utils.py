"""Unit-Tests für End-Gate Goal-Recovery Hilfslogik in UV3."""

import pytest

from backend.core.unified_restorer_v3 import _compute_weighted_goal_gap


@pytest.mark.unit
def test_compute_weighted_goal_gap_zero_when_all_goals_pass() -> None:
    scores = {"natuerlichkeit": 0.92, "authentizitaet": 0.90}
    thresholds = {"natuerlichkeit": 0.90, "authentizitaet": 0.88}
    applicable = {"natuerlichkeit", "authentizitaet"}

    gap, violations = _compute_weighted_goal_gap(scores, thresholds, applicable, None)

    assert gap == pytest.approx(0.0)
    assert violations == 0


def test_compute_weighted_goal_gap_counts_only_applicable_goals() -> None:
    scores = {
        "natuerlichkeit": 0.80,
        "authentizitaet": 0.70,
        "brillanz": 0.50,
    }
    thresholds = {
        "natuerlichkeit": 0.90,
        "authentizitaet": 0.88,
        "brillanz": 0.78,
    }
    applicable = {"natuerlichkeit", "authentizitaet"}

    gap, violations = _compute_weighted_goal_gap(scores, thresholds, applicable, None)

    # brillanz ist nicht applicable und darf nicht einfließen
    assert gap == pytest.approx((0.90 - 0.80) + (0.88 - 0.70), abs=1e-9)
    assert violations == 2


def test_compute_weighted_goal_gap_applies_goal_weights_with_clamp() -> None:
    scores = {"natuerlichkeit": 0.82, "authentizitaet": 0.80}
    thresholds = {"natuerlichkeit": 0.90, "authentizitaet": 0.88}
    applicable = {"natuerlichkeit", "authentizitaet"}

    # natuerlichkeit weight > 2.0 wird auf 2.0 geclippt
    # authentizitaet weight < 0.3 wird auf 0.3 geclippt
    weights = {"natuerlichkeit": 3.5, "authentizitaet": 0.1}

    gap, violations = _compute_weighted_goal_gap(scores, thresholds, applicable, weights)

    expected = (0.90 - 0.82) * 2.0 + (0.88 - 0.80) * 0.3
    assert gap == pytest.approx(expected, abs=1e-9)
    assert violations == 2
