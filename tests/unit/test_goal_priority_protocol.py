from __future__ import annotations

import pytest

"""Tests für core/goal_priority_protocol.py — Spec §2.34.

≥ 25 Unit-Tests: Prioritätsregeln, Konfliktlösung, Iterations-Abbruch,
Singleton-Thread-Safety, alle 15 Goals abgedeckt.
"""


import threading

from backend.core.goal_priority_protocol import (
    ConflictResolutionResult,
    GoalPriorityProtocol,
    IterationAbortResult,
    check_iteration_abort,
    get_goal_priority_protocol,
    resolve_goal_conflict,
)

ALL_15_GOALS = [
    "bass_kraft",
    "brillanz",
    "waerme",
    "natuerlichkeit",
    "authentizitaet",
    "emotionalitaet",
    "transparenz",
    "groove",
    "spatial_depth",
    "timbre_authentizitaet",
    "tonal_center",
    "micro_dynamics",
    "separation_fidelity",
    "artikulation",
    "transient_energie",
]

PRIORITY_1 = ["natuerlichkeit", "authentizitaet"]
PRIORITY_2 = ["tonal_center", "timbre_authentizitaet", "artikulation", "transient_energie"]
PRIORITY_5 = ["brillanz", "spatial_depth"]


def _gpp() -> GoalPriorityProtocol:
    return get_goal_priority_protocol()


# ---------------------------------------------------------------------------
# 1. PRIORITY_MAP Vollständigkeit
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPriorityMap:
    def test_all_15_goals_in_map(self):
        gpp = _gpp()
        for g in ALL_15_GOALS:
            assert g in gpp.PRIORITY_MAP, f"Missing goal: {g}"

    def test_priority_levels_range(self):
        gpp = _gpp()
        for g, lvl in gpp.PRIORITY_MAP.items():
            assert 1 <= lvl <= 5, f"{g}: priority={lvl}"

    def test_level_1_goals(self):
        gpp = _gpp()
        for g in PRIORITY_1:
            assert gpp.PRIORITY_MAP[g] == 1

    def test_level_2_goals(self):
        gpp = _gpp()
        for g in PRIORITY_2:
            assert gpp.PRIORITY_MAP[g] == 2

    def test_level_5_goals(self):
        gpp = _gpp()
        for g in PRIORITY_5:
            assert gpp.PRIORITY_MAP[g] == 5

    def test_priority_of_method(self):
        gpp = _gpp()
        assert gpp.priority_of("natuerlichkeit") == 1
        assert gpp.priority_of("brillanz") == 5


# ---------------------------------------------------------------------------
# 2. resolve_conflict — Prioritätsregeln
# ---------------------------------------------------------------------------


class TestResolveConflict:
    def test_level1_beats_level5(self):
        result = resolve_goal_conflict(
            "natuerlichkeit",
            "brillanz",
            delta_a=0.02,
            delta_b=0.05,
        )
        assert isinstance(result, ConflictResolutionResult)
        assert result.winner == "natuerlichkeit"
        assert result.loser == "brillanz"

    def test_level5_beats_nothing_over_level1(self):
        result = resolve_goal_conflict(
            "brillanz",
            "natuerlichkeit",
            delta_a=0.10,
            delta_b=0.01,
        )
        assert result.winner == "natuerlichkeit"

    def test_same_priority_larger_headroom_wins(self):
        result = resolve_goal_conflict(
            "natuerlichkeit",
            "authentizitaet",
            delta_a=0.01,
            delta_b=0.08,
            headroom_a=0.05,
            headroom_b=0.20,
        )
        # headroom_b größer → authentizitaet gewinnt
        assert result.winner == "authentizitaet"

    def test_same_priority_same_headroom_goal_a_fallback(self):
        result = resolve_goal_conflict(
            "natuerlichkeit",
            "authentizitaet",
            delta_a=0.05,
            delta_b=0.05,
            headroom_a=0.10,
            headroom_b=0.10,
        )
        # Gleichstand → goal_a (natuerlichkeit) als Fallback
        assert result.winner == "natuerlichkeit"

    def test_result_has_reasons(self):
        result = resolve_goal_conflict(
            "brillanz",
            "groove",
            delta_a=0.03,
            delta_b=0.03,
        )
        assert isinstance(result.reason, str)
        assert len(result.reason) > 0

    def test_priorities_stored_in_result(self):
        result = resolve_goal_conflict(
            "natuerlichkeit",
            "spatial_depth",
            delta_a=0.01,
            delta_b=0.01,
        )
        assert result.priority_winner == 1
        assert result.priority_loser == 5

    def test_level2_beats_level4(self):
        result = resolve_goal_conflict(
            "tonal_center",
            "waerme",
            delta_a=0.01,
            delta_b=0.10,
        )
        assert result.winner == "tonal_center"

    def test_level3_loses_to_level2(self):
        result = resolve_goal_conflict(
            "groove",
            "artikulation",
            delta_a=0.05,
            delta_b=0.01,
        )
        assert result.winner == "artikulation"


# ---------------------------------------------------------------------------
# 3. should_abort_iteration
# ---------------------------------------------------------------------------


class TestShouldAbortIteration:
    def _constant(self, value: float = 0.80) -> dict:
        return dict.fromkeys(ALL_15_GOALS, value)

    def test_returns_dataclass(self):
        result = check_iteration_abort(self._constant(0.80), self._constant(0.80))
        assert isinstance(result, IterationAbortResult)

    def test_no_abort_when_all_stable(self):
        result = check_iteration_abort(self._constant(0.80), self._constant(0.80))
        assert result.should_abort is False

    def test_no_abort_when_all_improve(self):
        before = self._constant(0.70)
        after = self._constant(0.72)
        result = check_iteration_abort(before, after)
        assert result.should_abort is False

    def test_abort_on_level1_regression(self):
        before = dict.fromkeys(ALL_15_GOALS, 0.8)
        after = dict(before)
        after["natuerlichkeit"] = 0.78  # Regression 0.02 > epsilon 0.012
        result = check_iteration_abort(before, after)
        assert result.should_abort is True

    def test_abort_on_priority2_regression(self):
        before = dict.fromkeys(ALL_15_GOALS, 0.85)
        after = dict(before)
        after["tonal_center"] = 0.83  # Regression 0.02 > epsilon 0.012
        result = check_iteration_abort(before, after)
        assert result.should_abort is True

    def test_no_abort_on_level5_only_regression(self):
        before = dict.fromkeys(ALL_15_GOALS, 0.8)
        after = dict(before)
        after["brillanz"] = 0.75  # level 5 — kein Abbruch
        after["spatial_depth"] = 0.70  # level 5 — kein Abbruch
        result = check_iteration_abort(before, after)
        assert result.should_abort is False

    def test_degraded_goals_listed(self):
        before = dict.fromkeys(ALL_15_GOALS, 0.8)
        after = dict(before)
        after["natuerlichkeit"] = 0.75
        result = check_iteration_abort(before, after)
        assert "natuerlichkeit" in result.degraded_goals

    def test_abort_reason_nonempty(self):
        before = dict.fromkeys(ALL_15_GOALS, 0.8)
        after = dict(before)
        after["authentizitaet"] = 0.78
        result = check_iteration_abort(before, after)
        assert isinstance(result.reason, str)
        if result.should_abort:
            assert len(result.reason) > 0

    def test_epsilon_boundary_no_abort(self):
        before = dict.fromkeys(ALL_15_GOALS, 0.8)
        after = dict(before)
        gpp = _gpp()
        # Änderung exakt auf Epsilon-Grenze: sollte KEIN Abbruch sein
        after["natuerlichkeit"] = 0.80 - gpp.REGRESSION_EPSILON * 0.5
        result = check_iteration_abort(before, after)
        # Keine Regression über epsilon
        assert result.should_abort is False


# ---------------------------------------------------------------------------
# 4. Hilfsmethoden
# ---------------------------------------------------------------------------


class TestHelperMethods:
    def test_sort_goals_by_priority(self):
        gpp = _gpp()
        goals = ["brillanz", "natuerlichkeit", "groove", "tonal_center"]
        sorted_goals = gpp.sort_goals_by_priority(goals)
        # natuerlichkeit (1) < tonal_center (2) < groove (3) < brillanz (5)
        assert sorted_goals[0] == "natuerlichkeit"
        assert sorted_goals[-1] == "brillanz"

    def test_goals_at_priority_level(self):
        gpp = _gpp()
        lvl1 = gpp.goals_at_priority(1)
        assert set(lvl1) == set(PRIORITY_1)

    def test_goals_at_priority_level5(self):
        gpp = _gpp()
        lvl5 = gpp.goals_at_priority(5)
        assert set(lvl5) == set(PRIORITY_5)

    def test_would_violate_priority_true(self):
        gpp = _gpp()
        # brillanz verbessern auf Kosten von natuerlichkeit → Verletzung
        assert gpp.would_violate_priority("brillanz", "natuerlichkeit") is True

    def test_would_violate_priority_false(self):
        gpp = _gpp()
        # natuerlichkeit verbessern auf Kosten von brillanz → kein Problem
        assert gpp.would_violate_priority("natuerlichkeit", "brillanz") is False

    def test_user_message_for_failure_german(self):
        gpp = _gpp()
        msg = gpp.user_message_for_failure("natuerlichkeit")
        assert isinstance(msg, str)
        # Meldung sollte Deutsch sein und mindestens 10 Zeichen haben
        assert len(msg) >= 10

    def test_user_message_for_failure_level5(self):
        gpp = _gpp()
        msg = gpp.user_message_for_failure("brillanz")
        assert isinstance(msg, str)
        assert len(msg) >= 10


# ---------------------------------------------------------------------------
# 5. Singleton & Thread-Safety
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_same_instance(self):
        a = get_goal_priority_protocol()
        b = get_goal_priority_protocol()
        assert a is b

    def test_thread_safe(self):
        instances = []
        errors = []

        def worker():
            try:
                instances.append(get_goal_priority_protocol())
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert all(i is instances[0] for i in instances)

    def test_convenience_wrappers_consistent(self):
        scores_before = dict.fromkeys(ALL_15_GOALS, 0.8)
        scores_after = dict(scores_before)
        scores_after["brillanz"] = 0.75

        gpp = _gpp()
        direct = gpp.should_abort_iteration(scores_before, scores_after)
        wrapper = check_iteration_abort(scores_before, scores_after)
        assert direct.should_abort == wrapper.should_abort

    def test_resolve_conflict_convenience(self):
        result = resolve_goal_conflict("natuerlichkeit", "brillanz", 0.01, 0.10)
        assert result.winner == "natuerlichkeit"
