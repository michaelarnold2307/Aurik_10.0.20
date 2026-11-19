from __future__ import annotations

"""Normative test: §0c Recovery-Lite — best_effort PMGG actions must propagate
transparent recovery metadata into PhaseGateLogEntry.metadata.

These tests verify that no best_effort action is silently treated as a success —
every best_effort outcome MUST be marked with recovery_attempted=True and
best_possible_reached=True so downstream components (UV3, bridge, export_workflow)
can distinguish a full pass from a tolerated recovery.
"""


from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_phase_mock(phase_id: str, regression_delta: float = -0.10) -> MagicMock:
    """Return a minimal phase stub that always causes a regression of regression_delta."""
    phase = MagicMock()
    phase.get_metadata.return_value = MagicMock(phase_id=phase_id)

    def _process(audio: np.ndarray, sr: int = 48_000, strength: float = 1.0, **_kw: Any) -> np.ndarray:
        # Scale audio slightly to simulate a regression in goal measurement
        return audio * (1.0 - abs(regression_delta))

    phase.process.side_effect = _process
    return phase


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.normative
@pytest.mark.timeout(15)
def test_best_effort_action_sets_recovery_attempted_in_log_entry() -> None:
    """§0c: A best_effort action MUST populate recovery_attempted=True in the log entry.

    This test drives PMGG to a best_effort outcome by forcing all retries to still
    exceed the regression threshold, then asserts that the returned PhaseGateLogEntry
    carries the mandatory recovery metadata.
    """
    from backend.core.per_phase_musical_goals_gate import get_phase_gate

    gate = get_phase_gate()

    audio = np.random.default_rng(42).random(48_000).astype(np.float32) * 0.5

    # Scores before: at threshold boundary so any decrease triggers regression path
    scores_before = {
        "natuerlichkeit": 0.91,
        "authentizitaet": 0.90,
    }

    # Phase always returns a regressed signal — will exhaust all retries → best_effort
    phase = _make_phase_mock("phase_03_denoise", regression_delta=-0.15)

    # Patch _measure_quick so regression always exceeds threshold regardless of strength
    def _always_regressed(sample: np.ndarray, sr: int, **_kw: Any) -> dict[str, float]:
        return {
            "natuerlichkeit": 0.70,  # clearly below 0.91
            "authentizitaet": 0.72,  # clearly below 0.90
        }

    with patch("backend.core.per_phase_musical_goals_gate._measure_quick", side_effect=_always_regressed):
        _result, _scores_out, log_entry = gate.check_phase(
            phase,
            audio,
            sr=48_000,
            scores_before=scores_before,
            effective_goals=["natuerlichkeit", "authentizitaet"],
        )

    # The action must be a best_effort variant
    assert log_entry.action.startswith("best_effort"), f"Expected best_effort* action, got: {log_entry.action!r}"
    # §0c Recovery-Lite transparency invariant:
    assert log_entry.metadata.get("recovery_attempted") is True, (
        "best_effort action MUST set metadata['recovery_attempted'] = True"
    )
    assert log_entry.metadata.get("best_possible_reached") is True, (
        "best_effort action MUST set metadata['best_possible_reached'] = True"
    )


@pytest.mark.normative
@pytest.mark.timeout(15)
def test_passed_action_does_not_set_recovery_metadata() -> None:
    """§0c: A clean passed action must NOT set recovery metadata (no false positives)."""
    from backend.core.per_phase_musical_goals_gate import get_phase_gate

    gate = get_phase_gate()

    audio = np.random.default_rng(7).random(48_000).astype(np.float32) * 0.5

    scores_before = {
        "natuerlichkeit": 0.80,
        "authentizitaet": 0.82,
    }

    phase = _make_phase_mock("phase_18_noise_gate", regression_delta=0.0)

    # Patch _measure_quick to return improvement (no regression)
    def _always_improved(sample: np.ndarray, sr: int, **_kw: Any) -> dict[str, float]:
        return {
            "natuerlichkeit": 0.93,
            "authentizitaet": 0.94,
        }

    with patch("backend.core.per_phase_musical_goals_gate._measure_quick", side_effect=_always_improved):
        _result, _scores_out, log_entry = gate.check_phase(
            phase,
            audio,
            sr=48_000,
            scores_before=scores_before,
            effective_goals=["natuerlichkeit", "authentizitaet"],
        )

    assert log_entry.action in {
        "passed",
        "sub_threshold",
    }, f"Clean pass should yield 'passed' or 'sub_threshold', got: {log_entry.action!r}"
    assert not log_entry.metadata.get("recovery_attempted", False), "A clean pass must NOT set recovery_attempted=True"


@pytest.mark.normative
@pytest.mark.timeout(15)
def test_p4_p5_goal_regression_within_tolerance_yields_passed_p4p5_tolerated() -> None:
    """§0c Recovery-Lite: P4/P5 regression within 2.0×/2.5× tolerance band → passed_p4p5_tolerated.

    Regression that is above the base threshold but below the P4 tolerance band (2×)
    must produce 'passed_p4p5_tolerated' — not best_effort — and must NOT set
    recovery_attempted (it is a controlled tolerance, not a recovery).
    """
    from backend.core.per_phase_musical_goals_gate import get_phase_gate

    gate = get_phase_gate()

    audio = np.random.default_rng(13).random(48_000).astype(np.float32) * 0.5

    # Only a P4 goal in the effective set (transparenz)
    scores_before = {
        "transparenz": 0.85,
    }

    phase = _make_phase_mock("phase_29_tape_hiss_reduction", regression_delta=-0.03)

    # Regression: 0.03 — above base threshold (~0.020) but below 2.0× = 0.040
    def _p4_mild_regression(sample: np.ndarray, sr: int, **_kw: Any) -> dict[str, float]:
        return {"transparenz": 0.82}  # 0.85 - 0.03 = 0.82

    with patch("backend.core.per_phase_musical_goals_gate._measure_quick", side_effect=_p4_mild_regression):
        _result, _scores_out, log_entry = gate.check_phase(
            phase,
            audio,
            sr=48_000,
            scores_before=scores_before,
            effective_goals=["transparenz"],
        )

    assert log_entry.action == "passed_p4p5_tolerated", (
        f"P4 mild regression within tolerance band should yield 'passed_p4p5_tolerated', got: {log_entry.action!r}"
    )
    assert not log_entry.metadata.get("recovery_attempted", False), (
        "passed_p4p5_tolerated must NOT set recovery_attempted"
    )
