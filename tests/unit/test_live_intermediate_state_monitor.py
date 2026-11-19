import pytest

from audit.live_intermediate_state_monitor import _detect_spec_gap_from_line, _has_offtrack_trigger


@pytest.mark.unit
def test_stereo_remediation_is_not_treated_as_open_spec_gap() -> None:
    line = (
        "[2026-05-24 10:56:41,814] WARNING backend.core.unified_restorer_v3: "
        "§2.50 Stereo-Notfall-Remediation: ratio=0.00, mean_compat=0.666 → injiziert: ['phase_15_stereo_balance']"
    )

    assert _detect_spec_gap_from_line(line) is None


def test_recursion_error_still_maps_to_critical_gap() -> None:
    line = "RecursionError: maximum recursion depth exceeded while calling a Python object"

    gap = _detect_spec_gap_from_line(line)

    assert gap is not None
    assert gap["gap_id"] == "ui_recursion"
    assert gap["severity"] == "critical"


def test_high_gap_without_hard_id_does_not_force_offtrack() -> None:
    assert not _has_offtrack_trigger(
        [
            {
                "gap_id": "active_intervention_rejected",
                "severity": "high",
                "recommended_action": "fallback",
                "evidence": "ActiveIntervention phase_01_click_removal REJECTED",
            }
        ]
    )
