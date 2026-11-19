from __future__ import annotations

import pytest


@pytest.mark.unit
@pytest.mark.timeout(15)
def test_innovation_orchestrator_prioritizes_high_gap_low_confidence_goals():
    from backend.core.innovation_superiority_orchestrator import (
        get_innovation_superiority_orchestrator,
    )

    orch = get_innovation_superiority_orchestrator()
    plan = orch.build_realtime_plan(
        goal_scores={
            "natuerlichkeit": 0.74,
            "authentizitaet": 0.78,
            "transparenz": 0.68,
            "vocal_quality": 0.63,
            "formant_fidelity": 0.65,
        },
        goal_targets={
            "natuerlichkeit": 0.90,
            "authentizitaet": 0.88,
            "transparenz": 0.82,
            "vocal_quality": 0.85,
            "formant_fidelity": 0.88,
        },
        goal_confidence={
            "natuerlichkeit": 0.72,
            "authentizitaet": 0.75,
            "transparenz": 0.60,
            "vocal_quality": 0.35,
            "formant_fidelity": 0.40,
        },
        pipeline_confidence=0.66,
        material_type="vinyl",
        transfer_chain=["vinyl", "cassette", "mp3_low"],
        is_studio_2026=False,
        phase_goal_delta={"transparenz": -0.02, "vocal_quality": -0.01},
        phase_metadata={
            "pmgg_reconstruction_epistemic_confidence": 0.62,
            "pmgg_reconstruction_localized": True,
        },
    )

    assert 0.0 <= plan.innovation_intensity <= 1.0
    assert len(plan.priority_goals) >= 1
    assert "vocal_quality" in plan.priority_goals or "formant_fidelity" in plan.priority_goals
    for value in plan.goal_confidence_uplift.values():
        assert 0.0 <= float(value) <= 0.05


@pytest.mark.unit
@pytest.mark.timeout(15)
def test_innovation_orchestrator_outputs_are_bounded_and_advisory_safe():
    from backend.core.innovation_superiority_orchestrator import (
        get_innovation_superiority_orchestrator,
    )

    orch = get_innovation_superiority_orchestrator()
    plan = orch.build_realtime_plan(
        goal_scores={"natuerlichkeit": 0.89, "authentizitaet": 0.87, "transparenz": 0.81},
        goal_targets={"natuerlichkeit": 0.90, "authentizitaet": 0.88, "transparenz": 0.82},
        goal_confidence={"natuerlichkeit": 0.90, "authentizitaet": 0.88, "transparenz": 0.86},
        pipeline_confidence=0.92,
        material_type="cd_digital",
        transfer_chain=["cd_digital"],
        is_studio_2026=False,
        phase_goal_delta={"natuerlichkeit": 0.02, "authentizitaet": 0.01},
        phase_metadata={"pmgg_reconstruction_epistemic_confidence": 0.90},
    )

    assert isinstance(plan.discipline_scores, dict)
    assert isinstance(plan.recovery_phase_hints, dict)
    assert all(0.0 <= float(v) <= 1.0 for v in plan.discipline_scores.values())
    assert all(0.0 <= float(v) <= 0.05 for v in plan.goal_confidence_uplift.values())
    # Advisory-only: bei nahezu erreichten Zielen darf kein aggressiver Uplift entstehen.
    assert all(float(v) <= 0.03 for v in plan.goal_confidence_uplift.values())
