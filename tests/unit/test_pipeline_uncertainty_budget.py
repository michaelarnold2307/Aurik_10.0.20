from __future__ import annotations

import pytest


@pytest.mark.unit
@pytest.mark.timeout(15)
def test_estimate_goal_confidence_map_returns_bounded_values() -> None:
    from backend.core.pipeline_uncertainty import estimate_goal_confidence_map

    goal_scores = {
        "natuerlichkeit": 0.71,
        "authentizitaet": 0.69,
        "brillanz": 0.62,
        "transparenz": 0.58,
        "waerme": 0.55,
        "spatial_depth": 0.64,
    }
    conf_map = estimate_goal_confidence_map(
        goal_scores,
        pipeline_confidence=0.78,
        restoration_context={"transfer_chain": ["vinyl", "mp3_low"]},
    )

    assert conf_map
    assert set(conf_map.keys()) == set(goal_scores.keys())
    for value in conf_map.values():
        assert 0.20 <= float(value) <= 0.98


@pytest.mark.unit
@pytest.mark.timeout(15)
def test_estimate_uncertainty_budget_rises_with_chain_complexity() -> None:
    from backend.core.pipeline_uncertainty import estimate_uncertainty_budget

    goal_conf = {
        "natuerlichkeit": 0.80,
        "authentizitaet": 0.79,
        "brillanz": 0.77,
    }

    simple_budget = estimate_uncertainty_budget(
        goal_confidence=goal_conf,
        pipeline_confidence=0.82,
        transfer_chain=["cd_digital"],
    )
    complex_budget = estimate_uncertainty_budget(
        goal_confidence=goal_conf,
        pipeline_confidence=0.82,
        transfer_chain=["wire_recording", "cassette", "vinyl", "mp3_low"],
    )

    assert 0.0 <= simple_budget <= 1.0
    assert 0.0 <= complex_budget <= 1.0
    assert complex_budget > simple_budget
