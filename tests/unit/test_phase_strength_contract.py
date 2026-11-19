import pytest

"""Unit-Tests fuer den zentralen Phase-Strength-Contract."""

from backend.core.phase_strength_contract import resolve_phase_strength_contract


@pytest.mark.unit
def test_strength_contract_defaults_are_stable():
    ctx = resolve_phase_strength_contract({})
    assert ctx["phase_locality_factor"] == 1.0
    assert ctx["pmgg_strength"] == 1.0
    assert ctx["effective_strength"] == 1.0
    assert ctx["vocal_cap_applied"] is False


def test_strength_contract_clips_locality_and_effective_strength():
    ctx = resolve_phase_strength_contract({"phase_locality_factor": 0.1, "strength": 2.0})
    assert abs(ctx["phase_locality_factor"] - 0.35) < 1e-9
    assert abs(ctx["effective_strength"] - 0.7) < 1e-9


def test_strength_contract_handles_non_finite_inputs():
    ctx = resolve_phase_strength_contract({"phase_locality_factor": float("nan"), "strength": float("inf")})
    assert ctx["phase_locality_factor"] == 1.0
    assert ctx["pmgg_strength"] == 1.0
    assert ctx["effective_strength"] == 1.0


def test_strength_contract_applies_vocal_cap_when_gate_active():
    ctx = resolve_phase_strength_contract(
        {"strength": 1.0, "phase_locality_factor": 1.0, "panns_singing": 0.8},
        vocal_gate_threshold=0.35,
        vocal_strength_cap=0.7,
    )
    assert abs(ctx["effective_strength"] - 0.7) < 1e-9
    assert ctx["vocal_cap_applied"] is True


def test_strength_contract_ignores_vocal_cap_below_gate():
    ctx = resolve_phase_strength_contract(
        {"strength": 1.0, "phase_locality_factor": 1.0, "panns_singing": 0.2},
        vocal_gate_threshold=0.35,
        vocal_strength_cap=0.7,
    )
    assert abs(ctx["effective_strength"] - 1.0) < 1e-9
    assert ctx["vocal_cap_applied"] is False
