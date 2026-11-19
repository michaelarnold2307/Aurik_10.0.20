from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _make_phase_mock(phase_id: str) -> MagicMock:
    phase = MagicMock()
    phase.get_metadata.return_value = MagicMock(phase_id=phase_id)

    def _process(audio: np.ndarray, sr: int = 48_000, strength: float = 1.0, **_kw: Any) -> np.ndarray:
        return np.clip(audio * (1.0 - 0.12 * float(strength)), -1.0, 1.0).astype(np.float32)

    phase.process.side_effect = _process
    return phase


@pytest.mark.unit
@pytest.mark.timeout(15)
def test_assess_reconstruction_localized_confidence_accepts_clear_local_case() -> None:
    from backend.core.per_phase_musical_goals_gate import _assess_reconstruction_localized_confidence

    accepted, confidence, reason = _assess_reconstruction_localized_confidence(
        target_coverage=0.28,
        control_coverage=0.0,
        control_regression=0.004,
        threshold=0.02,
    )

    assert accepted is True
    assert confidence >= 0.55
    assert reason == "high_confidence"


@pytest.mark.unit
@pytest.mark.timeout(15)
def test_assess_reconstruction_localized_confidence_rejects_uncertain_case() -> None:
    from backend.core.per_phase_musical_goals_gate import _assess_reconstruction_localized_confidence

    accepted, confidence, reason = _assess_reconstruction_localized_confidence(
        target_coverage=0.06,
        control_coverage=0.0,
        control_regression=0.002,
        threshold=0.02,
    )

    assert accepted is False
    assert confidence < 0.55
    assert reason == "low_target_coverage"


@pytest.mark.unit
@pytest.mark.timeout(15)
def test_threshold_multiplier_is_stricter_for_analog_lossy_chain() -> None:
    from backend.core.per_phase_musical_goals_gate import _compute_reconstruction_threshold_multiplier

    analog_multiplier, analog_reason, analog_tcci, analog_family = _compute_reconstruction_threshold_multiplier(
        phase_kwargs={"material_type": "vinyl", "transfer_chain": ["vinyl", "mp3_low"]},
        phase_id="phase_24_dropout_repair",
    )
    digital_multiplier, digital_reason, digital_tcci, digital_family = _compute_reconstruction_threshold_multiplier(
        phase_kwargs={"material_type": "cd_digital", "transfer_chain": ["cd_digital"]},
        phase_id="phase_24_dropout_repair",
    )

    assert analog_multiplier < digital_multiplier
    assert analog_reason != digital_reason
    assert analog_tcci > digital_tcci
    assert analog_family == "vinyl"
    assert digital_family == "cd_digital"


@pytest.mark.unit
@pytest.mark.timeout(15)
def test_wrap_phase_emits_reconstruction_confidence_metadata() -> None:
    from backend.core.per_phase_musical_goals_gate import get_phase_gate

    gate = get_phase_gate()
    gate.reset()

    audio = np.random.default_rng(7).random(48_000).astype(np.float32) * 0.2
    scores_before = {"natuerlichkeit": 0.90}
    phase = _make_phase_mock("phase_24_dropout_repair")

    calls = {"idx": 0}

    def _measure_quick_side_effect(sample: np.ndarray, sr: int, **_kw: Any) -> dict[str, float]:
        calls["idx"] += 1
        if calls["idx"] == 1:
            return {"natuerlichkeit": 0.82}  # target window -> regression > threshold
        if calls["idx"] == 2:
            return {"natuerlichkeit": 0.90}  # control before
        return {"natuerlichkeit": 0.89}  # control after -> low collateral regression

    with (
        patch(
            "backend.core.per_phase_musical_goals_gate._get_reconstruction_control_window_bounds",
            return_value=(0, 24_000, 0.28, 0.0),
        ),
        patch("backend.core.per_phase_musical_goals_gate._measure_quick", side_effect=_measure_quick_side_effect),
    ):
        _audio_out, _scores_after, log_entry = gate.check_phase(
            phase,
            audio,
            sr=48_000,
            scores_before=scores_before,
            effective_goals=["natuerlichkeit"],
            phase_kwargs={
                "defect_locations": {"DROPOUTS": [(0.1, 0.2)]},
            },
        )

    assert log_entry.action == "passed_reconstruction_localized"
    assert log_entry.metadata.get("pmgg_reconstruction_localized") is True
    assert float(log_entry.metadata.get("pmgg_reconstruction_confidence", 0.0)) >= 0.55
    assert log_entry.metadata.get("pmgg_reconstruction_reason") == "high_confidence"
    assert int(log_entry.metadata.get("pmgg_reconstruction_retry_budget_bias", 0)) == 2
    assert float(log_entry.metadata.get("pmgg_reconstruction_threshold_multiplier", 1.0)) >= 1.0
    assert log_entry.metadata.get("pmgg_reconstruction_material_family") == "digital"
    assert float(log_entry.metadata.get("pmgg_reconstruction_threshold_multiplier_tcci", 0.0)) == 0.0
    assert float(log_entry.metadata.get("pmgg_reconstruction_epistemic_confidence", 0.0)) > 0.0
    assert log_entry.metadata.get("pmgg_reconstruction_epistemic_reason") in {
        "epistemic_high",
        "epistemic_medium",
        "epistemic_low",
    }
    assert float(log_entry.metadata.get("pmgg_reconstruction_uncertainty_budget", 2.0)) < 1.0


@pytest.mark.unit
@pytest.mark.timeout(15)
def test_wrap_phase_reconstruction_without_control_window_sets_uncertainty_metadata() -> None:
    from backend.core.per_phase_musical_goals_gate import get_phase_gate

    gate = get_phase_gate()
    gate.reset()

    audio = np.random.default_rng(11).random(48_000).astype(np.float32) * 0.2
    scores_before = {"natuerlichkeit": 0.90}
    phase = _make_phase_mock("phase_50_spectral_repair")

    def _measure_quick_side_effect(sample: np.ndarray, sr: int, **_kw: Any) -> dict[str, float]:
        return {"natuerlichkeit": 0.82}

    with (
        patch(
            "backend.core.per_phase_musical_goals_gate._get_reconstruction_control_window_bounds",
            return_value=None,
        ),
        patch("backend.core.per_phase_musical_goals_gate._measure_quick", side_effect=_measure_quick_side_effect),
    ):
        _audio_out, _scores_after, log_entry = gate.check_phase(
            phase,
            audio,
            sr=48_000,
            scores_before=scores_before,
            effective_goals=["natuerlichkeit"],
            phase_kwargs={
                "defect_locations": {"SPECTRAL_HOLES": [(0.1, 0.2)]},
            },
        )

    assert log_entry.metadata.get("pmgg_reconstruction_localized") is False
    assert float(log_entry.metadata.get("pmgg_reconstruction_confidence", 1.0)) == 0.0
    assert log_entry.metadata.get("pmgg_reconstruction_reason") == "counterfactual_window_unavailable"
    assert int(log_entry.metadata.get("pmgg_reconstruction_retry_budget_bias", 0)) == -1
    assert float(log_entry.metadata.get("pmgg_reconstruction_threshold_multiplier", 1.0)) >= 1.0
    assert float(log_entry.metadata.get("pmgg_reconstruction_epistemic_confidence", 1.0)) == 0.0
    assert log_entry.metadata.get("pmgg_reconstruction_epistemic_reason") == "counterfactual_window_unavailable"
    assert float(log_entry.metadata.get("pmgg_reconstruction_uncertainty_budget", 0.0)) == 1.0


@pytest.mark.unit
@pytest.mark.timeout(15)
def test_wrap_phase_retry_bias_is_stricter_for_analog_lossy_chain() -> None:
    from backend.core.per_phase_musical_goals_gate import get_phase_gate

    gate = get_phase_gate()
    gate.reset()

    audio = np.random.default_rng(21).random(48_000).astype(np.float32) * 0.2
    scores_before = {"natuerlichkeit": 0.90}
    phase = _make_phase_mock("phase_24_dropout_repair")

    calls = {"idx": 0}

    def _measure_quick_side_effect(sample: np.ndarray, sr: int, **_kw: Any) -> dict[str, float]:
        calls["idx"] += 1
        if calls["idx"] == 1:
            return {"natuerlichkeit": 0.82}
        if calls["idx"] == 2:
            return {"natuerlichkeit": 0.90}
        return {"natuerlichkeit": 0.89}

    with (
        patch(
            "backend.core.per_phase_musical_goals_gate._get_reconstruction_control_window_bounds",
            return_value=(0, 24_000, 0.28, 0.0),
        ),
        patch("backend.core.per_phase_musical_goals_gate._measure_quick", side_effect=_measure_quick_side_effect),
    ):
        _audio_out, _scores_after, analog_entry = gate.check_phase(
            phase,
            audio,
            sr=48_000,
            scores_before=scores_before,
            effective_goals=["natuerlichkeit"],
            phase_kwargs={
                "defect_locations": {"DROPOUTS": [(0.1, 0.2)]},
                "material_type": "vinyl",
                "transfer_chain": ["vinyl", "mp3_low"],
            },
        )

    gate.reset()
    calls = {"idx": 0}

    with (
        patch(
            "backend.core.per_phase_musical_goals_gate._get_reconstruction_control_window_bounds",
            return_value=(0, 24_000, 0.28, 0.0),
        ),
        patch("backend.core.per_phase_musical_goals_gate._measure_quick", side_effect=_measure_quick_side_effect),
    ):
        _audio_out, _scores_after, digital_entry = gate.check_phase(
            phase,
            audio,
            sr=48_000,
            scores_before=scores_before,
            effective_goals=["natuerlichkeit"],
            phase_kwargs={
                "defect_locations": {"DROPOUTS": [(0.1, 0.2)]},
                "material_type": "cd_digital",
                "transfer_chain": ["cd_digital"],
            },
        )

    assert int(analog_entry.metadata.get("pmgg_reconstruction_retry_budget_bias", 0)) <= int(
        digital_entry.metadata.get("pmgg_reconstruction_retry_budget_bias", 0)
    )
    assert float(analog_entry.metadata.get("pmgg_reconstruction_transfer_chain_tcci", 0.0)) > float(
        digital_entry.metadata.get("pmgg_reconstruction_transfer_chain_tcci", 0.0)
    )
    assert analog_entry.metadata.get("pmgg_reconstruction_material_family") == "analog"
    assert digital_entry.metadata.get("pmgg_reconstruction_material_family") == "digital"
