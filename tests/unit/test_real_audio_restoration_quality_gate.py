from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.core.real_audio_restoration_quality_gate import (
    RestorationQualityThresholds,
    evaluate_restoration_quality_gate,
    run_real_audio_restoration_quality_gate,
)


def _execution_case(**overrides):
    base = {
        "case_id": "case",
        "hpi": 0.82,
        "vqi": None,
        "vocal_required": False,
        "export_strategy": "success",
        "export_blocked": False,
        "degradation_status": "ok",
        "fail_reasons": [],
        "metadata": {
            "quality_estimate": 0.86,
            "vqi_floor": None,
            "path": "audio.wav",
            "material": "vinyl",
        },
    }
    base.update(overrides)
    return base


@pytest.mark.unit
def test_restoration_quality_gate_flags_worldclass_gaps() -> None:
    execution_report = {
        "gate": {"runtime_factor": 19.5},
        "cases": [
            _execution_case(
                case_id="vocal_bad",
                hpi=0.55,
                vqi=0.66,
                vocal_required=True,
                export_strategy="blocked",
                export_blocked=True,
                degradation_status="degraded",
                fail_reasons=[
                    "MUSICAL_GOALS_VIOLATION",
                    "NOISE_TEXTURE_INCOHERENT",
                    "GOOSEBUMPS_LOW",
                    "VQI_BELOW_THRESHOLD",
                ],
                metadata={"quality_estimate": 0.69, "vqi_floor": 0.72, "path": "vocal.wav", "material": "vinyl"},
            )
        ],
    }

    gate, cases = evaluate_restoration_quality_gate(
        execution_report,
        RestorationQualityThresholds(min_real_audio_cases=2, min_vocal_cases=2, min_external_benchmark_cases=1),
    )

    assert gate.passed is False
    assert gate.non_degraded_export_rate == 0.0
    assert gate.unblocked_export_rate == 0.0
    assert gate.musical_goal_case_pass_rate == 0.0
    assert gate.noise_texture_case_pass_rate == 0.0
    assert gate.goosebumps_case_pass_rate == 0.0
    assert gate.vocal_floor_pass_rate == 0.0
    assert "goal_directed_candidate_recovery" in gate.prioritized_actions
    assert "noise_texture_repair" in gate.prioritized_actions
    assert "frisson_goosebumps_protection" in gate.prioritized_actions
    assert "vocal_vqi_recovery" in gate.prioritized_actions
    assert "phase_minimalism_runtime_budget" in gate.prioritized_actions
    assert "expand_real_audio_golden_set" in gate.prioritized_actions
    assert "add_external_rx_top_tool_benchmark" in gate.prioritized_actions
    assert cases[0].final_quality_passed is False


def test_restoration_quality_gate_passes_high_quality_execution_report(tmp_path: Path) -> None:
    execution_report = {
        "gate": {"runtime_factor": 4.0},
        "cases": [
            _execution_case(case_id="instrumental_good"),
            _execution_case(
                case_id="vocal_good",
                vocal_required=True,
                vqi=0.84,
                metadata={"quality_estimate": 0.88, "vqi_floor": 0.72, "path": "vocal.wav", "material": "vinyl"},
            ),
        ],
    }
    report_path = tmp_path / "execution_report.json"
    manifest_path = tmp_path / "manifest.json"
    report_path.write_text(json.dumps(execution_report), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "cases": [],
                "restoration_quality_thresholds": {
                    "min_non_degraded_export_rate": 1.0,
                    "min_unblocked_export_rate": 1.0,
                    "min_musical_goal_case_pass_rate": 1.0,
                    "min_noise_texture_case_pass_rate": 1.0,
                    "min_goosebumps_case_pass_rate": 1.0,
                    "min_vocal_floor_pass_rate": 1.0,
                    "min_hpi_average": 0.80,
                    "min_quality_estimate_average": 0.85,
                    "max_runtime_factor": 5.0,
                    "min_real_audio_cases": 2,
                    "min_vocal_cases": 1,
                    "min_external_benchmark_cases": 2,
                },
            }
        ),
        encoding="utf-8",
    )

    report = run_real_audio_restoration_quality_gate(
        execution_report_path=report_path,
        manifest_path=manifest_path,
        external_benchmark_cases=2,
    )

    assert report.gate.passed is True
    assert report.gate.non_degraded_export_rate == 1.0
    assert report.gate.unblocked_export_rate == 1.0
    assert report.gate.vocal_floor_pass_rate == 1.0
    assert report.gate.real_audio_cases == 2
    assert report.gate.vocal_cases == 1
