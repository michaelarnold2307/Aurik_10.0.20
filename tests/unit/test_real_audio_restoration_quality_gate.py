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
        "artifact_freedom": 0.99,
        "artifact_contract_passed": True,
        "hpi_contract_passed": True,
        "vocal_contract_passed": True,
        "export_contract_passed": True,
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
                artifact_contract_passed=False,
                hpi_contract_passed=False,
                export_contract_passed=False,
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
        expected_case_ids=("vocal_bad",),
    )

    assert gate.passed is False
    assert gate.non_degraded_export_rate == 0.0
    assert gate.unblocked_export_rate == 0.0
    assert gate.artifact_contract_rate == 0.0
    assert gate.hpi_contract_rate == 0.0
    assert gate.export_contract_rate == 0.0
    assert gate.case_evidence_complete_rate == 1.0
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


def test_restoration_quality_gate_fails_closed_on_missing_case_and_runtime_evidence() -> None:
    case = _execution_case()
    case.pop("artifact_freedom")
    case.pop("fail_reasons")
    case.pop("export_contract_passed")
    execution_report = {"gate": {}, "cases": [case]}
    thresholds = RestorationQualityThresholds(
        min_non_degraded_export_rate=0.0,
        min_unblocked_export_rate=0.0,
        min_musical_goal_case_pass_rate=0.0,
        min_noise_texture_case_pass_rate=0.0,
        min_goosebumps_case_pass_rate=0.0,
        min_final_quality_case_pass_rate=0.0,
        min_vocal_floor_pass_rate=0.0,
        min_hpi_average=0.0,
        max_runtime_factor=100.0,
        min_real_audio_cases=1,
        min_vocal_cases=0,
        min_external_benchmark_cases=0,
    )

    gate, cases = evaluate_restoration_quality_gate(execution_report, thresholds, expected_case_ids=("case",))

    assert gate.passed is False
    assert gate.case_evidence_complete_rate == 0.0
    assert cases[0].evidence_complete is False
    assert {"artifact_freedom", "fail_reasons", "export_contract_passed"} <= set(cases[0].missing_evidence)
    assert "case_evidence_complete_rate 0.000 < 1.000" in gate.fail_reasons
    assert "execution_gate runtime_factor missing, non-finite, or non-positive" in gate.fail_reasons
    assert "complete_quality_evidence" in gate.prioritized_actions


def test_restoration_quality_gate_requires_exact_manifest_case_coverage() -> None:
    execution_report = {
        "gate": {"runtime_factor": 4.0},
        "cases": [_execution_case(), _execution_case(case_id="case"), _execution_case(case_id="unexpected")],
    }

    gate, _ = evaluate_restoration_quality_gate(
        execution_report,
        RestorationQualityThresholds(min_real_audio_cases=1, min_vocal_cases=0, min_external_benchmark_cases=0),
        expected_case_ids=("case", "other"),
    )

    assert gate.passed is False
    assert gate.case_id_coverage_complete is False
    assert gate.case_id_coverage_rate == 0.5
    assert gate.missing_case_ids == ("other",)
    assert gate.unexpected_case_ids == ("unexpected",)
    assert gate.duplicate_case_ids == ("case",)
    assert any(reason.startswith("case_id_coverage incomplete") for reason in gate.fail_reasons)
    assert "verify_golden_set_case_coverage" in gate.prioritized_actions


def test_restoration_quality_gate_rejects_duplicate_expected_case_ids() -> None:
    gate, _ = evaluate_restoration_quality_gate(
        {"gate": {"runtime_factor": 4.0}, "cases": [_execution_case()]},
        RestorationQualityThresholds(min_real_audio_cases=1, min_vocal_cases=0, min_external_benchmark_cases=0),
        expected_case_ids=("case", "case"),
    )

    assert gate.passed is False
    assert gate.case_id_coverage_complete is False


@pytest.mark.parametrize(
    ("fail_code", "required_action"),
    [
        ("STEREO_SAFETY_HARD_FAIL", "stereo_safety_recovery"),
        ("SINGER_ID_BELOW_THRESHOLD", "vocal_vqi_recovery"),
        ("UNRECOGNIZED_GATE_FAILURE", "triage_quality_failure_codes"),
    ],
)
def test_restoration_quality_gate_never_waives_execution_fail_reasons(
    fail_code: str,
    required_action: str,
) -> None:
    thresholds = RestorationQualityThresholds(
        min_non_degraded_export_rate=0.0,
        min_unblocked_export_rate=0.0,
        min_musical_goal_case_pass_rate=0.0,
        min_noise_texture_case_pass_rate=0.0,
        min_goosebumps_case_pass_rate=0.0,
        min_final_quality_case_pass_rate=1.0,
        min_vocal_floor_pass_rate=0.0,
        min_hpi_average=0.0,
        max_runtime_factor=100.0,
        min_real_audio_cases=1,
        min_vocal_cases=0,
        min_external_benchmark_cases=0,
    )

    gate, cases = evaluate_restoration_quality_gate(
        {"gate": {"runtime_factor": 4.0}, "cases": [_execution_case(fail_reasons=[fail_code])]},
        thresholds,
        expected_case_ids=("case",),
    )

    assert gate.passed is False
    assert gate.final_quality_case_pass_rate == 0.0
    assert cases[0].final_quality_passed is False
    assert required_action in gate.prioritized_actions


def test_restoration_quality_gate_rejects_malformed_fail_reasons() -> None:
    gate, cases = evaluate_restoration_quality_gate(
        {"gate": {"runtime_factor": 4.0}, "cases": [_execution_case(fail_reasons=[""])]},
        RestorationQualityThresholds(min_real_audio_cases=1, min_vocal_cases=0, min_external_benchmark_cases=0),
        expected_case_ids=("case",),
    )

    assert gate.case_evidence_complete_rate == 0.0
    assert "fail_reasons" in cases[0].missing_evidence


def test_restoration_quality_gate_keeps_quality_estimate_report_only(tmp_path: Path) -> None:
    execution_report = {
        "gate": {"runtime_factor": 4.0},
        "cases": [
            _execution_case(
                case_id="instrumental_good",
                metadata={"quality_estimate": 0.1, "vqi_floor": None, "path": "audio.wav", "material": "vinyl"},
            ),
            _execution_case(
                case_id="vocal_good",
                vocal_required=True,
                vqi=0.84,
                metadata={"quality_estimate": 0.1, "vqi_floor": 0.72, "path": "vocal.wav", "material": "vinyl"},
            ),
        ],
    }
    report_path = tmp_path / "execution_report.json"
    manifest_path = tmp_path / "manifest.json"
    report_path.write_text(json.dumps(execution_report), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "cases": [{"case_id": "instrumental_good"}, {"case_id": "vocal_good"}],
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
    assert report.gate.quality_estimate_average == pytest.approx(0.1)
    assert report.gate.real_audio_cases == 2
    assert report.gate.vocal_cases == 1
    assert report.gate.case_id_coverage_complete is True
    assert report.gate.case_id_coverage_rate == 1.0
