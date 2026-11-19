from __future__ import annotations

"""Normative Contracts fuer Worldclass-KPI-Dashboard und Release-Gate.

Ziel:
- Sicherstellen, dass die KPI-Ziele versioniert sind.
- Sicherstellen, dass Dashboard + Release-Gate die 5 Kernmetriken erzwingen.
"""


import json
from pathlib import Path

import pytest

_THRESHOLDS = Path("config/worldclass_kpi_thresholds.json")
_DASHBOARD_SCRIPT = Path("scripts/worldclass_kpi_dashboard.py")
_RELEASE_GATE_SCRIPT = Path("scripts/worldclass_release_gate.py")
_AUTOPILOT_PIPELINE_SCRIPT = Path("scripts/worldclass_autopilot_pipeline.py")
_TRUSTED_REPORT_SCRIPT = Path("scripts/trusted_vocal_restoration_report.py")

_REQUIRED_TARGET_KEYS = {
    "artifact_freedom_pass_rate_min",
    "vqi_margin_pass_rate_min",
    "wcs_pass_rate_min",
    "defect_detection_pass_rate_min",
    "defect_confidence_pass_rate_min",
    "defect_confidence_coverage_rate_min",
    "defect_confidence_value_min",
    "era_confidence_pass_rate_min",
    "genre_confidence_pass_rate_min",
    "material_confidence_pass_rate_min",
    "pipeline_confidence_pass_rate_min",
    "uncertainty_coverage_rate_min",
    "era_confidence_value_min",
    "genre_confidence_value_min",
    "material_confidence_value_min",
    "pipeline_confidence_value_min",
    "defect_inaudible_or_max_reduction_pass_rate_min",
    "false_reject_rate_max",
    "runtime_p95_seconds_max",
}


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_worldclass_threshold_config_exists_and_has_required_targets() -> None:
    assert _THRESHOLDS.exists(), "config/worldclass_kpi_thresholds.json fehlt."
    data = json.loads(_THRESHOLDS.read_text(encoding="utf-8"))
    assert isinstance(data, dict), "KPI-Threshold-Config muss ein JSON-Objekt sein."
    targets = data.get("targets")
    assert isinstance(targets, dict), "KPI-Threshold-Config braucht ein targets-Objekt."

    missing = sorted(k for k in _REQUIRED_TARGET_KEYS if k not in targets)
    assert not missing, f"Fehlende KPI-Target-Keys: {missing}"


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_worldclass_threshold_config_defines_real_audio_corpus_requirements() -> None:
    data = json.loads(_THRESHOLDS.read_text(encoding="utf-8"))
    corpus = data.get("real_audio_corpus")
    assert isinstance(corpus, dict), "KPI-Config braucht real_audio_corpus fuer R5-R12-Real-Audio-Gate."
    assert int(corpus.get("min_cases", 0)) >= 5
    assert set(corpus.get("required_materials", [])) >= {"shellac", "vinyl", "tape", "cd_digital", "mp3_low"}
    assert set(corpus.get("required_case_ids", [])) >= {
        "shellac_vocal_001",
        "vinyl_vocal_001",
        "tape_vocal_001",
        "cd_vocal_001",
        "mp3_vocal_001",
    }
    assert corpus.get("all_required_cases_must_be_vocal_focus") is True


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_worldclass_threshold_config_defines_trusted_vocal_restoration_requirements() -> None:
    data = json.loads(_THRESHOLDS.read_text(encoding="utf-8"))
    trusted = data.get("trusted_vocal_restoration")
    assert isinstance(trusted, dict), "KPI-Config braucht trusted_vocal_restoration fuer Profi-Evidenz."
    assert trusted.get("allow_metric_only_claims") is False
    assert int(trusted.get("min_professional_cases", 0)) >= 20
    assert int(trusted.get("target_cases", 0)) >= 50
    assert set(trusted.get("required_baseline_families", [])) >= {
        "input_passthrough",
        "classical_dsp",
        "sota_ml",
        "commercial_reference",
    }
    assert set(trusted.get("required_aurik_metrics", [])) >= {
        "artifact_freedom",
        "hpi",
        "vqi",
        "timbral_fidelity",
    }
    hearing = trusted.get("human_hearing_focus")
    assert isinstance(hearing, dict), "Trusted-Report muss Human-Hearing-Fokus erzwingen."
    assert set(hearing.get("required_metrics", [])) >= {
        "naturalness",
        "emotional_arc_preservation",
        "micro_dynamic_correlation",
        "formant_integrity",
        "vibrato_depth_preservation",
        "noise_texture_distance",
    }
    thresholds = hearing.get("thresholds", {})
    assert thresholds.get("naturalness_min") >= 0.90
    assert thresholds.get("micro_dynamic_correlation_min") >= 0.97
    assert thresholds.get("noise_texture_distance_max") <= 0.25
    automation = trusted.get("fully_automated_operation")
    assert isinstance(automation, dict), "Trusted-Report muss vollautomatisierten Betrieb erzwingen."
    assert automation.get("required") is True
    assert automation.get("allowed_user_decisions") == ["mode_selection"]
    assert automation.get("max_manual_interventions_per_case") == 0
    assert automation.get("max_user_parameters_per_case") == 0
    assert automation.get("require_canonical_bridge_contract") is True
    assert automation.get("require_autonomous_export_decision") is True


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_worldclass_vqi_margin_floors_keep_cd_digital_release_floor() -> None:
    data = json.loads(_THRESHOLDS.read_text(encoding="utf-8"))
    floors = data.get("vqi_margin", {}).get("material_floors", {})

    assert floors.get("cd_digital") == 0.82, "CD-Digital-VQI-Floor darf nicht unter §0p/PMGG 0.82 fallen."


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_dashboard_script_collects_all_required_kpis() -> None:
    assert _DASHBOARD_SCRIPT.exists(), "scripts/worldclass_kpi_dashboard.py fehlt."
    text = _DASHBOARD_SCRIPT.read_text(encoding="utf-8")

    for key in [
        "artifact_freedom_pass_rate",
        "vqi_margin_pass_rate",
        "wcs_pass_rate",
        "defect_detection_pass_rate",
        "defect_confidence_pass_rate",
        "defect_confidence_coverage_rate",
        "era_confidence_pass_rate",
        "genre_confidence_pass_rate",
        "material_confidence_pass_rate",
        "pipeline_confidence_pass_rate",
        "uncertainty_coverage_rate",
        "defect_inaudible_or_max_reduction_pass_rate",
        "false_reject_rate",
        "runtime_p95_seconds",
        "real_audio_corpus",
        "trusted_vocal_restoration",
        "missing_required_materials",
        "missing_required_case_ids",
        "non_vocal_required_case_ids",
    ]:
        assert key in text, f"Dashboard-Skript muss KPI '{key}' verarbeiten."


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_release_gate_script_enforces_all_required_kpis() -> None:
    assert _RELEASE_GATE_SCRIPT.exists(), "scripts/worldclass_release_gate.py fehlt."
    text = _RELEASE_GATE_SCRIPT.read_text(encoding="utf-8")

    # Min-Gates
    assert "artifact_freedom_pass_rate" in text
    assert "vqi_margin_pass_rate" in text
    assert "wcs_pass_rate" in text
    assert "defect_detection_pass_rate" in text
    assert "defect_confidence_pass_rate" in text
    assert "defect_confidence_coverage_rate" in text
    assert "era_confidence_pass_rate" in text
    assert "genre_confidence_pass_rate" in text
    assert "material_confidence_pass_rate" in text
    assert "pipeline_confidence_pass_rate" in text
    assert "uncertainty_coverage_rate" in text
    assert "defect_inaudible_or_max_reduction_pass_rate" in text

    # Max-Gates
    assert "false_reject_rate" in text
    assert "runtime_p95_seconds" in text
    assert "real_audio_corpus" in text
    assert "trusted_vocal_restoration" in text
    assert "trusted_vocal_restoration_report" in text
    assert "missing_required_materials" in text
    assert "missing_required_case_ids" in text
    assert "non_vocal_required_case_ids" in text

    # Harte Fail-Semantik
    assert "WORLDCLASS RELEASE GATE: FAIL" in text
    assert "SystemExit(2)" in text


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_autopilot_pipeline_script_wires_required_worldclass_steps() -> None:
    assert _AUTOPILOT_PIPELINE_SCRIPT.exists(), "scripts/worldclass_autopilot_pipeline.py fehlt."
    text = _AUTOPILOT_PIPELINE_SCRIPT.read_text(encoding="utf-8")

    for required in [
        "run_class_c_revalidation_plan.py",
        "run_wp1_material_vqi_revalidation.py",
        "summarize_class_c_revalidation_results.py",
        "worldclass_kpi_dashboard.py",
        "trusted_vocal_restoration_report.py",
        "trusted_vocal_restoration_report.json",
        "worldclass_release_gate.py",
    ]:
        assert required in text, f"Autopilot-Pipeline muss Schritt '{required}' aufrufen."


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_trusted_vocal_restoration_report_script_enforces_professional_evidence_contract() -> None:
    assert _TRUSTED_REPORT_SCRIPT.exists(), "scripts/trusted_vocal_restoration_report.py fehlt."
    text = _TRUSTED_REPORT_SCRIPT.read_text(encoding="utf-8")
    for required in [
        "trusted_vocal_restoration",
        "min_professional_cases",
        "human_hearing_focus",
        "fully_automated_operation",
        "user_confidence_summary",
        "required_baseline_families",
        "commercial_reference",
        "safety_regressions",
        "best_possible_restoration",
        "User Confidence Summary",
        "manual_action_required",
        "allowed_user_decisions",
        "phase_hardening_actions",
        "professional_limitations",
        "TRUSTED VOCAL RESTORATION",
    ]:
        assert required in text, f"Trusted-Report muss '{required}' erzwingen."
