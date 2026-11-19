from __future__ import annotations

"""Normative Tests fuer den Trusted Vocal Restoration Report."""


import csv
import json
from pathlib import Path

import pytest


def _trusted_case_fields() -> dict[str, object]:
    return {
        "naturalness": "0.93",
        "emotional_arc_preservation": "0.88",
        "micro_dynamic_correlation": "0.98",
        "formant_integrity": "0.82",
        "vibrato_depth_preservation": "0.90",
        "noise_texture_distance": "0.18",
        "manual_intervention_count": "0",
        "user_parameter_count": "0",
        "canonical_bridge_contract": "true",
        "autonomous_export_decision": "true",
        "mode": "restoration",
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_trusted_report_fails_without_required_baselines(tmp_path: Path) -> None:
    from scripts.trusted_vocal_restoration_report import _load_rows, build_report

    csv_path = tmp_path / "results.csv"
    _write_csv(
        csv_path,
        [
            {
                **_trusted_case_fields(),
                "case_id": "shellac_vocal_001",
                "system": "aurik",
                "vocal_focus": "true",
                "material": "shellac",
                "era": "1930",
                "genre": "vocal",
                "artifact_freedom": "0.98",
                "hpi": "0.12",
                "vqi": "0.76",
                "timbral_fidelity": "0.90",
            }
        ],
    )
    cfg = json.loads(Path("config/worldclass_kpi_thresholds.json").read_text(encoding="utf-8"))

    report = build_report(_load_rows([csv_path]), cfg)

    assert report["verdict"] == "RECOVERED"
    assert report["best_possible_restoration"]["best_possible_reached"] is True
    assert report["user_confidence_summary"]["confidence_level"] == "begrenzt"
    assert report["user_confidence_summary"]["manual_action_required"] is False
    assert any("professional_vocal_cases" in violation for violation in report["violations"])
    assert any("missing_baseline_families" in violation for violation in report["violations"])


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_trusted_report_ignores_empty_template_rows_for_aurik_selection(tmp_path: Path) -> None:
    from scripts.trusted_vocal_restoration_report import _load_rows, build_report

    csv_path = tmp_path / "results.csv"
    _write_csv(
        csv_path,
        [
            {
                **_trusted_case_fields(),
                "case_id": "vinyl_vocal_001",
                "system": "aurik",
                "is_aurik": "true",
                "variant": "baseline",
                "vocal_focus": "true",
                "material": "vinyl",
                "era": "1970",
                "genre": "vocal",
                "artifact_freedom": "0.98",
                "hpi": "0.20",
                "vqi": "0.80",
                "timbral_fidelity": "0.91",
                "status": "recovered",
            },
            {
                "case_id": "vinyl_vocal_001",
                "variant": "mert_0p50",
                "vocal_focus": "true",
                "material": "vinyl",
                "status": "",
            },
        ],
    )
    cfg = json.loads(Path("config/worldclass_kpi_thresholds.json").read_text(encoding="utf-8"))

    report = build_report(_load_rows([csv_path]), cfg)

    assert "vinyl_vocal_001" in report["professional_case_ids"]
    assert not any("missing_aurik_metrics" in violation for violation in report["violations"])


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_trusted_report_blocks_safety_regression_against_best_baseline(tmp_path: Path) -> None:
    from scripts.trusted_vocal_restoration_report import _load_rows, build_report

    rows: list[dict[str, object]] = []
    for idx in range(20):
        case_id = f"vocal_case_{idx:03d}"
        rows.append(
            {
                **_trusted_case_fields(),
                "case_id": case_id,
                "system": "aurik",
                "vocal_focus": "true",
                "material": "vinyl",
                "era": "1970",
                "genre": "vocal",
                "artifact_freedom": "0.94" if idx == 0 else "0.98",
                "hpi": "0.20",
                "vqi": "0.80",
                "timbral_fidelity": "0.91",
            }
        )
        for family in ["input_passthrough", "classical_dsp", "sota_ml", "commercial_reference"]:
            rows.append(
                {
                    "case_id": case_id,
                    "system": family,
                    "baseline_family": family,
                    "vocal_focus": "true",
                    "artifact_freedom": "0.96",
                    "hpi": "0.10",
                    "vqi": "0.74",
                }
            )
    csv_path = tmp_path / "results.csv"
    _write_csv(csv_path, rows)
    cfg = json.loads(Path("config/worldclass_kpi_thresholds.json").read_text(encoding="utf-8"))

    report = build_report(_load_rows([csv_path]), cfg)

    assert report["verdict"] == "DEGRADED"
    assert report["best_possible_restoration"]["export_policy"] == "input_or_best_safe_checkpoint"
    assert report["user_confidence_summary"]["confidence_level"] == "geschuetzt"
    assert report["user_confidence_summary"]["manual_action_required"] is False
    assert report["baseline_comparison"]["missing_baseline_families"] == []
    assert any(item["metric"] == "artifact_freedom" for item in report["safety_regressions"])


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_trusted_report_passes_with_20_vocal_cases_and_all_baselines(tmp_path: Path) -> None:
    from scripts.trusted_vocal_restoration_report import _load_rows, build_report

    rows: list[dict[str, object]] = []
    for idx in range(20):
        case_id = f"vocal_case_{idx:03d}"
        rows.append(
            {
                **_trusted_case_fields(),
                "case_id": case_id,
                "system": "aurik",
                "vocal_focus": "true",
                "material": "vinyl",
                "era": "1970",
                "genre": "vocal",
                "artifact_freedom": "0.98",
                "hpi": "0.20",
                "vqi": "0.80",
                "timbral_fidelity": "0.91",
            }
        )
        for family in ["input_passthrough", "classical_dsp", "sota_ml", "commercial_reference"]:
            rows.append(
                {
                    "case_id": case_id,
                    "system": family,
                    "baseline_family": family,
                    "vocal_focus": "true",
                    "artifact_freedom": "0.96",
                    "hpi": "0.10",
                    "vqi": "0.74",
                }
            )
    csv_path = tmp_path / "results.csv"
    _write_csv(csv_path, rows)
    cfg = json.loads(Path("config/worldclass_kpi_thresholds.json").read_text(encoding="utf-8"))

    report = build_report(_load_rows([csv_path]), cfg)

    assert report["verdict"] == "PASS"
    assert report["professional_case_count"] == 20
    assert report["baseline_comparison"]["missing_baseline_families"] == []
    assert report["human_hearing_focus"]["passed"] is True
    assert report["fully_automated_operation"]["passed"] is True
    assert report["user_confidence_summary"]["confidence_level"] == "hoch"
    assert report["user_confidence_summary"]["allowed_user_decisions"] == ["mode_selection"]


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_trusted_report_blocks_metric_win_with_human_hearing_damage(tmp_path: Path) -> None:
    from scripts.trusted_vocal_restoration_report import _load_rows, build_report

    rows: list[dict[str, object]] = []
    for idx in range(20):
        case_id = f"vocal_case_{idx:03d}"
        fields = _trusted_case_fields()
        if idx == 0:
            fields["naturalness"] = "0.70"
            fields["noise_texture_distance"] = "0.40"
        rows.append(
            {
                **fields,
                "case_id": case_id,
                "system": "aurik",
                "vocal_focus": "true",
                "material": "vinyl",
                "era": "1970",
                "genre": "vocal",
                "artifact_freedom": "0.98",
                "hpi": "0.20",
                "vqi": "0.80",
                "timbral_fidelity": "0.91",
            }
        )
        for family in ["input_passthrough", "classical_dsp", "sota_ml", "commercial_reference"]:
            rows.append(
                {
                    "case_id": case_id,
                    "system": family,
                    "baseline_family": family,
                    "vocal_focus": "true",
                    "artifact_freedom": "0.96",
                    "hpi": "0.10",
                    "vqi": "0.74",
                }
            )
    csv_path = tmp_path / "results.csv"
    _write_csv(csv_path, rows)
    cfg = json.loads(Path("config/worldclass_kpi_thresholds.json").read_text(encoding="utf-8"))

    report = build_report(_load_rows([csv_path]), cfg)

    assert report["verdict"] == "DEGRADED"
    assert report["human_hearing_focus"]["passed"] is False
    assert report["user_confidence_summary"]["export_policy"] == "input_or_best_safe_checkpoint"
    assert any("human_hearing_focus" in violation for violation in report["violations"])


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_trusted_report_blocks_manual_intervention_even_when_audio_metrics_pass(tmp_path: Path) -> None:
    from scripts.trusted_vocal_restoration_report import _load_rows, build_report

    rows: list[dict[str, object]] = []
    for idx in range(20):
        case_id = f"vocal_case_{idx:03d}"
        fields = _trusted_case_fields()
        if idx == 0:
            fields["manual_intervention_count"] = "1"
        rows.append(
            {
                **fields,
                "case_id": case_id,
                "system": "aurik",
                "vocal_focus": "true",
                "material": "vinyl",
                "era": "1970",
                "genre": "vocal",
                "artifact_freedom": "0.98",
                "hpi": "0.20",
                "vqi": "0.80",
                "timbral_fidelity": "0.91",
            }
        )
        for family in ["input_passthrough", "classical_dsp", "sota_ml", "commercial_reference"]:
            rows.append(
                {
                    "case_id": case_id,
                    "system": family,
                    "baseline_family": family,
                    "vocal_focus": "true",
                    "artifact_freedom": "0.96",
                    "hpi": "0.10",
                    "vqi": "0.74",
                }
            )
    csv_path = tmp_path / "results.csv"
    _write_csv(csv_path, rows)
    cfg = json.loads(Path("config/worldclass_kpi_thresholds.json").read_text(encoding="utf-8"))

    report = build_report(_load_rows([csv_path]), cfg)

    assert report["verdict"] == "RECOVERED"
    assert report["fully_automated_operation"]["passed"] is False
    assert report["user_confidence_summary"]["manual_action_required"] is False
    assert any("fully_automated_operation" in violation for violation in report["violations"])
