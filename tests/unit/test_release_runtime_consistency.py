from __future__ import annotations

import json
from pathlib import Path

import pytest

from audit.release_runtime_consistency import consolidate


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.unit
def test_consolidate_detects_contradiction(tmp_path: Path) -> None:
    release = tmp_path / "release_report.json"
    runtime = tmp_path / "runtime_spec_report.json"
    output = tmp_path / "consolidated.json"

    _write_json(
        release,
        {
            "timestamp": "2026-04-12T09:04:50.198941",
            "release_ready": True,
            "compliance_ok": True,
        },
    )
    _write_json(
        runtime,
        {
            "timestamp": "2026-04-14T17:48:50.472080",
            "compliance_ok": False,
            "required_passed": 8,
            "required_total": 9,
        },
    )

    report = consolidate(str(release), str(runtime), str(output))

    assert report["contradiction"] is True
    assert report["final_ready"] is False
    assert "release_runtime_contradiction" in report["reasons"]
    assert report["latest_source"] == "runtime"


def test_consolidate_green_only_when_both_green(tmp_path: Path) -> None:
    release = tmp_path / "release_report.json"
    runtime = tmp_path / "runtime_spec_report.json"
    output = tmp_path / "consolidated.json"

    _write_json(
        release,
        {
            "timestamp": "2026-04-14T18:00:00",
            "release_ready": True,
            "compliance_ok": True,
        },
    )
    _write_json(
        runtime,
        {
            "timestamp": "2026-04-14T18:00:01",
            "compliance_ok": True,
            "required_passed": 9,
            "required_total": 9,
        },
    )

    report = consolidate(str(release), str(runtime), str(output))

    assert report["contradiction"] is False
    assert report["final_ready"] is True
    assert report["latest_source"] == "runtime"


def test_consolidate_handles_missing_runtime_report(tmp_path: Path) -> None:
    release = tmp_path / "release_report.json"
    runtime = tmp_path / "runtime_spec_report.json"
    output = tmp_path / "consolidated.json"

    _write_json(
        release,
        {
            "timestamp": "2026-04-14T18:00:00",
            "release_ready": True,
            "compliance_ok": True,
        },
    )

    report = consolidate(str(release), str(runtime), str(output))

    assert report["final_ready"] is False
    assert "runtime_report_missing_or_invalid" in report["reasons"]


def test_consolidate_accepts_valid_coverage_manifest_reference(tmp_path: Path) -> None:
    release = tmp_path / "release_report.json"
    runtime = tmp_path / "runtime_spec_report.json"
    manifest = tmp_path / "voice_first_guard_coverage_manifest.json"
    output = tmp_path / "consolidated.json"

    _write_json(
        release,
        {
            "timestamp": "2026-05-21T04:20:00",
            "release_ready": True,
            "compliance_ok": True,
        },
    )
    _write_json(
        runtime,
        {
            "timestamp": "2026-05-21T04:20:01",
            "compliance_ok": True,
            "required_passed": 3,
            "required_total": 3,
        },
    )
    _write_json(
        manifest,
        {
            "schema": "voice-first-guard-coverage-manifest.v1",
            "checks": [
                {
                    "id": "decision_quality_learning_guard",
                    "required": True,
                    "guard_class": "critical",
                    "effective_policy": {
                        "max_allowed_prior_drift_ratio": 0.25,
                    },
                    "field_coverage": ["decision_quality.learning_applied"],
                }
            ],
        },
    )

    report = consolidate(
        str(release),
        str(runtime),
        coverage_manifest_path=str(manifest),
        output_path=str(output),
    )

    assert report["coverage_manifest_ok"] is True
    assert report["coverage_drift_ok"] is True
    assert report["coverage_drift_findings"] == []
    assert report["final_ready"] is True


def test_consolidate_detects_guard_coverage_drift_against_baseline(tmp_path: Path) -> None:
    release = tmp_path / "release_report.json"
    runtime = tmp_path / "runtime_spec_report.json"
    manifest = tmp_path / "voice_first_guard_coverage_manifest.json"
    baseline = tmp_path / "voice_first_guard_coverage_manifest_baseline.json"
    output = tmp_path / "consolidated.json"

    _write_json(
        release,
        {
            "timestamp": "2026-05-21T04:20:00",
            "release_ready": True,
            "compliance_ok": True,
        },
    )
    _write_json(
        runtime,
        {
            "timestamp": "2026-05-21T04:20:01",
            "compliance_ok": True,
            "required_passed": 3,
            "required_total": 3,
        },
    )
    _write_json(
        manifest,
        {
            "schema": "voice-first-guard-coverage-manifest.v1",
            "checks": [
                {
                    "id": "decision_quality_learning_guard",
                    "required": True,
                    "guard_class": "critical",
                    "effective_policy": {
                        "max_allowed_prior_drift_ratio": 0.25,
                    },
                    "field_coverage": ["decision_quality.learning_applied"],
                }
            ],
        },
    )
    _write_json(
        baseline,
        {
            "schema": "voice-first-guard-coverage-manifest.v1",
            "checks": [
                {
                    "id": "decision_quality_learning_guard",
                    "required": True,
                    "guard_class": "critical",
                    "effective_policy": {
                        "max_allowed_prior_drift_ratio": 0.20,
                    },
                    "field_coverage": ["decision_quality.learning_applied"],
                }
            ],
        },
    )

    report = consolidate(
        str(release),
        str(runtime),
        coverage_manifest_path=str(manifest),
        coverage_baseline_path=str(baseline),
        output_path=str(output),
    )

    assert report["coverage_manifest_ok"] is True
    assert report["coverage_baseline_ok"] is True
    assert report["coverage_drift_ok"] is False
    assert "effective_policy_changed:decision_quality_learning_guard" in report["coverage_drift_findings"]
    assert report["final_ready"] is False
    assert any(str(reason).startswith("guard_coverage_drift_detected:") for reason in report["reasons"])


def test_consolidate_surfaces_bridge_import_status_non_blocking_failure(tmp_path: Path) -> None:
    release = tmp_path / "release_report.json"
    runtime = tmp_path / "runtime_spec_report.json"
    output = tmp_path / "consolidated.json"

    _write_json(
        release,
        {
            "timestamp": "2026-05-21T05:20:00",
            "release_ready": True,
            "compliance_ok": True,
        },
    )
    _write_json(
        runtime,
        {
            "timestamp": "2026-05-21T05:20:01",
            "compliance_ok": True,
            "required_passed": 3,
            "required_total": 3,
            "checks": [
                {
                    "id": "bridge_import_status_runtime",
                    "required": False,
                    "passed": False,
                    "evidence": "entry[0] startup_check_status.failures=1>0",
                }
            ],
        },
    )

    report = consolidate(str(release), str(runtime), str(output))

    assert report["final_ready"] is True
    assert report["bridge_import_status_present"] is True
    assert report["bridge_import_status_passed"] is False
    assert "startup_check_status.failures" in report["bridge_import_status_evidence"]
    assert "bridge_import_status_runtime_failed_non_blocking" in report["reasons"]


def test_consolidate_handles_missing_bridge_import_status_check(tmp_path: Path) -> None:
    release = tmp_path / "release_report.json"
    runtime = tmp_path / "runtime_spec_report.json"
    output = tmp_path / "consolidated.json"

    _write_json(
        release,
        {
            "timestamp": "2026-05-21T05:30:00",
            "release_ready": True,
            "compliance_ok": True,
        },
    )
    _write_json(
        runtime,
        {
            "timestamp": "2026-05-21T05:30:01",
            "compliance_ok": True,
            "required_passed": 3,
            "required_total": 3,
            "checks": [
                {
                    "id": "decision_quality_learning_guard",
                    "required": True,
                    "passed": True,
                    "evidence": "ok",
                }
            ],
        },
    )

    report = consolidate(str(release), str(runtime), str(output))

    assert report["final_ready"] is True
    assert report["bridge_import_status_present"] is False
    assert report["bridge_import_status_passed"] is None
    assert report["bridge_import_status_evidence"] == ""
