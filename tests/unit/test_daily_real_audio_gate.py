from __future__ import annotations

import json
from pathlib import Path

import pytest

from audit.daily_real_audio_gate import (
    build_daily_points,
    build_status,
    generate_daily_real_audio_gate_report,
    load_uat_runs,
)


def _write_uat(
    path: Path,
    *,
    generated_at: str,
    gates_passed: int,
    gates_total: int,
    recommendation: str,
    r_statuses: dict[str, str],
) -> None:
    payload = {
        "generated_at": generated_at,
        "summary": {
            "gates_passed": gates_passed,
            "gates_total": gates_total,
            "recommendation": recommendation,
        },
        "restoration_criteria": [{"criterion_id": cid, "status": status} for cid, status in r_statuses.items()],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@pytest.mark.unit
def test_load_uat_runs_sorted_by_timestamp(tmp_path: Path) -> None:
    _write_uat(
        tmp_path / "uat_results_2026-04-12.json",
        generated_at="2026-04-12T10:00:00",
        gates_passed=7,
        gates_total=7,
        recommendation="GO",
        r_statuses={
            "R5": "PASSED",
            "R6": "PASSED",
            "R7": "PASSED",
            "R8": "PASSED",
            "R9": "PASSED",
            "R10": "PASSED",
            "R11": "PASSED",
            "R12": "PASSED",
        },
    )
    _write_uat(
        tmp_path / "uat_results_2026-04-11.json",
        generated_at="2026-04-11T10:00:00",
        gates_passed=6,
        gates_total=7,
        recommendation="CONDITIONAL",
        r_statuses={
            "R5": "PASSED",
            "R6": "FAILED",
            "R7": "PASSED",
            "R8": "PASSED",
            "R9": "PASSED",
            "R10": "PASSED",
            "R11": "PASSED",
            "R12": "PASSED",
        },
    )

    runs = load_uat_runs(tmp_path)
    assert len(runs) == 2
    assert runs[0][0].name == "uat_results_2026-04-11.json"
    assert runs[1][0].name == "uat_results_2026-04-12.json"


def test_build_status_marks_ready_when_latest_is_fully_passing(tmp_path: Path) -> None:
    _write_uat(
        tmp_path / "uat_results_2026-04-11.json",
        generated_at="2026-04-11T10:00:00",
        gates_passed=6,
        gates_total=7,
        recommendation="CONDITIONAL",
        r_statuses={
            "R5": "PASSED",
            "R6": "FAILED",
            "R7": "PASSED",
            "R8": "PASSED",
            "R9": "PASSED",
            "R10": "PASSED",
            "R11": "PASSED",
            "R12": "PASSED",
        },
    )
    _write_uat(
        tmp_path / "uat_results_2026-04-12.json",
        generated_at="2026-04-12T10:00:00",
        gates_passed=7,
        gates_total=7,
        recommendation="GO",
        r_statuses={
            "R5": "PASSED",
            "R6": "PASSED",
            "R7": "PASSED",
            "R8": "PASSED",
            "R9": "PASSED",
            "R10": "PASSED",
            "R11": "PASSED",
            "R12": "PASSED",
        },
    )

    points = build_daily_points(load_uat_runs(tmp_path))
    status = build_status(points)

    assert status["status"] == "ready"
    assert status["latest"]["gates"]["passed"] == 7
    assert status["latest"]["r5_r12"]["passed"] == 8


def test_generate_daily_real_audio_gate_report_writes_outputs(tmp_path: Path) -> None:
    _write_uat(
        tmp_path / "uat_results_2026-04-12.json",
        generated_at="2026-04-12T10:00:00",
        gates_passed=7,
        gates_total=7,
        recommendation="GO",
        r_statuses={
            "R5": "PASSED",
            "R6": "PASSED",
            "R7": "PASSED",
            "R8": "PASSED",
            "R9": "PASSED",
            "R10": "PASSED",
            "R11": "PASSED",
            "R12": "PASSED",
        },
    )

    out_json = tmp_path / "daily_status.json"
    out_md = tmp_path / "daily_status.md"

    status = generate_daily_real_audio_gate_report(
        audit_dir=tmp_path,
        output_json=out_json,
        output_md=out_md,
    )

    assert status["status"] == "ready"
    assert out_json.exists()
    assert out_md.exists()

    text = out_md.read_text(encoding="utf-8")
    assert "Daily Real-Audio-Gate Status" in text
    assert "R5-R12" in text
