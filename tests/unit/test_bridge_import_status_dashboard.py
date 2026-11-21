from __future__ import annotations

import json
from pathlib import Path

import pytest

from audit.bridge_import_status_dashboard import (
    build_bridge_import_status_summary,
    format_one_line,
    main,
)


@pytest.mark.unit
def test_build_bridge_import_status_summary_warning_case() -> None:
    consolidated = {
        "bridge_import_status_present": True,
        "bridge_import_status_passed": False,
        "bridge_import_status_evidence": "entry[0] startup_check_status.failures=1>0",
        "reasons": ["bridge_import_status_runtime_failed_non_blocking"],
        "final_ready": True,
    }

    summary = build_bridge_import_status_summary(consolidated)

    assert summary["bridge_import_status_present"] is True
    assert summary["bridge_import_status_passed"] is False
    assert summary["reason_flagged"] is True
    assert summary["severity"] == "warning"


def test_build_summary_does_not_coerce_malformed_status_values() -> None:
    summary = build_bridge_import_status_summary(
        {
            "bridge_import_status_present": "false",
            "bridge_import_status_passed": "false",
            "final_ready": "false",
        }
    )

    assert summary["bridge_import_status_present"] is False
    assert summary["bridge_import_status_passed"] is None
    assert summary["final_ready"] is False
    assert summary["severity"] == "unknown"


def test_format_one_line_contains_expected_tokens() -> None:
    summary = {
        "bridge_import_status_present": True,
        "bridge_import_status_passed": True,
        "bridge_import_status_evidence": "ok",
        "reason_flagged": False,
        "final_ready": True,
        "severity": "ok",
    }

    line = format_one_line(summary)

    assert "BRIDGE_IMPORT_STATUS" in line
    assert "severity=ok" in line
    assert "status=passed" in line
    assert "final_ready=True" in line


def test_format_one_line_escapes_multiline_evidence() -> None:
    evidence = 'ok"\nBRIDGE_IMPORT_STATUS severity=ok'
    line = format_one_line(
        {
            "bridge_import_status_present": True,
            "bridge_import_status_passed": True,
            "bridge_import_status_evidence": evidence,
            "reason_flagged": False,
            "final_ready": True,
            "severity": "ok",
        }
    )

    assert "\n" not in line
    assert json.loads(line.split(" evidence=", maxsplit=1)[1]) == evidence


def test_main_writes_summary_and_returns_zero(tmp_path: Path) -> None:
    consolidated = tmp_path / "consolidated.json"
    output = tmp_path / "summary.json"
    consolidated.write_text(
        json.dumps(
            {
                "bridge_import_status_present": True,
                "bridge_import_status_passed": True,
                "bridge_import_status_evidence": "all good",
                "reasons": [],
                "final_ready": True,
            }
        ),
        encoding="utf-8",
    )

    rc = main(["--consolidated", str(consolidated), "--output", str(output)])

    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["severity"] == "ok"


def test_main_fail_on_warning_returns_one(tmp_path: Path) -> None:
    consolidated = tmp_path / "consolidated.json"
    output = tmp_path / "summary.json"
    consolidated.write_text(
        json.dumps(
            {
                "bridge_import_status_present": True,
                "bridge_import_status_passed": False,
                "bridge_import_status_evidence": "entry[0] ml_memory_budget_import_status.failures=2>0",
                "reasons": ["bridge_import_status_runtime_failed_non_blocking"],
                "final_ready": True,
            }
        ),
        encoding="utf-8",
    )

    rc = main(["--consolidated", str(consolidated), "--output", str(output), "--fail-on-warning"])

    assert rc == 1


def test_main_fail_on_warning_rejects_unknown_bridge_status(tmp_path: Path) -> None:
    consolidated = tmp_path / "consolidated.json"
    consolidated.write_text(
        json.dumps({"bridge_import_status_present": True, "bridge_import_status_passed": None}),
        encoding="utf-8",
    )

    rc = main(["--consolidated", str(consolidated), "--output", str(tmp_path / "summary.json"), "--fail-on-warning"])

    assert rc == 1


def test_main_reports_missing_input_path(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["--consolidated", str(tmp_path / "missing.json"), "--output", str(tmp_path / "summary.json")])

    assert rc == 1
    assert "could not read consolidated report" in capsys.readouterr().err


def test_main_reports_non_utf8_input(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    consolidated = tmp_path / "consolidated.json"
    consolidated.write_bytes(b"\xff")

    rc = main(["--consolidated", str(consolidated), "--output", str(tmp_path / "summary.json")])

    assert rc == 1
    assert "not valid UTF-8" in capsys.readouterr().err


def test_main_rejects_output_overwriting_input(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    consolidated = tmp_path / "consolidated.json"
    original = json.dumps({"bridge_import_status_present": True, "bridge_import_status_passed": True})
    consolidated.write_text(original, encoding="utf-8")

    rc = main(["--consolidated", str(consolidated), "--output", str(consolidated)])

    assert rc == 1
    assert consolidated.read_text(encoding="utf-8") == original
    assert "same file" in capsys.readouterr().err
