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
