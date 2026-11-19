from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def _run_gate(tmp_path: Path, payload: dict, *extra_args: str) -> subprocess.CompletedProcess[str]:
    in_file = tmp_path / "panel_results.json"
    out_file = tmp_path / "freude_gate_report.json"
    in_file.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")

    cmd = [
        sys.executable,
        "scripts/freude_gate_check.py",
        "--input",
        str(in_file),
        "--output",
        str(out_file),
        *extra_args,
    ]
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


@pytest.mark.unit
def test_freude_gate_pass(tmp_path: Path):
    items = [
        {"item_id": f"s{i}", "mushra": 82.0, "enjoyment": 4.4, "fatigue": 2.0, "artifact_flag": False}
        for i in range(20)
    ]
    res = _run_gate(tmp_path, {"items": items})
    assert res.returncode == 0, res.stdout + res.stderr
    report = json.loads((tmp_path / "freude_gate_report.json").read_text(encoding="utf-8"))
    assert report["passed"] is True


def test_freude_gate_fail_on_enjoyment(tmp_path: Path):
    items = [
        {"item_id": f"s{i}", "mushra": 83.0, "enjoyment": 3.8, "fatigue": 2.0, "artifact_flag": False}
        for i in range(20)
    ]
    res = _run_gate(tmp_path, {"items": items})
    assert res.returncode == 0
    report = json.loads((tmp_path / "freude_gate_report.json").read_text(encoding="utf-8"))
    assert report["passed"] is False
    assert report["mode"] == "improve"
    assert isinstance(report.get("recommendations"), list)
    assert len(report["recommendations"]) >= 1


def test_freude_gate_fail_on_enjoyment_in_enforce_mode(tmp_path: Path):
    items = [
        {"item_id": f"s{i}", "mushra": 83.0, "enjoyment": 3.8, "fatigue": 2.0, "artifact_flag": False}
        for i in range(20)
    ]
    res = _run_gate(tmp_path, {"items": items}, "--enforce")
    assert res.returncode == 1
    report = json.loads((tmp_path / "freude_gate_report.json").read_text(encoding="utf-8"))
    assert report["passed"] is False
    assert report["mode"] == "enforce"


def test_freude_gate_fail_on_min_items(tmp_path: Path):
    items = [
        {"item_id": "s0", "mushra": 90.0, "enjoyment": 4.8, "fatigue": 1.7, "artifact_flag": False} for _ in range(5)
    ]
    res = _run_gate(tmp_path, {"items": items})
    assert res.returncode == 2
    assert "Zu wenige Items" in (res.stdout + res.stderr)
