from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_phase_harmonization_audit_generates_report(tmp_path: Path):
    out_file = tmp_path / "phase_harmonization_audit.json"
    cmd = [
        sys.executable,
        "scripts/phase_harmonization_audit.py",
        "--output",
        str(out_file),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0, res.stdout + res.stderr

    report = json.loads(out_file.read_text(encoding="utf-8"))
    assert report["audit"] == "phase_harmonization_v1"
    summary = report["summary"]
    assert int(summary["total_phases"]) >= 50
    assert 0.0 <= float(summary["coverage_ratio"]) <= 1.0
    assert isinstance(report["rows"], list)
    assert len(report["rows"]) == int(summary["total_phases"])
    assert isinstance(report["recommendations"], list)
