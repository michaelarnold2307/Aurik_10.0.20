from __future__ import annotations

from pathlib import Path

import pytest

from backend.core.real_audio_defect_golden_gate import run_real_audio_defect_golden_gate


def test_real_audio_defect_golden_manifest_passes_gate() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    manifest = repo_root / "audit" / "real_audio_defect_golden_manifest.json"

    if not (repo_root / "test_audio").is_dir():
        pytest.skip("test_audio-Korpus nicht vorhanden (gitignored)")

    report = run_real_audio_defect_golden_gate(manifest_path=manifest, repo_root=repo_root)

    assert report.scanned_cases >= 8
    assert report.skipped_cases == []
    assert report.gate.passed is True
    assert report.gate.recall == 1.0
    assert report.gate.precision == 1.0
    assert report.gate.locality_recall == 1.0
    assert report.gate.fail_reasons == ()
