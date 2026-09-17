from __future__ import annotations

"""Normative tests for recovery-gated export behavior."""


import numpy as np
import pytest

from backend.core.export_workflow import export_audio


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_export_requires_recovery_metadata_when_quality_gate_fails(tmp_path) -> None:
    audio = np.zeros(48_000, dtype=np.float32)
    quality_gate = {
        "passed": False,
        "fail_reason": "PQS unter Mindestschwelle",
        "fail_reasons": [
            {
                "component": "quality_gate",
                "error_code": "PQS_GATE_FAILED",
                "severity": "blocked",
                "exc_msg": "PQS unter Mindestschwelle",
            }
        ],
        "required_gates": ["musical_goals", "pqs", "oqs"],
    }

    path = export_audio(audio, 48_000, "best_effort", quality_gate=quality_gate, output_dir=str(tmp_path))
    assert path.endswith(".wav")

    meta_path = tmp_path / "best_effort.json"
    assert meta_path.exists()
    payload = meta_path.read_text(encoding="utf-8")
    assert '"quality_gate_passed": false' in payload
    assert '"export_strategy": "degraded"' in payload


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_export_allowed_after_recovery_attempt_with_degraded_status(tmp_path) -> None:
    audio = np.zeros(48_000, dtype=np.float32)
    quality_gate = {
        "passed": False,
        "fail_reason": "PQS unter Mindestschwelle",
        "fail_reasons": [
            {
                "component": "quality_gate",
                "error_code": "PQS_GATE_FAILED",
                "severity": "blocked",
                "exc_msg": "PQS unter Mindestschwelle",
            }
        ],
        "required_gates": ["musical_goals", "pqs", "oqs"],
        "recovery_attempted": True,
        "best_possible_reached": False,
    }

    path = export_audio(audio, 48_000, "degraded", quality_gate=quality_gate, output_dir=str(tmp_path))
    assert path.endswith(".wav")

    meta_path = tmp_path / "degraded.json"
    assert meta_path.exists()
    payload = meta_path.read_text(encoding="utf-8")
    assert "quality_gate_passed" in payload
    assert "quality_gate_fail_reason" in payload
    assert "quality_gate_degradation_status" in payload
    assert "quality_gate_fail_reasons" in payload
    assert "PQS_GATE_FAILED" in payload
    assert '"export_strategy": "degraded"' in payload


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_export_allowed_when_quality_gate_passes(tmp_path) -> None:
    audio = np.zeros(48_000, dtype=np.float32)
    quality_gate = {
        "passed": True,
        "required_gates": ["musical_goals", "pqs", "oqs"],
    }

    path = export_audio(audio, 48_000, "ok", quality_gate=quality_gate, output_dir=str(tmp_path))
    assert path.endswith(".wav")

    meta_path = tmp_path / "ok.json"
    assert meta_path.exists()
    payload = meta_path.read_text(encoding="utf-8")
    assert '"quality_gate_degradation_status": "ok"' in payload
