from __future__ import annotations

import json
from pathlib import Path

import pytest

from audit.build_voice_first_audit_snapshot import build_snapshot


@pytest.mark.unit
def test_build_snapshot_normalizes_legacy_voice_first_keys(tmp_path: Path) -> None:
    src = tmp_path / "audit_trail.json"
    src.write_text(
        json.dumps(
            [
                {
                    "timestamp": "2026-05-20T11:00:00",
                    "scores": {"media_characteristics": {"vocal": True}},
                    "vocal_quality_check": {
                        "authentizitaet": True,
                        "klarheit": True,
                        "expressivitaet": True,
                        "emotionalitaet": True,
                        "transparenz": True,
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "voice_first.json"

    snapshot = build_snapshot(str(src), str(out), max_entries=1)

    assert len(snapshot) == 1
    saved = json.loads(out.read_text(encoding="utf-8"))
    entry = saved[0]
    vqc = entry["vocal_quality_check"]
    assert vqc["vqi"] is True
    assert vqc["formant_integrity"] is True
    assert vqc["vibrato_depth_preserved"] is True
    assert vqc["micro_dynamic_correlation"] is True
    assert entry["quality_gate_passed"] is True


def test_build_snapshot_returns_empty_when_no_vocal_entries(tmp_path: Path) -> None:
    src = tmp_path / "audit_trail.json"
    src.write_text(
        json.dumps(
            [
                {"timestamp": "2026-05-20T11:00:00", "results": {"gate_a": True}},
                {"timestamp": "2026-05-20T11:00:01", "scores": {"Music": 0.2}},
            ]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "voice_first.json"

    snapshot = build_snapshot(str(src), str(out), max_entries=1)

    assert snapshot == []
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved == []


def test_build_snapshot_normalizes_decision_quality_aliases(tmp_path: Path) -> None:
    src = tmp_path / "audit_trail.json"
    src.write_text(
        json.dumps(
            [
                {
                    "timestamp": "2026-05-21T08:00:00",
                    "scores": {"media_characteristics": {"vocal": True}},
                    "vocal_quality_check": {
                        "vqi": True,
                        "formant_integrity": True,
                        "vibrato_depth_preserved": True,
                        "micro_dynamic_correlation": True,
                    },
                    "learning_decision": {
                        "learn_applied": True,
                        "causal_confidence": 0.42,
                        "drift_ratio": 0.11,
                        "stability_score": 0.89,
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "voice_first.json"

    snapshot = build_snapshot(str(src), str(out), max_entries=1)

    assert len(snapshot) == 1
    saved = json.loads(out.read_text(encoding="utf-8"))
    dq = saved[0].get("decision_quality", {})
    assert dq.get("learning_applied") is True
    assert dq.get("causal_credit_confidence") == 0.42
    assert dq.get("prior_drift_ratio") == 0.11
    assert dq.get("decision_stability_score") == 0.89


def test_build_snapshot_synthesizes_legacy_decision_quality_when_missing(tmp_path: Path) -> None:
    src = tmp_path / "audit_trail.json"
    src.write_text(
        json.dumps(
            [
                {
                    "timestamp": "2026-05-21T09:00:00",
                    "scores": {"media_characteristics": {"vocal": True}},
                    "vocal_quality_check": {
                        "vqi": True,
                        "formant_integrity": True,
                        "vibrato_depth_preserved": True,
                        "micro_dynamic_correlation": True,
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "voice_first.json"

    snapshot = build_snapshot(str(src), str(out), max_entries=1)

    assert len(snapshot) == 1
    saved = json.loads(out.read_text(encoding="utf-8"))
    dq = saved[0].get("decision_quality", {})
    assert dq.get("learning_applied") is False
    assert dq.get("causal_credit_confidence") == 0.0
    assert dq.get("prior_drift_ratio") == 0.0
    assert dq.get("decision_stability_score") == 1.0
    assert dq.get("legacy_bridge") is True
