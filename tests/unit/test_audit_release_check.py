from __future__ import annotations

import json
from pathlib import Path

import pytest

from audit import release_check


@pytest.mark.unit
def test_calculate_release_score_full_compliance_reaches_10() -> None:
    audit_data = [{"results": {"gate_a": True, "gate_b": True}}]
    score = release_check.calculate_release_score(True, [], audit_data)
    assert score == 10.0


def test_calculate_release_score_penalizes_failures_and_missing_docs() -> None:
    audit_data = [{"results": {"gate_a": True, "gate_b": False}}]
    changes = [
        "Quality-Gate 'gate_b' nicht bestanden.",
        "Quality-Gate 'gate_x' nicht in Dokumentation.",
    ]
    score = release_check.calculate_release_score(False, changes, audit_data)
    assert score < 5.0


def test_check_compliance_detects_failed_gate_and_missing_policy(tmp_path: Path) -> None:
    gates_doc = tmp_path / "QUALITY_GATES.md"
    gates_doc.write_text("gate_a\n", encoding="utf-8")
    policy_file = tmp_path / "missing_policy.py"

    compliance_ok, changes = release_check.check_compliance(
        audit_data=[{"results": {"gate_a": True, "gate_b": False}}],
        doc_gates_path=str(gates_doc),
        doc_policy_path=str(policy_file),
    )

    assert compliance_ok is False
    assert any("nicht bestanden" in change for change in changes)
    assert any("Policy-Datei fehlt" in change for change in changes)


def test_generate_release_report_writes_json(tmp_path: Path) -> None:
    report_path = tmp_path / "release_report.json"

    report = release_check.generate_release_report(
        compliance_ok=True,
        changes=[],
        audit_data=[{"results": {"gate_a": True}}],
        output_path=str(report_path),
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["score"] == 10.0
    assert payload["release_ready"] is True
    assert report["release_ready"] is True


def test_gate_stats_reads_nested_quality_gates_and_vocal_checks() -> None:
    audit_data = [
        {
            "vocal_quality_check": {"authentizitaet": True, "transparenz": False},
            "scores": {"quality_gates": {"lufs": True}},
            "features": {"quality_gates": {"tp": False}},
        }
    ]

    total, passed = release_check._gate_stats(audit_data, include_diagnostic_gates=True)
    assert total == 4
    assert passed == 2


def test_gate_stats_counts_negative_release_status_as_failed_gate() -> None:
    audit_data = [{"release_result": {"status": "release_check_not_available"}}]
    total, passed = release_check._gate_stats(audit_data, include_diagnostic_gates=True)
    assert total == 1
    assert passed == 0


def test_gate_stats_can_ignore_diagnostic_gates_explicitly() -> None:
    audit_data = [
        {
            "vocal_quality_check": {"authentizitaet": False},
            "scores": {"quality_gates": {"lufs": False}},
            "release_result": {"status": "release_check_not_available"},
        }
    ]
    total, passed = release_check._gate_stats(audit_data, include_diagnostic_gates=False)
    assert total == 0
    assert passed == 0


def test_check_release_returns_structured_status(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit_trail.json"
    audit_path.write_text(json.dumps([{"results": {"gate_a": True}}]), encoding="utf-8")
    gates_doc = tmp_path / "QUALITY_GATES.md"
    gates_doc.write_text("gate_a\n", encoding="utf-8")
    policy_file = tmp_path / "policy_engine.py"
    policy_file.write_text("# ok\n", encoding="utf-8")

    result = release_check.check_release(
        audit_path=str(audit_path),
        gates_doc=str(gates_doc),
        policy_path=str(policy_file),
        include_diagnostic_gates=False,
        output_path=str(tmp_path / "release_report.json"),
    )

    assert result["status"] == "release_ready"
    assert result["release_ready"] is True
    assert result["score"] == 10.0


def test_check_release_strict_mode_blocks_if_gate_coverage_is_too_low(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit_trail.json"
    audit_path.write_text(json.dumps([{"results": {"gate_a": True}}]), encoding="utf-8")
    gates_doc = tmp_path / "QUALITY_GATES.md"
    gates_doc.write_text("gate_a\n", encoding="utf-8")
    policy_file = tmp_path / "policy_engine.py"
    policy_file.write_text("# ok\n", encoding="utf-8")

    result = release_check.check_release(
        audit_path=str(audit_path),
        gates_doc=str(gates_doc),
        policy_path=str(policy_file),
        include_diagnostic_gates=True,
        output_path=str(tmp_path / "release_report_strict.json"),
    )

    assert result["status"] == "blocked"
    assert result["release_ready"] is False
    assert any("Audit-Abdeckung zu gering" in change for change in result["changes"])


def test_check_compliance_strict_mode_blocks_when_voice_first_blockers_missing(tmp_path: Path) -> None:
    gates_doc = tmp_path / "QUALITY_GATES.md"
    gates_doc.write_text("gate_a\ngate_b\ngate_c\ngate_d\ngate_e\n", encoding="utf-8")
    policy_file = tmp_path / "policy_engine.py"
    policy_file.write_text("# ok\n", encoding="utf-8")

    audit_data = [
        {
            "results": {
                "gate_a": True,
                "gate_b": True,
                "gate_c": True,
                "gate_d": True,
                "gate_e": True,
            },
            "scores": {"media_characteristics": {"vocal": True}},
            "vocal_quality_check": {"authentizitaet": True},
        }
    ]

    compliance_ok, changes = release_check.check_compliance(
        audit_data=audit_data,
        doc_gates_path=str(gates_doc),
        doc_policy_path=str(policy_file),
        include_diagnostic_gates=True,
    )

    assert compliance_ok is False
    assert any("Voice-First-Blocker fehlen" in change for change in changes)


def test_check_compliance_strict_mode_accepts_present_voice_first_blockers(tmp_path: Path) -> None:
    gates_doc = tmp_path / "QUALITY_GATES.md"
    gates_doc.write_text("gate_a\ngate_b\ngate_c\ngate_d\ngate_e\n", encoding="utf-8")
    policy_file = tmp_path / "policy_engine.py"
    policy_file.write_text("# ok\n", encoding="utf-8")

    audit_data = [
        {
            "results": {
                "gate_a": True,
                "gate_b": True,
                "gate_c": True,
                "gate_d": True,
                "gate_e": True,
            },
            "scores": {"media_characteristics": {"vocal": True}},
            "vocal_quality_check": {
                "vqi": True,
                "formant_integrity": True,
                "vibrato_depth_preserved": True,
                "micro_dynamic_correlation": True,
            },
        }
    ]

    compliance_ok, changes = release_check.check_compliance(
        audit_data=audit_data,
        doc_gates_path=str(gates_doc),
        doc_policy_path=str(policy_file),
        include_diagnostic_gates=True,
    )

    assert compliance_ok is True
    assert not any("Voice-First-Blocker" in change for change in changes)
