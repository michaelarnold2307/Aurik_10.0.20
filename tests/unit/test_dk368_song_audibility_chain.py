"""Regression contract for full-song audibility evidence after chunk assembly."""

from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "backend" / "core" / "unified_restorer_v3.py"


def _src() -> str:
    return SRC.read_text(encoding="utf-8")


def test_empty_scan_cannot_claim_full_song_coverage() -> None:
    from backend.core.defect_audibility_gate import evaluate_defect_audibility

    rep_ok = evaluate_defect_audibility({}, material_key="vinyl", post_scan_ran=True, scan_scope="whole_song")
    assert rep_ok.evidence_state == "coverage_incomplete", rep_ok.evidence_state
    assert rep_ok.gate_passed is True
    assert rep_ok.gate_verified is False
    assert rep_ok.coverage_complete is False

    rep_missing = evaluate_defect_audibility({}, material_key="vinyl", post_scan_ran=False)
    assert rep_missing.evidence_state == "scan_missing"
    assert rep_missing.gate_verified is False


def test_full_song_scan_runs_after_chunk_assembly() -> None:
    src = _src()
    assert "_song_post_scan = self.defect_scanner.scan(" in src
    assert "_evaluate_full_song_audibility(" in src
    assert "Vollsong-Audibility-Scan nach Chunk-Assembly" in src


def test_full_song_report_is_written_to_export_metadata() -> None:
    src = _src()
    assert "_write_song_audibility_metadata(_meta, _song_report)" in src


def test_scan_failure_is_explicitly_fail_closed() -> None:
    src = _src()
    assert "post_scan_ran=False," in src
    assert "Evidenz bleibt ungeprüft" in src
