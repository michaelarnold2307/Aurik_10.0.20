"""Unit tests for UAT report generator marker parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from audit.uat_report_generator import ResultStatus, UATReportGenerator

_GATE_FUNCTIONS = (
    "test_no_docker_in_production_paths",
    "test_kmv_batch_audio_correct",
    "test_no_silent_refinement_cancellation",
    "test_progress_counter_consistency",
    "test_pmgg_no_rollback_skipping",
    "test_hybrid_release_mode_determinism",
)


def _synthetic_uat_output(
    restoration: list[str],
    studio: list[str] | None = None,
    gates: list[str] | None = None,
) -> str:
    """Build the pytest -v lines the generator parses (no test execution)."""
    studio = studio if studio is not None else ["PASSED"] * 15
    gates = gates if gates is not None else ["PASSED"] * len(_GATE_FUNCTIONS)
    lines = [
        f"tests/test_uat_acceptance_criteria.py::test_restoration_criteria[R{i}] {status}"
        for i, status in enumerate(restoration, 1)
    ]
    lines.extend(
        f"tests/test_uat_acceptance_criteria.py::test_studio_2026_criteria[S{i}] {status}"
        for i, status in enumerate(studio, 1)
    )
    lines.extend(
        f"tests/test_uat_acceptance_criteria.py::{fn} {status}"
        for fn, status in zip(_GATE_FUNCTIONS, gates, strict=True)
    )
    return "\n".join(lines)


def _summarize(
    tmp_path: Path,
    restoration: list[str],
    gates: list[str] | None = None,
) -> UATReportGenerator:
    generator = UATReportGenerator(output_dir=tmp_path, json_output=tmp_path / "uat.json")
    generator.parse_pytest_output(_synthetic_uat_output(restoration, gates=gates))
    generator.populate_criterion_names()
    generator.compute_summary()
    return generator


@pytest.mark.unit
def test_parse_pytest_output_prefers_machine_readable_marker(tmp_path: Path) -> None:
    generator = UATReportGenerator(output_dir=tmp_path, json_output=tmp_path / "uat.json")
    output = "\n".join(
        [
            'tests/test_uat_acceptance_criteria.py::test_restoration_criteria[R5] UAT_RESULT_JSON:{"criterion_id": "R5", "evidence": "segmentiertes Stereo ok", "kind": "restoration", "notes": "worst segment 2 corr 0.41->0.44", "result": "PASS", "timestamp": ""}',
            "PASSED",
        ]
    )

    generator.parse_pytest_output(output)

    assert len(generator.restoration_results) == 1
    result = generator.restoration_results[0]
    assert result.criterion_id == "R5"
    assert result.status == ResultStatus.PASSED
    assert result.evidence == "segmentiertes Stereo ok"
    assert result.notes == "worst segment 2 corr 0.41->0.44"


def test_scorecard_cell_combines_evidence_and_notes() -> None:
    cell = UATReportGenerator._scorecard_cell(
        "Real-Audio Musical Goals segmentiert gemessen",
        "worst segment 1 goal=artikulation delta=-0.021",
        limit=200,
    )

    assert "segmentiert" in cell
    assert "worst segment 1" in cell


def test_error_criteria_are_counted_and_block_conditional(tmp_path: Path) -> None:
    """ERROR is not PASS and not a harmless SKIP: it must fail closed."""
    generator = _summarize(tmp_path, ["ERROR"] * 15)

    assert generator.summary.restoration_errors == 15
    assert generator.summary.criteria_errors == 15
    assert generator.summary.criteria_passed == 15  # only the studio half ran
    assert generator.summary.criteria_failed == 0
    assert generator.summary.recommendation == "NO-GO"
    assert "nicht auswertbar" in generator.summary.rationale


def test_skipped_criteria_remain_conditional(tmp_path: Path) -> None:
    """A deliberate SKIP is still a documented CONDITIONAL, unlike ERROR."""
    generator = _summarize(tmp_path, ["PASSED"] * 5 + ["SKIPPED"] * 10)

    assert generator.summary.criteria_errors == 0
    assert generator.summary.criteria_skipped == 10
    assert generator.summary.criteria_passed == 20
    assert generator.summary.recommendation == "CONDITIONAL"


def test_missing_criteria_block_conditional(tmp_path: Path) -> None:
    """A half that produced no result lines cannot be sold as CONDITIONAL."""
    generator = UATReportGenerator(output_dir=tmp_path, json_output=tmp_path / "uat.json")
    generator.parse_pytest_output(_synthetic_uat_output([]))
    generator.populate_criterion_names()
    generator.compute_summary()

    assert generator.summary.criteria_errors == 0
    assert generator.summary.criteria_passed == 15
    assert generator.summary.recommendation == "NO-GO"
    assert "ohne Ergebnis" in generator.summary.rationale


def test_gate_error_blocks_conditional(tmp_path: Path) -> None:
    """An unexecutable release gate is not a pass either."""
    generator = _summarize(tmp_path, ["PASSED"] * 15, gates=["ERROR"] + ["PASSED"] * 5)

    assert generator.summary.gates_errors == 1
    assert generator.summary.criteria_errors == 0
    assert generator.summary.recommendation == "NO-GO"
    assert "Release-Gates nicht auswertbar" in generator.summary.rationale


def test_scorecard_and_final_report_render_error_not_skip(tmp_path: Path) -> None:
    generator = _summarize(tmp_path, ["ERROR"] + ["SKIPPED"] * 14)

    scorecard_row = next(
        line for line in generator.generate_scorecard_markdown().splitlines() if line.startswith("| R1 |")
    )
    assert "⚠ ERROR" in scorecard_row
    assert "⊘ SKIP" not in scorecard_row

    report_row = next(line for line in generator.generate_final_report().splitlines() if line.startswith("| R1 |"))
    assert "⚠ ERROR" in report_row
    assert "⊘ SKIPPED" not in report_row
    assert "Total Errors (nicht auswertbar):** 1" in generator.generate_final_report()
