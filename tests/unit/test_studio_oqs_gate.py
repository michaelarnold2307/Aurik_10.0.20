"""OQS diagnostic threshold tests; the proxy must not decide export status."""

from __future__ import annotations

from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def _make_mushra_result(score: float):
    """Erstellt ein minimales MushraResult-Objekt mit gegebenem Score."""
    from backend.core.mushra_evaluator import MushraResult

    return MushraResult(
        mushra_score=score,
        grade="Good",
        itu_grade="B",
        nsim=0.85,
        musical_goals={},
        anchor_score=50.0,
        hidden_ref_score=score,
        details={},
    )


@pytest.mark.unit
def test_mushra_result_passes_at_exactly_88():
    """passes_mushra_threshold(88.0) bei Score 88.0 → True (>=)."""
    result = _make_mushra_result(88.0)
    assert result.passes_mushra_threshold(88.0) is True


def test_mushra_result_fails_below_88():
    """passes_mushra_threshold(88.0) bei Score 87.9 → False."""
    result = _make_mushra_result(87.9)
    assert result.passes_mushra_threshold(88.0) is False


def test_mushra_result_passes_at_exactly_80():
    """passes_mushra_threshold(80.0) bei Score 80.0 → True (Restoration-Grenze)."""
    result = _make_mushra_result(80.0)
    assert result.passes_mushra_threshold(80.0) is True


def test_mushra_result_fails_below_80():
    """passes_mushra_threshold(80.0) bei Score 79.9 → False."""
    result = _make_mushra_result(79.9)
    assert result.passes_mushra_threshold(80.0) is False


def test_oqs_reference_threshold_cannot_become_an_export_veto() -> None:
    source_path = Path(__file__).resolve().parents[2] / "backend" / "core" / "unified_restorer_v3.py"
    source = source_path.read_text(encoding="utf-8")

    assert "STUDIO_OQS_GATE_FAIL" not in source
    assert '"threshold_role": "report_only_unvalidated"' in source
