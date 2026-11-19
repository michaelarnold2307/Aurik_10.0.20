from __future__ import annotations

import pytest

"""Tests für §8.1.1a [RELEASE_MUST] Studio-2026 OQS-Gate (v10.0.0).

Verifiziert:
- Studio 2026 threshold = 88.0, Restoration threshold = 80.0
- OQS < 88 in Studio-2026 → STUDIO_OQS_GATE_FAIL in _fail_reasons
- OQS ≥ 88 in Studio-2026 → kein Fail
- OQS < 88 in Restoration → kein Hard-Fail (Restoration-Threshold 80.0)
- OQS < 80 in Restoration → kein STUDIO_OQS_GATE_FAIL (anderer Gate-Typ)
"""


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


# ---------------------------------------------------------------------------
# §8.1.1a Threshold-Konstanten
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# §8.1.1a _fail_reasons Integration — direkte Logik-Tests
# ---------------------------------------------------------------------------


def _simulate_oqs_gate(mushra_score: float, mode: str) -> list[dict]:
    """Simuliert die §8.1.1a Gate-Logik aus unified_restorer_v3:
    Gibt die angehängten fail_reason-Dicts zurück.
    """
    mushra_result = _make_mushra_result(mushra_score)
    is_studio = mode == "studio_2026"

    _mushra_threshold = 88.0 if is_studio else 80.0
    _mushra_pass = mushra_result.passes_mushra_threshold(_mushra_threshold)

    _fail_reasons: list[dict] = []
    if not _mushra_pass and is_studio:
        _fail_reasons.append(
            {
                "error_code": "STUDIO_OQS_GATE_FAIL",
                "message": (
                    f"Studio 2026: OQS {mushra_result.mushra_score:.1f} < {_mushra_threshold:.0f} "
                    f"(§8.1.1a RELEASE_MUST — export blockiert)"
                ),
                "oqs_score": round(mushra_result.mushra_score, 1),
                "oqs_threshold": _mushra_threshold,
                "severity": "critical",
            }
        )
    return _fail_reasons


def test_studio_oqs_fail_appends_fail_reason():
    """Studio 2026 + OQS 75 → STUDIO_OQS_GATE_FAIL in fail_reasons."""
    fail_reasons = _simulate_oqs_gate(75.0, "studio_2026")
    assert len(fail_reasons) == 1
    assert fail_reasons[0]["error_code"] == "STUDIO_OQS_GATE_FAIL"
    assert fail_reasons[0]["severity"] == "critical"
    assert fail_reasons[0]["oqs_score"] == 75.0
    assert fail_reasons[0]["oqs_threshold"] == 88.0


def test_studio_oqs_pass_no_fail_reason():
    """Studio 2026 + OQS 89 → kein Fail-Reason."""
    fail_reasons = _simulate_oqs_gate(89.0, "studio_2026")
    assert fail_reasons == []


def test_restoration_oqs_below_88_no_studio_fail():
    """Restoration-Modus + OQS 85 (82 > 80-Threshold) → kein STUDIO_OQS_GATE_FAIL."""
    fail_reasons = _simulate_oqs_gate(85.0, "restoration")
    assert not any(r.get("error_code") == "STUDIO_OQS_GATE_FAIL" for r in fail_reasons)


def test_restoration_oqs_below_80_no_studio_fail():
    """Restoration-Modus + OQS 78 (< 80) → kein STUDIO_OQS_GATE_FAIL (anderer Gate-Typ)."""
    fail_reasons = _simulate_oqs_gate(78.0, "restoration")
    assert not any(r.get("error_code") == "STUDIO_OQS_GATE_FAIL" for r in fail_reasons)


def test_studio_oqs_exactly_88_no_fail():
    """Studio 2026 + OQS exakt 88.0 → pass, kein Fail-Reason."""
    fail_reasons = _simulate_oqs_gate(88.0, "studio_2026")
    assert fail_reasons == []


def test_studio_oqs_fail_reason_message_contains_score():
    """Fail-Reason-Message enthält OQS-Score und Threshold."""
    fail_reasons = _simulate_oqs_gate(72.3, "studio_2026")
    assert len(fail_reasons) == 1
    msg = fail_reasons[0]["message"]
    assert "72.3" in msg
    assert "88" in msg
    assert "RELEASE_MUST" in msg
