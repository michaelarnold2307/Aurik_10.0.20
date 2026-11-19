"""Regressionstest: Audibility-Gate im PhaseSkipper (§4 Hörordnung, Ebene 2).

Prinzip: Ein Defekt wird nur repariert, wenn er ÜBER der Maskierungsschwelle
hörbar ist („Mess-Null" ≠ „Maskierungsschwelle"). Liegt der Ziel-Defekt eines
reparierenden Schritts unter der Schwelle (maskiert/unhörbar), wird die Phase
übersprungen — kein Overprocessing an unhörbaren Stellen (Qualität ↑,
Performance ↑). Verdrahtet in ``PhaseSkipper.analyze_pipeline(audibility=...)``.
"""

from __future__ import annotations

from backend.core.defect_analysis import SourceMedium
from backend.core.phase_skipping import (
    _PHASE_TARGET_DEFECT,
    PhaseSkipper,
    ProcessingPhase,
)


class _FakeDefectAnalysis:
    """Minimaler Defekt-Analysen-Stand für die _should_skip_*-Heuristiken."""

    noise_floor_db = -70.0
    has_hiss = True
    has_hum = True
    has_aliasing = True
    has_codec_artifacts = True
    clipping_percentage = 5.0
    click_count = 50
    dropout_count = 10
    medium = SourceMedium.VINYL


def test_inaudible_target_defect_skips_phase() -> None:
    """Unhörbarer Ziel-Defekt → zugehörige Phase wird übersprungen (§4)."""
    p = PhaseSkipper()
    d = p.analyze_pipeline(_FakeDefectAnalysis(), audibility={"noise": False})
    assert d[ProcessingPhase.PHASE_2_DENOISE].skip is True
    assert "Maskierungsschwelle" in d[ProcessingPhase.PHASE_2_DENOISE].reason


def test_audible_target_defect_runs_phase() -> None:
    """Hörbarer Ziel-Defekt → Phase läuft (Reparatur ist gerechtfertigt)."""
    p = PhaseSkipper()
    d = p.analyze_pipeline(_FakeDefectAnalysis(), audibility={"noise": True, "hum": True})
    assert d[ProcessingPhase.PHASE_2_DENOISE].skip is False
    assert d[ProcessingPhase.PHASE_6_DEHUM].skip is False


def test_no_audibility_preserves_legacy_behaviour() -> None:
    """audibility=None → detektionsbasiertes Bestandsverhalten (nicht brechend)."""
    p = PhaseSkipper()
    legacy = p.analyze_pipeline(_FakeDefectAnalysis())
    gated = p.analyze_pipeline(_FakeDefectAnalysis(), audibility=None)
    assert legacy == gated


def test_missing_entry_fails_safe_to_detection() -> None:
    """Fehlt ein Ziel-Defekt im Mapping → detektionsbasiert (fail-safe §V6 (copilot-instructions.md))."""
    p = PhaseSkipper()
    d = p.analyze_pipeline(_FakeDefectAnalysis(), audibility={"unrelated": False})
    assert d[ProcessingPhase.PHASE_2_DENOISE].skip is False


def test_forensic_phase_never_gated() -> None:
    """PHASE_1_FORENSIC ist kein Reparatur-Schritt → nie vom Gate geskippt."""
    p = PhaseSkipper()
    d = p.analyze_pipeline(
        _FakeDefectAnalysis(),
        audibility={
            "noise": False,
            "clipping": False,
            "clicks": False,
            "hum": False,
            "dropouts": False,
            "spectral": False,
        },
    )
    assert d[ProcessingPhase.PHASE_1_FORENSIC].skip is False
    assert d[ProcessingPhase.PHASE_10_FINALIZE].skip is False


def test_target_defect_map_covers_repair_phases() -> None:
    """Jede reparierende Phase hat einen Ziel-Defekt für das Audibilitäts-Gate."""
    repair_phases = {
        ProcessingPhase.PHASE_2_DENOISE,
        ProcessingPhase.PHASE_4_DECLIP,
        ProcessingPhase.PHASE_5_CLICK_REMOVAL,
        ProcessingPhase.PHASE_6_DEHUM,
        ProcessingPhase.PHASE_7_DROPOUT_REPAIR,
        ProcessingPhase.PHASE_8_SPECTRAL_REPAIR,
    }
    assert repair_phases <= set(_PHASE_TARGET_DEFECT)
