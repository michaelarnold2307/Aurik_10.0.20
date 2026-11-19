from __future__ import annotations

"""Normative Release-Must-Vertraege fuer §0m und §0q.

Diese Tests sichern zwei Meta-Pflichten ab, die sonst leicht als reine
Dokumentation erscheinen: volle Defektintelligenz in beiden Modi und der
Bug-Gap-Erkennungsworkflow inklusive Type-/Worldclass-Gates.
"""


import inspect
from pathlib import Path

import pytest


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_section_0m_defect_intelligence_inventory_is_mode_invariant() -> None:
    """§0m: Defekt-Erkennung/-Kausalitaet darf nicht je Modus verkleinert werden."""
    from backend.core.causal_defect_reasoner import CAUSE_TO_PHASES, CAUSES, CausalDefectReasoner
    from backend.core.defect_scanner import DefectScanner, DefectType

    assert len(DefectType) >= 54, "DefectScanner muss alle bekannten DetectionTypes abdecken."
    assert len(CAUSES) >= 62, "CausalDefectReasoner muss den vollen Ursachenbestand abdecken."

    scanner_params = inspect.signature(DefectScanner.__init__).parameters
    reasoner_params = inspect.signature(CausalDefectReasoner.__init__).parameters
    assert "mode" not in scanner_params, "DefectScanner darf keine mode-reduzierte Detektionsabdeckung haben."
    assert "mode" not in reasoner_params, "CausalDefectReasoner darf keine mode-reduzierte Ursachenabdeckung haben."

    required_worldclass_causes = {
        "tape_head_level_dip",
        "scrape_flutter",
        "tape_head_clog",
        "proximity_effect_excess",
        "room_mode_resonance",
        "nr_breathing_artifact",
        "flutter_spectral_sidebands",
        "speed_calibration_error",
        "overload_distortion",
        "lacquer_disc_degradation",
        "vocal_quality_degradation",
        "vocal_stem_noise",
    }
    missing_causes = sorted(required_worldclass_causes - set(CAUSES))
    assert not missing_causes, f"§0m: bekannte Weltklasse-Ursachen fehlen: {missing_causes}"

    missing_phase_routes = sorted(cause for cause in required_worldclass_causes if cause not in CAUSE_TO_PHASES)
    assert not missing_phase_routes, f"§0m: Ursachen ohne CAUSE_TO_PHASES-Route: {missing_phase_routes}"


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_section_0q_bug_gap_strategy_is_wired_into_release_gates() -> None:
    """§0q: 5-Layer-Scan, Type-Gates und Trusted-Vocal-Report muessen verdrahtet sein."""
    spec10 = Path(".github/specs/10_bug_gap_strategy.md").read_text(encoding="utf-8")
    copilot = Path(".github/copilot-instructions.md").read_text(encoding="utf-8")
    precommit = Path(".pre-commit-config.yaml").read_text(encoding="utf-8")
    real_bug_gate = Path("scripts/check_mypy_real_bugs.py").read_text(encoding="utf-8")
    release_gate = Path("scripts/worldclass_release_gate.py").read_text(encoding="utf-8")

    for layer in ("L1 Frontend", "L2 Bridge/CLI", "L3 Denker", "L4 UV3-Pipeline", "L5 Phasen/DSP"):
        assert layer in spec10, f"§0q: 5-Layer-Scan unvollstaendig, fehlt {layer}."

    for bug_class in ("R-BLOCKER", "AUDIO-QUALITY", "SPEC-GAP", "TYPE-SAFETY"):
        assert bug_class in spec10 and bug_class in copilot, f"§0q: Bug-Klasse {bug_class} nicht normativ verankert."

    assert "aurik-type-ignore-order" in precommit
    assert "aurik-mypy-real-bug-gate" in precommit
    assert "IGNORED_CODES: set[str] = set()" in real_bug_gate
    assert "trusted_vocal_restoration_report" in release_gate
    assert "best_possible_restoration" in release_gate
    assert "user_confidence_summary" in release_gate
