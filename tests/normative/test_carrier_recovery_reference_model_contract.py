from __future__ import annotations

"""[RELEASE_MUST] §0d / §1.2a Carrier-Recovery-Referenzmodell — Contract-Gates.

Diese Tests verankern das dreischichtige Referenzmodell als CI-Vertrag:
1) Per-Phase PMGG Baseline-Capping (+0.05) fuer restorative Phasen
2) End-of-Pipeline Referenz-Shift auf best_carrier_checkpoint bei ratio > 0.15
3) Metadata-Pflichtfelder zur transparenten Auditierbarkeit
"""


from pathlib import Path

import pytest

_PM = Path("backend/core/per_phase_musical_goals_gate.py")
_UV3 = Path("backend/core/unified_restorer_v3.py")


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_layer1_pmgg_restorative_baseline_capping_present() -> None:
    assert _PM.exists(), f"Fehlt: {_PM}"
    text = _PM.read_text(encoding="utf-8")

    assert "_is_restorative" in text, "PMGG muss restorative Phasen erkennen."
    assert "min(v, _thresholds.get(g, v) + 0.05)" in text, "§2.29c Baseline-Capping (+0.05) fehlt im PMGG-Retry-Pfad."
    assert "PMGG restorative baseline cap" in text, "PMGG soll Baseline-Capping explizit loggen (Audit-Transparenz)."


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_layer2_end_of_pipeline_reference_shift_present() -> None:
    assert _UV3.exists(), f"Fehlt: {_UV3}"
    text = _UV3.read_text(encoding="utf-8")

    assert "_ccr_ratio > 0.15" in text, "§1.2a Schwelle 0.15 fuer Reference-Shift fehlt."
    assert "_mg_ref = _ccr_checkpoint" in text, (
        "End-of-Pipeline Goal-Referenz muss auf best_carrier_checkpoint geschoben werden."
    )
    assert "Goal-Referenz auf best_carrier_checkpoint" in text, "Reference-Shift sollte im Log erkennbar sein."


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_layer3_metadata_contract_present() -> None:
    text = _UV3.read_text(encoding="utf-8")

    assert '"carrier_chain_recovery_ratio"' in text, "Metadata-Pflichtfeld carrier_chain_recovery_ratio fehlt in UV3."
    assert '"reference_shifted"' in text, "Metadata-Feld reference_shifted fuer §0d-Ebene-2-Audit fehlt."
