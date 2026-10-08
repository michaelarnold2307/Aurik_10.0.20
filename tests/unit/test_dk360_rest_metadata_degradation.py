"""[D-K3-60] Degradations-Evidenz erreicht den Export (RT-Budget-Timeout-Pfad).

Endpunkt-Beweis 2026-10-08 (Lauf 10.12.17, `processing_metadata` des Exports):
Ein RT-Budget-Timeout des RestaurierDenkers (§2.38-Guard) ließ `_rest_metadata`
leer — der Export trug Default-Leerwerte (worldclass `threshold=0.0`,
`profile=""`, `hybrid_engineer_vector={}`) und `degradation_status="ok"`,
obwohl das Lauf-Log `⚠ RestaurierDenker fehlgeschlagen: …überschritt
RT-Budget` und `⚠ Kritische Stufen ausgefallen: restaurierung` zeigte
(§G8 (copilot-instructions.md)).

Kontrakte:
1. `aurik_denker.py`: Timeout-`except` UND Skip-`else` markieren den
   Metadaten-Verlust explizit (`restaurierdenker_stage_error`).
2. `aurik_denker.py`: `degradation_status`/`fail_reason`/`fail_reasons`
   werden vor der AurikErgebnis-Konstruktion in `_rest_metadata` gespiegelt —
   exakt die Keys, die `build_export_quality_gate_payload` liest.
3. BEIDE Builder (bridge.py + bridge_export.py): das Payload-Feld
   `worldclass_composite_gate.evidence_state` ist `evaluated`/`not_evaluated`
   (keine stillen Default-Leerwerte, §G8-Transparenz).
4. CLI und Batch führen `quality_gate_evidence_worldclass_state`
   (§G9 (copilot-instructions.md) Canonical-Contract-Parität).
5. Semantik (Runtime): leere WCS-Metadaten + kritische Stage ⇒ Payload
   `passed=False` + `degradation_status` erreicht den Export.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

_REPO = Path(__file__).resolve().parents[2]
_AURIK_DENKER = _REPO / "denker" / "aurik_denker.py"
_BRIDGE = _REPO / "backend" / "api" / "bridge.py"
_BRIDGE_EXPORT = _REPO / "backend" / "api" / "bridge_export.py"
_CLI = _REPO / "cli" / "aurik_cli.py"
_BATCH = _REPO / "batch_processor.py"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


@pytest.mark.timeout(10)
def test_dk360_timeout_und_skip_pfad_markieren_metadaten_verlust() -> None:
    """§G8 (copilot-instructions.md): Kein stiller Metadaten-Verlust mehr — beide Stage-Fehlerpfade markieren."""
    src = _text(_AURIK_DENKER)
    assert '_rest_metadata["restaurierdenker_stage_error"] = str(exc)' in src
    assert '_rest_metadata["restaurierdenker_stage_error"] = "RT-Budget ausgeschöpft' in src


@pytest.mark.timeout(10)
def test_dk360_degradations_evidenz_wird_gespiegelt() -> None:
    """Die Degradations-Evidenz liegt in genau den Keys, die der Builder liest."""
    src = _text(_AURIK_DENKER)
    assert '_rest_metadata["degradation_status"] = str(_degradation_status)' in src
    assert '_rest_metadata["fail_reason"] = str(_fail_reason)' in src
    assert '_rest_metadata["fail_reasons"] = [dict(entry) for entry in _stage_fail_reasons]' in src


@pytest.mark.timeout(10)
def test_dk360_evidence_state_in_beiden_buildern_verdrahtet() -> None:
    """Beide Builder-Varianten (bridge.py + bridge_export.py) tragen den Evidenzstand."""
    for path in (_BRIDGE, _BRIDGE_EXPORT):
        src = _text(path)
        assert '"evidence_state": "evaluated" if _wcs_gate else "not_evaluated"' in src, path.name


@pytest.mark.timeout(10)
def test_dk360_cli_und_batch_fuehren_evidence_state() -> None:
    """§G9 (copilot-instructions.md): CLI und Batch exportieren denselben Evidenzstand (Canonical Contract)."""
    for path in (_CLI, _BATCH):
        src = _text(path)
        assert '"quality_gate_evidence_worldclass_state"' in src, path.name


@pytest.mark.timeout(10)
def test_dk360_builder_evidence_state_und_degradations_semantik_runtime() -> None:
    """Runtime: Evidenzstand + Fail-closed-Semantik bei kritischer Degradation."""
    from backend.api.bridge import build_export_quality_gate_payload as _build_bridge
    from backend.api.bridge_export import build_export_quality_gate_payload as _build_export

    class _FakeResult:
        audio = None
        original_audio = None
        chroma_correlation = 0.9
        lufs_delta = 0.0
        quality_estimate = 0.42
        metadata: dict[str, Any] = {}

    _builders = (("bridge.py", _build_bridge), ("bridge_export.py", _build_export))

    # (a) Berechnetes Gate im Metadaten-Baum ⇒ evidence_state="evaluated".
    classified = _FakeResult()
    classified.metadata = {
        "worldclass_composite_gate": {
            "wcs": 0.77,
            "threshold": 0.85,
            "profile": "vocal",
            "artifact_veto": False,
            "passed": True,
        },
        "export_gate_profile": "neutral",
        "export_gate_material": "vinyl",
        "export_gate_thresholds": {"quality_estimate": 0.42, "level_drop_db": 4.3},
        "export_gate_signal_signature": {
            "crest_db": 10.0,
            "hf_ratio": 0.5,
            "transient_ratio": 0.5,
            "micro_dynamic_db": 1.0,
        },
        "export_gate_preserve_signal": 0.5,
    }
    for name, builder in _builders:
        payload = builder(classified)
        gate = payload.get("worldclass_composite_gate", {})
        assert gate.get("evidence_state") == "evaluated", f"{name}: Evidenzstand verloren"
        assert gate.get("wcs") == pytest.approx(0.77), f"{name}: wcs verloren"

    # (b) Fehlende Gate-Metadaten + kritische Stage ⇒ "not_evaluated" + fail-closed.
    missing = _FakeResult()
    missing.metadata = {
        "degradation_status": "critical_degraded",
        "fail_reason": "critical_stage_failure:restaurierung",
        "fail_reasons": [
            {
                "component": "RestaurierDenker",
                "stage": "restaurierung",
                "error_code": "STAGE_RESTAURIERUNG_FAILED",
                "severity": "critical",
                "exc_type": "RuntimeError",
                "exc_msg": "RestaurierDenker überschritt RT-Budget",
            }
        ],
    }
    for name, builder in _builders:
        payload = builder(missing)
        gate = payload.get("worldclass_composite_gate", {})
        assert gate.get("evidence_state") == "not_evaluated", f"{name}: stille Nullwerte"
        assert payload.get("passed") is False, f"{name}: Degradation nicht fail-closed"
        assert payload.get("degradation_status") == "critical_degraded", (
            f"{name}: Degradations-Evidenz erreicht den Export nicht"
        )
        assert payload.get("fail_reason") == "critical_stage_failure:restaurierung", name
