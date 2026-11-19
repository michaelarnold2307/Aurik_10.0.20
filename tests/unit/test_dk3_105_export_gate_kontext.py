"""[D-K3-105] Export-Gate-Kontext: GUI/Batch exportierten ein Gate ohne Profil.

Befund (gemessen 2026-11-16, vor dem Fix): `profile`/`material`/`thresholds`
im `quality_gate`-Payload stammen ausschließlich aus Metadaten, die **nur**
`cli/aurik_cli.py` schreibt (`export_gate_profile`, `export_gate_material`,
`export_gate_thresholds`, `export_gate_signal_signature`). GUI, Batch und
Denker rufen denselben Builder ohne diesen Kontext auf und exportierten
deshalb `profile=""`, `material=""`, `thresholds={0.0, 0.0}` — obwohl derselbe
Lauf im Experience-Payload ein abgeleitetes Profil (`modern_stable`) und ein
berechnetes WCS-Gate (`profile="vocal"`) besaß — §G9 (copilot-instructions.md)
„eine Quelle", §G8 (copilot-instructions.md) Transparenz. Der Leerwert war
nicht harmlos: `_failure_class` las `_qe_threshold <= 1e-4` als Signal und
stufte jede Degradierung als `technical_failure` ein.

Kontrakte:
1. Ohne CLI-Metadaten leitet der Export-Payload Profil/Material aus der EINEN
   Regel ab (`profile_source="derived_preserve_signal"`), nie `""`.
2. Experience- und Export-Payload liefern dasselbe Profil/Material.
3. Die Ableitung deckt alle drei Zweige ab (fragil / modern / neutral).
4. Ein vorhandener CLI-Gate-Kontext gewinnt unverändert (bit-identisch,
   inkl. Risiko-Suffix in der Profilzeichenkette).
5. Fehlende Schwellen werden als `absent` ausgewiesen — Zahlen werden NICHT
   erfunden (§G8 (copilot-instructions.md)), sie bleiben 0.0.
6. Eine fehlende Schwelle darf `failure_class` nicht auf
   `technical_failure` kippen; echte Pipeline-Fehler bleiben technisch.
7. Die Ableitung ist deterministisch (§G5 (copilot-instructions.md)) und der WCS-Gate-Block bleibt
   unangetastet (Zeuge, kein Eingriff).
8. Beide Builder (`bridge.py` + `bridge_export.py`) liefern denselben
   Gate-Kontext und speisen sich aus genau EINER Definition.
9. GUI/Batch/CLI führen die Herkunft in die Export-Metadaten (Ende-zu-Ende).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

_REPO = Path(__file__).resolve().parents[2]
_BRIDGE = _REPO / "backend" / "api" / "bridge.py"
_BRIDGE_EXPORT = _REPO / "backend" / "api" / "bridge_export.py"
_GUI = _REPO / "Aurik10" / "ui" / "modern_window.py"
_CLI = _REPO / "cli" / "aurik_cli.py"
_BATCH = _REPO / "batch_processor.py"

_WCS_GATE: dict[str, Any] = {
    "wcs": 0.912,
    "threshold": 0.88,
    "profile": "vocal",
    "passed": True,
    "artifact_veto": False,
}


class _Result:
    """Minimaler RestorationResult-Doppelgänger (kein Audio, keine Modelle)."""

    def __init__(self, meta: dict[str, Any], stage_notes: dict[str, Any] | None = None) -> None:
        self.metadata = meta
        self.stage_notes = stage_notes or {}
        self.material_type = "vinyl"
        self.quality_estimate = 0.62
        self.chroma_correlation = 0.95


def _payload(meta: dict[str, Any], stage_notes: dict[str, Any] | None = None) -> dict[str, Any]:
    from backend.api.bridge import build_export_quality_gate_payload

    return build_export_quality_gate_payload(_Result(meta, stage_notes))


def _experience(meta: dict[str, Any], stage_notes: dict[str, Any] | None = None) -> dict[str, Any]:
    from backend.api.bridge import get_experience_insights

    return get_experience_insights(_Result(meta, stage_notes))["quality_gate"]


@pytest.mark.timeout(30)
def test_gui_pfad_exportiert_abgeleitetes_profil_statt_leerwert() -> None:
    payload = _payload({"worldclass_composite_gate": dict(_WCS_GATE), "degradation_status": "ok"})

    assert payload["profile"] == "modern_stable"
    assert payload["profile_source"] == "derived_preserve_signal"
    assert payload["material"] == "vinyl"
    assert payload["material_source"] == "result_material_type"
    # §G8 (copilot-instructions.md): fehlende Schwellen bleiben leer, werden aber als fehlend benannt.
    assert payload["thresholds"] == {"quality_estimate": 0.0, "level_drop_db": 0.0}
    assert payload["thresholds_source"] == "absent"


@pytest.mark.timeout(30)
def test_erfahrung_und_export_teilen_ein_profil() -> None:
    meta = {"worldclass_composite_gate": dict(_WCS_GATE), "degradation_status": "ok"}
    stage_notes = {"exzellenz_recovery_profile": {"preserve_signal": 0.12}}

    export = _payload(meta, stage_notes)
    experience = _experience(meta, stage_notes)

    assert export["profile"] == experience["profile"] == "modern_stable"
    assert export["material"] == experience["material"] == "vinyl"


@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    ("preserve_signal", "degradation", "expected"),
    [
        (0.72, "ok", "fragile_or_transient_risk"),
        (0.12, "ok", "modern_stable"),
        (0.35, "ok", "neutral"),
        (0.35, "degraded", "neutral"),
        (0.12, "degraded", "neutral"),
    ],
)
def test_ableitung_deckt_alle_zweige_ab(preserve_signal: float, degradation: str, expected: str) -> None:
    payload = _payload(
        {"worldclass_composite_gate": dict(_WCS_GATE), "degradation_status": degradation},
        {"exzellenz_recovery_profile": {"preserve_signal": preserve_signal}},
    )

    assert payload["profile"] == expected
    assert payload["profile_source"] == "derived_preserve_signal"


@pytest.mark.timeout(30)
def test_cli_kontext_gewinnt_unveraendert() -> None:
    payload = _payload(
        {
            "worldclass_composite_gate": dict(_WCS_GATE),
            "degradation_status": "ok",
            "export_gate_profile": "fragile_or_transient_risk(risk=0.54)",
            "export_gate_material": "vinyl",
            "export_gate_thresholds": {"quality_estimate": 0.42, "level_drop_db": 4.3},
            "export_gate_preserve_signal": 0.0,
        },
        {"exzellenz_recovery_profile": {"preserve_signal": 0.12}},
    )

    assert payload["profile"] == "fragile_or_transient_risk(risk=0.54)"
    assert payload["profile_source"] == "export_gate_meta"
    assert payload["material_source"] == "export_gate_meta"
    assert payload["thresholds"] == {"quality_estimate": 0.42, "level_drop_db": 4.3}
    assert payload["thresholds_source"] == "export_gate_meta"


@pytest.mark.timeout(30)
def test_fehlende_schwelle_kippt_failure_class_nicht_auf_technisch() -> None:
    payload = _payload(
        {
            "worldclass_composite_gate": dict(_WCS_GATE),
            "degradation_status": "degraded",
            "fail_reason": "level_drop_db",
        }
    )

    assert payload["thresholds_source"] == "absent"
    assert payload["failure_class"] == "quality_failure"


@pytest.mark.timeout(30)
def test_pipeline_fehler_bleibt_technisch_und_cli_schwelle_wirkt() -> None:
    pipeline_error = _payload(
        {
            "degradation_status": "critical_degraded",
            "fail_reason": "pipeline_blocked:restaurierung",
        }
    )
    assert pipeline_error["failure_class"] == "technical_failure"

    cli_threshold = _payload(
        {
            "degradation_status": "degraded",
            "fail_reason": "level_drop_db",
            "export_gate_thresholds": {"quality_estimate": 0.0, "level_drop_db": 4.3},
        }
    )
    assert cli_threshold["thresholds_source"] == "export_gate_meta"
    assert cli_threshold["failure_class"] == "technical_failure"


@pytest.mark.timeout(30)
def test_determinismus_und_wcs_gate_unangetastet() -> None:
    meta = {"worldclass_composite_gate": dict(_WCS_GATE), "degradation_status": "ok"}
    first = _payload(meta)
    second = _payload(meta)

    assert json.dumps(first, sort_keys=True, default=str) == json.dumps(second, sort_keys=True, default=str)
    assert first["worldclass_composite_gate"]["wcs"] == pytest.approx(0.912)
    assert first["worldclass_composite_gate"]["profile"] == "vocal"
    assert first["worldclass_composite_gate"]["passed"] is True


@pytest.mark.timeout(30)
def test_beide_builder_liefern_denselben_gate_kontext() -> None:
    from backend.api.bridge import build_export_quality_gate_payload as bridge_builder
    from backend.api.bridge_export import build_export_quality_gate_payload as export_builder

    result = _Result(
        {"worldclass_composite_gate": dict(_WCS_GATE), "degradation_status": "ok"},
        {"exzellenz_recovery_profile": {"preserve_signal": 0.12}},
    )
    from_bridge = bridge_builder(result)
    from_export = export_builder(result)

    for key in ("profile", "profile_source", "material", "material_source", "thresholds", "thresholds_source"):
        assert from_bridge[key] == from_export[key], f"Builder-Drift bei {key}"


@pytest.mark.timeout(30)
def test_genau_eine_definition_und_herkunft_im_export_meta() -> None:
    bridge_src = _BRIDGE.read_text(encoding="utf-8")
    export_src = _BRIDGE_EXPORT.read_text(encoding="utf-8")

    assert bridge_src.count("def _export_gate_kontext(") == 1
    # Kein zweites Regelwerk: der Export-Builder delegiert an dieselbe Definition.
    assert "from backend.api.bridge import _export_gate_kontext" in export_src
    assert 'return "fragile_or_transient_risk"' not in export_src

    for path in (_GUI, _CLI, _BATCH):
        text = path.read_text(encoding="utf-8")
        assert "quality_gate_profile_source" in text, f"{path.name}: Profil-Herkunft fehlt im Export-Meta"
        assert "quality_gate_thresholds_source" in text, f"{path.name}: Schwellen-Herkunft fehlt im Export-Meta"
