"""Weltklasse-Validierung §G10–§G12 (copilot-instructions.md) — Enforcement-Matrix.

Norm → Code-Instanz → Verdrahtung, fail-closed. Validiert den Weltklasse-Verbund
der copilot-instructions.md:

- §G10 (copilot-instructions.md): Defekt-Freiheit unter der Hörbarkeitsschwelle —
  Gate-API (`evidence_state`/`gate_verified`), Chunked-Song-Nachzug (D-K3-68),
  §G10-Exportfeld `defect_residual_audibility`.
- §G11 (copilot-instructions.md): Weltklasse-Abnahme — HPI-Produktformel,
  Einladungs-Gate auf Song-Ebene, WCS-Metadaten-Durchreichung UV3 → Chunked-Erbe
  → RestaurierDenker (§2.53-Pass-through) → AurikDenker → Bridge-Payload → CLI.
- §G12 (copilot-instructions.md): Universeller Import-Vertrag — Chunked-Pfad ==
  Voll-Pfad: Deferral-Flags, P0-1-/D-K3-63-/D-K3-68-Kontrakte.

Hinweis ID-Zitierdisziplin: Jedes §G/§V-Zitat qualifiziert (Quelle in Klammern).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_C1 = _ROOT / ".github" / "copilot-instructions.md"
_S13 = _ROOT / ".github" / "specs" / "13_human_ear_quality.md"
_UV3 = _ROOT / "backend" / "core" / "unified_restorer_v3.py"
_HPI = _ROOT / "backend" / "core" / "holistic_perceptual_gate.py"
_REST_DENKER = _ROOT / "denker" / "restaurier_denker.py"
_AURIK_DENKER = _ROOT / "denker" / "aurik_denker.py"
_CLI = _ROOT / "cli" / "aurik_cli.py"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ── §G10 (copilot-instructions.md): Defekt-Freiheit unter Hörbarkeitsschwelle ──


def test_g10_norm_verankert() -> None:
    """§G10-Norm in copilot-instructions.md + Spec 13 verankert."""
    c1 = _text(_C1)
    assert "§G10" in c1, "§G10 (copilot-instructions.md) fehlt in der normativen Spitze"
    assert "defect_residual_audibility" in c1, "§G10-Exportfeld fehlt in der Norm"
    s13 = _text(_S13)
    assert "§G10" in s13, "Spec-13-Querverweis auf §G10 (copilot-instructions.md) fehlt"


def test_g10_gate_semantik_fail_closed() -> None:
    """Gate-API: ohne Scan keine Zusage (fail-closed)."""
    from backend.core.defect_audibility_gate import evaluate_defect_audibility

    rep_ok = evaluate_defect_audibility({}, material_key="vinyl", post_scan_ran=True)
    assert rep_ok.evidence_state == "no_residual_defects"
    assert rep_ok.gate_passed is True
    assert rep_ok.gate_verified is True
    meta = rep_ok.to_metadata()
    assert meta["gate_verified"] is True
    assert meta["evidence_state"] == "no_residual_defects"

    rep_missing = evaluate_defect_audibility({}, material_key="vinyl", post_scan_ran=False)
    assert rep_missing.evidence_state == "scan_missing"
    assert rep_missing.gate_verified is False, "fail-closed: scan_missing darf nie gate_verified tragen"


def test_g10_chunked_song_nachzug_verdrahtet() -> None:
    """D-K3-68: §G10 (copilot-instructions.md)-Exportfeld — Song-Audibility-Nachzug nach der Assembly."""
    src = _text(_UV3)
    assert "§G10 Song-Audibility nach Assembly" in src  # Pin §G10 (copilot-instructions.md) — UV3-Logzeile
    assert '_fd_meta["defect_residual_audibility"] = _song_aud_meta' in src
    assert "post_scan_ran=bool(_song_aud_data)" in src, "post_scan_ran muss fail-closed aus der Datenlage kommen"


# ── §G11 (copilot-instructions.md): Weltklasse-Abnahme ────────────────────────


def test_g11_norm_verankert() -> None:
    """§G11-Norm + Weltklasse-Abnahmetabelle (§13.0.1) verankert."""
    c1 = _text(_C1)
    assert "§G11" in c1, "§G11 (copilot-instructions.md) fehlt in der normativen Spitze"
    s13 = _text(_S13)
    assert "§13.0.1" in s13, "Weltklasse-Abnahmetabelle §13.0.1 fehlt in Spec 13"
    assert "Hörerlebnis" in s13, "Hörerlebnis-Ziel fehlt in Spec 13"
    assert "Timbral-Treue" in s13, "Timbral-Treue-Schwelle fehlt in Spec 13"


def test_g11_hpi_produktformel() -> None:
    """HPI ist eine Produktformel — eine 0-Komponente senkt den Gesamtwert."""
    src = _text(_HPI)
    assert "mert_sim * timbral * artifact_freedom * emotional_arc_score" in src


def test_g11_wcs_payload_durchreichung_runtime() -> None:
    """Runtime: BEIDE Builder reichen WCS-Metadaten unverändert in den Export-Payload."""
    from backend.api.bridge import build_export_quality_gate_payload as _build_bridge
    from backend.api.bridge_export import build_export_quality_gate_payload as _build_export

    wcs = {
        "wcs": 0.77,
        "threshold": 0.88,
        "profile": "vocal",
        "artifact_veto": False,
        "passed": False,
    }
    evidence = {"worldclass_composite_gate": {"source_class": "C", "revalidate_by": "2026-09-30"}}

    class _FakeResult:
        metadata: dict[str, Any] = {
            "worldclass_composite_gate": dict(wcs),
            "hybrid_engineer_vector": {"vocal_identity_preservation": 0.9},
            "threshold_evidence": dict(evidence),
            "musical_goals": {"scores": {}, "thresholds": {}},
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
        audio = None
        original_audio = None
        chroma_correlation = 0.9
        lufs_delta = 0.0
        quality_estimate = 0.42

    for name, builder in (("bridge.py", _build_bridge), ("bridge_export.py", _build_export)):
        payload = builder(_FakeResult())
        got = payload.get("worldclass_composite_gate", {})
        assert got.get("wcs") == pytest.approx(0.77), f"{name}: wcs verloren"
        assert got.get("profile") == "vocal", f"{name}: profile verloren"
        assert payload.get("threshold_evidence", {}).get("worldclass_composite_gate"), f"{name}: WCS-Evidenz verloren"


def test_g11_kette_pins_uv3_denker_cli() -> None:
    """§G11-Transportkette ab UV3-Result bis CLI-Export verdrahtet (Source-Pins)."""
    uv3 = _text(_UV3)
    assert '_meta = dict(getattr(_first_result, "metadata", {}) or {})' in uv3

    rest_denker = _text(_REST_DENKER)
    assert "**(dict(raw.metadata or {})" in rest_denker, "§2.53-Pass-through im RestaurierDenker fehlt"

    aurik_denker = _text(_AURIK_DENKER)
    assert "metadata={**_rest_metadata" in aurik_denker, "AurikErgebnis-Metadaten-Erbe fehlt"

    cli = _text(_CLI)
    assert '"quality_gate_worldclass_profile": str(_wcs_payload.get("profile", "") or "")' in cli


def test_g11_einladungs_gate_song_level() -> None:
    """Einladungs-Gate §G11 (copilot-instructions.md) läuft einmal auf dem assemblierten Song."""
    src = _text(_UV3)
    assert "def _run_inviting_gate_measure" in src
    assert '_meta["inviting_gate_song_level"]' in src


# ── §G12 (copilot-instructions.md): Universeller Import-Vertrag (Chunked=Voll) ─


def test_g12_norm_verankert() -> None:
    """§G12-Norm + Universal-Abschnitt (§13.0.2) verankert."""
    c1 = _text(_C1)
    assert "§G12" in c1, "§G12 (copilot-instructions.md) fehlt in der normativen Spitze"
    s13 = _text(_S13)
    assert "§13.0.2" in s13, "Universal-Abschnitt §13.0.2 fehlt in Spec 13"


def test_g12_deferral_flags_verdrahtet() -> None:
    """Chunked-Deferral: Stufe-2-Tail nur auf dem letzten Chunk, Song-Tail nach Assembly."""
    src = _text(_UV3)
    assert '_chunk_kwargs["_chunked_tail_skip"] = True' in src
    assert '_chunk_kwargs["_chunked_last"] = False' in src


def test_g12_finale_kontrakte_existieren() -> None:
    """Die P0-1-/D-K3-63-/D-K3-68-Vertragstests (Chunked=Voll) existieren."""
    for name in (
        "test_p0_1_end_gate_chunking.py",
        "test_dk363_finalpolish_chunk_guard.py",
        "test_dk368_song_audibility_chain.py",
    ):
        assert (_ROOT / "tests" / "unit" / name).exists(), f"Vertragstest fehlt: {name}"
