"""Tests für den Per-Song-Provenance-Report (§G5 (GEBOTE.md)).

Verpflichtung „bit-identische Reproduzierbarkeit + Provenance-Report je Song
(Kette, Stärken, Seeds, Degradationen)“ — inklusive ehrlicher Lücken-Markierung
und Tamper-Nachweis der Audit-Kette.
"""

from __future__ import annotations

import json
from pathlib import Path

from backend.core.provenance_audit import (
    ProvenanceAudit,
    build_song_provenance_report,
    write_song_provenance_report,
)


def test_report_assembliert_verpflichtungen_und_zeugen(tmp_path: Path) -> None:
    meta = {
        "degradation_status": "degraded",
        "fail_reasons": [{"error_code": "WITNESS_MATERIAL_LOSS"}],
        "witness": {"verdict": "material_loss_audible"},
        "intelligibility_witness": {"delta": -0.02, "never_worsen_ok": True},
        "quality_gate_payload": {"go_nogo": {"verdict": "GO_CAUTION"}},
    }
    audit = ProvenanceAudit(source_file="x.wav", material="shellac")
    audit.record_decision("mode_selection", "balanced gewählt", 0.9)
    rep = build_song_provenance_report(meta, output_path=str(tmp_path / "out.wav"), provenance=audit)

    assert rep["aurik_provenance_report"] is True
    assert rep["software_version"]
    assert rep["degradationen"]["degradation_status"] == "degraded"
    assert rep["degradationen"]["fail_reasons"][0]["error_code"] == "WITNESS_MATERIAL_LOSS"
    assert rep["zeugen"]["witness"]["verdict"] == "material_loss_audible"
    assert rep["zeugen"]["go_nogo"]["verdict"] == "GO_CAUTION"
    # Ehrliche Lücken statt Schönschreibung (§V6 (VERBOTEN.md))
    assert rep["seeds"] == "nicht_erfasst"
    assert rep["kette"]["phasen"] == "nicht_erfasst"
    assert rep["audit"]["integrity"]["valid"] is True

    out = write_song_provenance_report(rep, tmp_path / "out.wav")
    assert out.name == "out.provenance.json"
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["audit"]["entries"][0]["entry_hash"]


def test_report_seeds_aus_seed_manager_schluessel() -> None:
    """§G5 (GEBOTE.md): Der kanonische Seed-Manager-Schlüssel ``session_master_seed``
    (seed_manager.py) wird gefunden und mit song_seed verdichtet."""
    rep = build_song_provenance_report({"session_master_seed": 4711, "song_seed": 815})
    assert rep["seeds"]["master"] == 4711
    assert rep["seeds"]["session"] == 815


def test_report_mit_kette_staerken_seeds() -> None:
    meta = {
        "phase_chain": ["phase_03_denoise", "phase_07_declipper"],
        "calibration_profile": {"global_scalar": 0.9, "family_scalars": {"denoise": 0.95}},
        "seeds": {"master": 42, "session": 4711},
    }
    rep = build_song_provenance_report(meta)
    assert rep["kette"]["phasen"] == ["phase_03_denoise", "phase_07_declipper"]
    assert rep["staerken"]["global_scalar"] == 0.9
    assert rep["seeds"] == {"master": 42, "session": 4711}


def test_report_tamper_nachweisbar() -> None:
    audit = ProvenanceAudit()
    entry = audit.record_decision("s1", "ratio", 1.0)
    entry.entry_hash = "faelschung"
    rep = build_song_provenance_report({}, provenance=audit)
    assert rep["audit"]["integrity"]["valid"] is False
    assert rep["audit"]["integrity"]["failed_entries"]
