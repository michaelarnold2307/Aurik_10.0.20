"""Contract tests for the fail-closed spec evidence gate."""

from __future__ import annotations

import json
from pathlib import Path

from scripts import spec_evidence_gate as gate
from scripts.compliance import wettbewerbs_paritaet_gate


def _valid_report() -> str:
    return """# Evidence

## Evidenzblock
- Spec-Datei: .github/specs/07_quality_and_tests.md
- Abschnitt: §8.1
- Aenderungstyp: Diagnostik- und Gate-Vertrag
- Alte Regel: Nicht kalibrierte Proxywerte konnten blockieren.
- Neue Regel: Die Proxywerte sind report-only, bis eine Hörkalibrierung vorliegt.
- Hypothese: Ein unkalibrierter Proxy darf kein Exportveto auslösen.
- Referenzen/Standards: Hörordnung §2 und Spec 07 §8.1.
- Datensätze/Szenarien: N/A — es wurde kein Audiomaterial ausgewertet.
- Umfang (n): N/A — Policy-Änderung ohne Stichprobe.
- Primarmetrik: N/A — keine Klangwirkung wird behauptet.
- Effektstärke: N/A — keine Audiomessung durchgeführt.
- 95 %-CI: N/A — keine Stichprobe oder inferenzstatistische Messung.
- Signifikanztest + p-Wert: N/A — kein Hypothesentest durchgeführt.
- Seed(s): N/A — es wurde kein Zufallsverfahren ausgeführt.
- Commit: abc1234
- Skript/Befehl: python -m pytest tests/unit/test_spec_evidence_gate.py
- Artefaktpfade: docs/reports/spec_evidence/2026-11-22_proxy_gate_policy.md
- Risiko für P1/P2: Export wird nicht länger allein durch unkalibrierte Scores blockiert.
- Risiko für Artefakte: Hörkalibrierte Invarianten bleiben als eigenständige Gates aktiv.
- Bekannte Unsicherheiten: Der Bericht enthält keine Aussage zur Klangverbesserung.
- Rollback-Kriterium: Belegter Exportfehler durch die geänderte Gate-Logik.
- Entscheidung: CONDITIONAL — keine Qualitätsfreigabe ohne Hörnachweis.
- Maintainer Sign-off: Michael Arnold
- Datum: 2026-11-22
"""


def test_evidence_requires_filled_fields_not_just_markers() -> None:
    marker_only = "## Evidenzblock\nSeed\n95 %-CI\nMaintainer Sign-off\n"
    errors = gate.validate_evidence_text(marker_only)
    assert any("Pflichtfeld fehlt: Spec-Datei" in error for error in errors)
    assert any("Pflichtfeld fehlt: Maintainer Sign-off" in error for error in errors)


def test_evidence_accepts_explicitly_justified_not_applicable_fields() -> None:
    assert (
        gate.validate_evidence_text(
            _valid_report(),
            expected_specs={".github/specs/07_quality_and_tests.md"},
        )
        == []
    )


def test_evidence_must_identify_each_changed_spec() -> None:
    errors = gate.validate_evidence_text(
        _valid_report(),
        expected_specs={
            ".github/specs/07_quality_and_tests.md",
            ".github/specs/26_defekt_reparatur_trainingsvertrag.md",
        },
    )
    assert any("Geänderte Specs nicht im Evidenzblock zugeordnet" in error for error in errors)


def test_evidence_rejects_pending_maintainer_signoff() -> None:
    report = _valid_report().replace("Maintainer Sign-off: Michael Arnold", "Maintainer Sign-off: pending")
    assert any("Sign-off ist nicht erteilt" in error for error in gate.validate_evidence_text(report))


def test_evidence_rejects_signoff_without_maintainer_identity() -> None:
    report = _valid_report().replace("Maintainer Sign-off: Michael Arnold", "Maintainer Sign-off: approved")
    assert any("Name der freigebenden Person fehlt" in error for error in gate.validate_evidence_text(report))


def test_main_push_gate_requires_changed_report_for_changed_spec(tmp_path: Path, capsys) -> None:
    spec = ".github/specs/07_quality_and_tests.md"
    assert gate._validate_changed_files(tmp_path, {spec}) == 1
    assert "require a changed report" in capsys.readouterr().out


def test_main_push_gate_accepts_report_covering_changed_spec(tmp_path: Path) -> None:
    spec = ".github/specs/07_quality_and_tests.md"
    report_path = tmp_path / "docs" / "reports" / "spec_evidence" / "evidence.md"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(_valid_report(), encoding="utf-8")
    assert gate._validate_changed_files(tmp_path, {spec, report_path.relative_to(tmp_path).as_posix()}) == 0


def test_pr_gate_validates_filled_body_and_spec_mapping(tmp_path: Path) -> None:
    spec = ".github/specs/07_quality_and_tests.md"
    assert gate._validate_changed_files(tmp_path, {spec}, pr_body=_valid_report()) == 0


def test_drt16_parity_claim_requires_canonical_validated_json(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(gate, "_ROOT", tmp_path)
    monkeypatch.setattr(wettbewerbs_paritaet_gate, "_ROOT", tmp_path)
    report = _valid_report().replace(
        "Entscheidung: CONDITIONAL — keine Qualitätsfreigabe ohne Hörnachweis.",
        "DRT-16-Entscheidung: PARITAET.",
    )
    errors = gate.validate_evidence_text(report, root=tmp_path)
    assert any("ohne kanonischen JSON-Beleg" in error for error in errors)


def test_drt16_parity_claim_rejects_non_parity_json(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(gate, "_ROOT", tmp_path)
    monkeypatch.setattr(wettbewerbs_paritaet_gate, "_ROOT", tmp_path)
    path = tmp_path / "docs" / "reports" / "current" / "rx12.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"schema": "wrong", "discipline": "Declip"}), encoding="utf-8")
    report = _valid_report().replace(
        "Entscheidung: CONDITIONAL — keine Qualitätsfreigabe ohne Hörnachweis.",
        "DRT-16-Entscheidung: PARITAET; Beleg: `docs/reports/current/rx12.json`.",
    )
    errors = gate.validate_evidence_text(report, root=tmp_path)
    assert any("falsches RX-12-Belegschema" in error for error in errors)


def test_drt16_parity_claim_rejects_path_traversal(tmp_path: Path) -> None:
    report = _valid_report().replace(
        "Entscheidung: CONDITIONAL — keine Qualitätsfreigabe ohne Hörnachweis.",
        "DRT-16-Entscheidung: PARITAET; Beleg: `docs/reports/current/../../private.json`.",
    )
    errors = gate.validate_evidence_text(report, root=tmp_path)
    assert any("Unsicherer DRT-16-Belegpfad" in error for error in errors)
