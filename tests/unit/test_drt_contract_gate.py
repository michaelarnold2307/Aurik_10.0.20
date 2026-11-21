"""Vertragstests für das DRT-Gate (Spec 26, 2026-11-15). Modell-frei."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import scripts.compliance.drt_contract_gate as drt
import scripts.compliance.wettbewerbs_paritaet_gate as parity


def test_baseline_gruen() -> None:
    """Der aktuelle Repo-Stand hält die Leitplanken ein (Fail-closed-Baseline)."""
    fehler: list[str] = []
    drt._pruefe_trainer(fehler)
    drt._pruefe_gate(fehler)
    drt._pruefe_dsp(fehler)
    assert fehler == [], f"DRT-Baseline verletzt: {fehler}"


def test_gate_erkennt_fehlenen_identitaets_anker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """DRT-2: Trainer ohne Identitäts-Anker ⇒ Verstoß (die Schutz-Arm-Verlustrunde)."""
    trainer = drt.TRAINER.read_text(encoding="utf-8")
    ohne_anker = trainer.replace("loss = l1 + lam * pen + lam_id * l1_id", "loss = l1 + lam * pen")
    fake = tmp_path / "train_witness_kopf.py"
    fake.write_text(ohne_anker, encoding="utf-8")
    monkeypatch.setattr(drt, "TRAINER", fake)
    fehler: list[str] = []
    drt._pruefe_trainer(fehler)
    assert any("DRT-2" in f for f in fehler)


def test_gate_erkennt_schwellen_senkung(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """DRT-5: Q_MIN-Senkung ⇒ Verstoß (Schwellwert-Senkung als Nacharbeit ist verboten)."""
    gate_skript = drt.GATE_SKRIPT.read_text(encoding="utf-8")
    gesenkt = gate_skript.replace("Q_MIN = 0.5", "Q_MIN = 0.3")
    fake = tmp_path / "verify_witness_kopf_gate.py"
    fake.write_text(gesenkt, encoding="utf-8")
    monkeypatch.setattr(drt, "GATE_SKRIPT", fake)
    fehler: list[str] = []
    drt._pruefe_gate(fehler)
    assert any("DRT-5" in f for f in fehler)


def test_spec_datei_existiert() -> None:
    spec = drt._ROOT / ".github" / "specs" / "26_defekt_reparatur_trainingsvertrag.md"
    assert spec.is_file()
    q = spec.read_text(encoding="utf-8")
    for regel in [
        "DRT-1",
        "DRT-2",
        "DRT-3",
        "DRT-4",
        "DRT-5",
        "DRT-6",
        "DRT-7",
        "DRT-8",
        "DRT-9",
        "DRT-10",
        "DRT-11",
        "DRT-12",
        "DRT-13",
        "DRT-14",
        "DRT-15",
        "DRT-16",
    ]:
        assert regel in q
    assert q.count("## DRT-12 —") == 1


def test_drt_789_baseline_gruen() -> None:
    """Verschärfung 2026-11-15: Sinn-Beweis-Beleg vorhanden, No-Op-Erkennung im Skript."""
    fehler: list[str] = []
    drt._pruefe_drt_789(fehler)
    assert fehler == [], f"DRT-7/8/9-Baseline verletzt: {fehler}"


def test_drt_8_ohne_sinn_beweis_blockt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """DRT-8: fehlender Sinn-Beweis-Beleg ⇒ Verstoß (keine Aktivierung ohne Ganz-Titel-Ebene)."""
    leer = tmp_path / "reports"
    leer.mkdir()
    monkeypatch.setattr(drt, "_ROOT", tmp_path)
    fehler: list[str] = []
    drt._pruefe_drt_789(fehler)
    assert any("DRT-8" in f for f in fehler)


def test_drt_16_baseline_gruen() -> None:
    fehler: list[str] = []
    drt._pruefe_drt_16(fehler)
    assert fehler == [], f"DRT-16-Vertrag fehlt: {fehler}"


def test_disziplinregister_baseline_strukturell_gruen() -> None:
    register = parity.LEDGER.read_text(encoding="utf-8")
    fehler: list[str] = []
    parity._pruefe_disziplin_tabelle(register, fehler)
    assert fehler == [], f"Disziplinregister strukturell ungültig: {fehler}"


def test_disziplinregister_erkennt_fehlende_und_doppelte_zeilen() -> None:
    register = parity.LEDGER.read_text(encoding="utf-8")
    declip_zeile = next(zeile for zeile in register.splitlines() if zeile.startswith("| Declip "))
    register = register.replace("| Dehum ", "| Dehum__entfernt ", 1)
    register = register.replace(declip_zeile, f"{declip_zeile}\n{declip_zeile}", 1)
    fehler: list[str] = []
    parity._pruefe_disziplin_tabelle(register, fehler)
    assert any("Dehum" in f and "fehlt" in f for f in fehler)
    assert any("Declip" in f and "mehrfach" in f for f in fehler)


def test_gesichert_erfordert_zeileneigenes_existierendes_runbook() -> None:
    register = parity.LEDGER.read_text(encoding="utf-8")
    register = register.replace(
        "§4-Runbook: `docs/guides/GO_NO_GO_DECISION_PROTOCOL.md`",
        "§4-Hör-/RX-Vergleichshörtest",
    )
    fehler: list[str] = []
    parity._pruefe_disziplin_tabelle(register, fehler)
    assert any("Dereverb" in f and "zeileneigenes Hör-Runbook" in f for f in fehler)


def test_gesichert_kriterium_b_prueft_beide_berichtsebenen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(parity, "_ROOT", tmp_path)
    regions_path = tmp_path / "docs" / "reports" / "current" / "regions.json"
    title_path = tmp_path / "data" / "titles.json"
    regions_path.parent.mkdir(parents=True)
    title_path.parent.mkdir(parents=True)

    titles = [f"title-{index}" for index in range(10)]
    title_rows = [
        {
            "song": title,
            "rezept": "clicks",
            "delta_defektlast": -0.02,
            "delta_musik_mos": 0.0,
            "besser": False,
            "ueberrestauration": False,
        }
        for title in titles
    ]
    title_report = {
        "schema": "aurik.schadensbewertung/1",
        "pegel_matching": True,
        "n_paare": 10,
        "n_besser": 0,
        "n_ueberrestauration": 0,
        "aggregat": {"clicks": {"n": 10, "besser": 0, "delta_defektlast": -0.02, "delta_musik_mos": 0.0}},
        "zeilen": title_rows,
    }
    regions_report = {
        "schema": "aurik.witness_kopf_gate/2",
        "klasse": "clicks",
        "holdout_titel": titles[0],
        "n_regionen": 8,
        "n_uebersprungen": 2,
        "n_q_bewertbar": 6,
        "n_never_worsen_bewertbar": 6,
        "q_values": [0.96] * 6,
        "never_worsen_flags": [True] * 6,
        "q_median": 0.96,
        "q_anteil_ge_0_5": 1.0,
        "never_worsen_anteil": 1.0,
        "gate_bestanden": True,
    }
    regions_path.write_text(json.dumps(regions_report), encoding="utf-8")
    title_path.write_text(json.dumps(title_report), encoding="utf-8")
    evidenz = "DRT-5-Klasse: `clicks`; DRT-8-Rezept: `clicks`; `docs/reports/current/regions.json`; `data/titles.json`"

    fehler: list[str] = []
    parity._pruefe_gesichert_evidenz("Declick", evidenz, fehler)
    assert fehler == []

    regions_report["q_anteil_ge_0_5"] = 0.95
    regions_path.write_text(json.dumps(regions_report), encoding="utf-8")
    fehler = []
    parity._pruefe_gesichert_evidenz("Declick", evidenz, fehler)
    assert any("DRT-5-Schwellen" in f for f in fehler)


def test_gesichert_ohne_messbaren_kriterienbeleg_blockt() -> None:
    register = parity.LEDGER.read_text(encoding="utf-8")
    register = register.replace(
        "| Declip (Clipping) | RX De-clip | LUECKE |", "| Declip (Clipping) | RX De-clip | GESICHERT |", 1
    )
    fehler: list[str] = []
    parity._pruefe_disziplin_tabelle(register, fehler)
    assert any("Declip" in f and "maschinenprüfbaren" in f for f in fehler)


def test_gesichert_meldet_unlesbaren_json_beleg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(parity, "_ROOT", tmp_path)
    report = tmp_path / "data" / "broken.json"
    report.parent.mkdir()
    report.write_text("{", encoding="utf-8")
    fehler: list[str] = []
    parity._pruefe_gesichert_evidenz(
        "Dereverb",
        "Sinn-Beweis-Rezept: `raum`; `data/broken.json`",
        fehler,
    )
    assert any("interner JSON-Beleg nicht lesbar" in f for f in fehler)


def test_paritaet_status_ohne_json_beleg_blockt() -> None:
    register = parity.LEDGER.read_text(encoding="utf-8")
    register += "\n| Testdisziplin | iZotope RX 12 | PARITAET | kein Beleg | — |\n"
    fehler: list[str] = []
    parity._pruefe_rx12_vertrag(register, fehler)
    assert any("PARITAET ohne verlinkten maschinenlesbaren RX-Beleg" in f for f in fehler)


def test_rx12_paritaet_beleg_prueft_ci_und_panel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    report_path = Path("docs/reports/current/rx12_decrackle.json")
    target = tmp_path / report_path
    target.parent.mkdir(parents=True)
    mos_deltas = [0.0, 0.01, 0.02, 0.03, -0.01, 0.01, 0.02, 0.0, 0.01, 0.01]
    defect_deltas = [-0.02, -0.01, 0.0, -0.03, 0.0, -0.01, -0.02, 0.0, -0.01, 0.0]
    report = {
        "schema": "aurik.rx12_parity/1",
        "discipline": "Decrackle",
        "reference": {
            "product": "iZotope RX",
            "version": "12.0.0",
            "build": "build-1",
            "module": "De-crackle",
            "settings_sha256": "a" * 64,
        },
        "aurik": {
            "version": "10.0.20",
            "commit": "abc123",
            "production_path": "bridge>denker>uv3>phase_68",
            "active_flags": [],
            "settings_sha256": "c" * 64,
        },
        "dataset": {
            "paired_input_manifest_sha256": "b" * 64,
            "holdout": True,
            "n_titles": 10,
            "pairs": [
                {
                    "source_id": f"title-{index}",
                    "input_pcm_sha256_aurik": f"{index + 1:064x}",
                    "input_pcm_sha256_reference": f"{index + 1:064x}",
                    "musik_mos_delta": mos_deltas[index],
                    "defektlast_delta": defect_deltas[index],
                }
                for index in range(10)
            ],
        },
        "objective": {
            "jnd": 0.05,
            "jnd_source": "wohlklang_zeuge.JND_TOLERANZ",
            "musik_mos_delta_by_title": mos_deltas,
            "defektlast_delta_by_title": defect_deltas,
            "musik_mos_delta_ci95": [0.001740, 0.018260],
            "defektlast_delta_ci95": [-0.017541, -0.002459],
            "noninferiority_passed": True,
        },
        "panel": {
            "design": "MUSHRA",
            "n_listeners": 30,
            "blind": True,
            "randomized": True,
            "level_matched": True,
            "verdict": "nicht_unterlegen",
            "signoff": "passed",
        },
        "decision": "PARITAET",
    }
    target.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(parity, "_ROOT", tmp_path)
    register = parity.LEDGER.read_text(encoding="utf-8")
    register += f"\n| Decrackle | iZotope RX 12 | PARITAET | `{report_path.as_posix()}` | — |\n"
    fehler: list[str] = []
    parity._pruefe_rx12_vertrag(register, fehler)
    assert fehler == [], f"Gültiger RX-12-Beleg wurde abgelehnt: {fehler}"

    register_ohne_cedar = register + (
        f"\n| Decrackle | iZotope RX 12 / Cedar | PARITAET | `{report_path.as_posix()}` | — |\n"
    )
    fehler = []
    parity._pruefe_rx12_vertrag(register_ohne_cedar, fehler)
    assert any("nicht jede genannte Referenz" in f for f in fehler)
    assert any("Cedar-Referenz" in f for f in fehler)

    report["dataset"]["pairs"][0]["input_pcm_sha256_reference"] = "d" * 64
    target.write_text(json.dumps(report), encoding="utf-8")
    fehler = []
    parity._pruefe_rx12_vertrag(register, fehler)
    assert any("bit-identische Holdout-Paare" in f for f in fehler)
    report["dataset"]["pairs"][0]["input_pcm_sha256_reference"] = "1".zfill(64)

    report["objective"]["musik_mos_delta_ci95"] = [-0.06, 0.02]
    target.write_text(json.dumps(report), encoding="utf-8")
    fehler = []
    parity._pruefe_rx12_vertrag(register, fehler)
    assert any("Nichtunterlegenheitsgrenzen" in f for f in fehler)

    report["objective"]["musik_mos_delta_by_title"] = [-0.2] * 10
    report["objective"]["musik_mos_delta_ci95"] = [0.01, 0.01]
    target.write_text(json.dumps(report), encoding="utf-8")
    fehler = []
    parity._pruefe_rx12_vertrag(register, fehler)
    assert any("Student-t-Neuberechnung" in f for f in fehler)
    assert any("Quellenpaaren" in f for f in fehler)
