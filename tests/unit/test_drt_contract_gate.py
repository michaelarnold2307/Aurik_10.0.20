"""Vertragstests für das DRT-Gate (Spec 26, 2026-11-15). Modell-frei."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import scripts.compliance.drt_contract_gate as drt


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
    for regel in ["DRT-1", "DRT-2", "DRT-3", "DRT-4", "DRT-5", "DRT-6"]:
        assert regel in q
