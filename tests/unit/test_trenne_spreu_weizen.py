"""Vertragstests für die Spreu/Weizen-Trennung (modell-frei).

Die harten Schwellen stammen aus dem Parity-Register (Regel 5(b), DRT-Schwellen)
und Spec 27 ANT-9 — sie dürfen NICHT abgeschwächt werden (DRT-5).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import scripts.trenne_spreu_weizen as tsw


def _beleg_basis(**over) -> dict:
    daten = {
        "klasse": "demo",
        "gate_bestanden": True,
        "gate_qualitaet_bestanden": True,
        "q_median": 1.05,
        "q_anteil_ge_0_5": 1.0,
        "never_worsen_anteil": 1.0,
        "musik_mos_delta_anteil_ge_minus_jnd": 1.0,
        "n_q_bewertbar": 90,
        "n_never_worsen_bewertbar": 169,
        "checkpoint": "models/klassen_koepfe/witness_demo_best.pt",
    }
    daten.update(over)
    return daten


def _pruefe(daten: dict) -> tuple[bool, list[str]]:
    return tsw._pruefe(daten)


def test_weizen_bei_allen_schwellen_erfuellt() -> None:
    bestanden, gruende = _pruefe(_beleg_basis())
    assert bestanden
    assert gruende == []


def test_spreu_wegen_q_median_unter_schwelle() -> None:
    bestanden, gruende = _pruefe(_beleg_basis(q_median=0.71))
    assert not bestanden
    assert any("q_median 0.71 < 0.95" in g for g in gruende)


def test_spreu_wegen_q_clip_symptom() -> None:
    bestanden, gruende = _pruefe(_beleg_basis(q_median=2.0))
    assert not bestanden
    assert any("q_clip" in g for g in gruende)


def test_spreu_wegen_mos_anteil_unter_1() -> None:
    bestanden, gruende = _pruefe(_beleg_basis(musik_mos_delta_anteil_ge_minus_jnd=0.8995))
    assert not bestanden
    assert any("mos_anteil" in g for g in gruende)


def test_spreu_wegen_q_anteil() -> None:
    bestanden, gruende = _pruefe(_beleg_basis(q_anteil_ge_0_5=0.63))
    assert not bestanden
    assert any("q_anteil" in g for g in gruende)


def test_spreu_wegen_never_worsen() -> None:
    bestanden, gruende = _pruefe(_beleg_basis(never_worsen_anteil=0.93))
    assert not bestanden
    assert any("never_worsen" in g for g in gruende)


def test_spreu_wegen_zu_wenig_bewertbare_paare() -> None:
    bestanden, gruende = _pruefe(_beleg_basis(n_q_bewertbar=4, n_never_worsen_bewertbar=80))
    assert not bestanden
    assert any("n_q_bewertbar" in g for g in gruende)


def test_spreu_ohne_mos_beleg() -> None:
    daten = _beleg_basis()
    daten.pop("musik_mos_delta_anteil_ge_minus_jnd")
    bestanden, gruende = _pruefe(daten)
    assert not bestanden
    assert any("mos_anteil fehlt" in g for g in gruende)


def test_gate_false_macht_spreu() -> None:
    bestanden, gruende = _pruefe(_beleg_basis(gate_bestanden=False))
    assert not bestanden
    assert any("gate_bestanden=false" in g for g in gruende)


def test_fail_closed_ohne_belege(tmp_path: Path) -> None:
    assert tsw.main(["--report-dir", str(tmp_path)]) == 1


def test_neuester_beleg_je_klasse_gewinnt(tmp_path: Path) -> None:
    alt = tmp_path / "2026-11-22_witness_kopf_gate_demo.json"
    neu = tmp_path / "2026-11-23_witness_kopf_gate_demo.json"
    alt.write_text(json.dumps({**_beleg_basis(), "q_median": 0.5}), encoding="utf-8")
    neu.write_text(json.dumps(_beleg_basis(q_median=1.1)), encoding="utf-8")
    belege = tsw._lade_letzte_belege(tmp_path)
    assert belege["demo"]["q_median"] == 1.1
    assert belege["demo"]["_pfad"].name == neu.name


def test_schema_und_deterministische_sortierung(tmp_path: Path) -> None:
    (tmp_path / "2026-11-23_witness_kopf_gate_b.json").write_text(
        json.dumps({**_beleg_basis(), "klasse": "b"}), encoding="utf-8"
    )
    (tmp_path / "2026-11-23_witness_kopf_gate_a.json").write_text(
        json.dumps({**_beleg_basis(), "klasse": "a", "q_median": 0.4}), encoding="utf-8"
    )
    out = tmp_path / "trennung.json"
    assert tsw.main(["--report-dir", str(tmp_path), "--out", str(out)]) == 0
    daten = json.loads(out.read_text(encoding="utf-8"))
    assert daten["schema"] == tsw._SCHEMA
    assert daten["weizen"] == ["b"]
    assert daten["spreu"] == {"a": daten["klassen"]["a"]["gruende"]}
