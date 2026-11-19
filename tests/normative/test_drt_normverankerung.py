"""Normativer Pin: Audit-2026-11-15-Verschärfungen sind in der Kette verankert.

Die drei Regeln, die aus den Befunden vom 2026-11-15 folgen, müssen in den
normativen Dokumenten stehen — sonst wäre die DRT-Spec 26 ohne Domänen-Verankerung.
"""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _lies(rel: str) -> str:
    return (_ROOT / rel).read_text(encoding="utf-8")


def test_dsp_repräsentations_paritaet_verankert() -> None:
    q = _lies(".github/instructions/dsp.instructions.md")
    assert "Repräsentations-Parität gelernter Modelle" in q
    assert "slaney" in q and "10·log10" in q and "amin" in q
    assert "Produktions-Paritäts-Test" in q


def test_goals_ganztitel_zwei_metrik_verankert() -> None:
    q = _lies(".github/instructions/musical_goals.instructions.md")
    assert "Ganz-Titel-Zwei-Metrik-Never-worsen" in q
    assert "ÜBERRESTAURATION" in q
    assert "Kontroll-Bedingung" in q


def test_hoerordnung_zwei_ebenen_pflicht_verankert() -> None:
    q = _lies(".github/instructions/hoerordnung.instructions.md")
    assert "Zwei-Ebenen-Zeugen-Pflicht" in q
    assert "nicht entscheidungsreif" in q


def test_querverweise_zu_spec26_stimmen() -> None:
    spec = _lies(".github/specs/26_defekt_reparatur_trainingsvertrag.md")
    for regel in [f"DRT-{n}" for n in range(1, 13)]:
        assert regel in spec, f"{regel} fehlt in Spec 26"
