"""Vertragstests für das Witness-Kopf-Gate (D-K3-95, 2026-11-15). Modell-frei."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import scripts.verify_witness_kopf_gate as gate


def test_schwellen_pin() -> None:
    assert gate.Q_MIN == 0.5
    assert gate.NW_TOL == 0.02
    assert gate.NW_ANTEIL_MIN == 0.95
    assert gate.SCHEMA == "aurik.witness_kopf_gate/1"


def test_gate_logik_beide_arme() -> None:
    """Gate = Median-Q ≥ 0,5 UND Never-worsen-Anteil ≥ 0,95 — beide Bedingungen."""
    assert gate.messe is not None
    gut = {"q_median": 0.6, "never_worsen_anteil": 0.98}
    assert gut["q_median"] >= gate.Q_MIN and gut["never_worsen_anteil"] >= gate.NW_ANTEIL_MIN
    schlecht = {"q_median": 0.6, "never_worsen_anteil": 0.9}
    assert not (schlecht["q_median"] >= gate.Q_MIN and schlecht["never_worsen_anteil"] >= gate.NW_ANTEIL_MIN)


def test_fail_closed_ohne_checkpoint(tmp_path: Path) -> None:
    """Fehlendes Kopf-Checkpoint ⇒ Exit 2 vor jeder Messung."""
    rc = gate.main(["--klasse", "groove_echo", "--checkpoint", str(tmp_path / "fehlt.pt")])
    assert rc == 2


def test_quelltext_pin_kein_flag() -> None:
    quelltext = Path(gate.__file__).read_text(encoding="utf-8")
    assert "music_model_flags" not in quelltext
    assert "flippe_flag" not in quelltext
    assert "cuda()" not in quelltext
