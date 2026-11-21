"""Vertragstests für scripts/bewerte_witness_q.py (Witness-Q-Erstbelege, 2026-11-14).

Modell-frei: Schema, fail-closed ohne Checkpoint, Q-Definition
((P_schaden − P_restauriert) / (P_schaden − P_clean), nur bei
Clean-Kontrast > 0,05), Determinismus der Stichprobe (§G5 (copilot-instructions.md)).
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import scripts.bewerte_witness_q as bw


def test_schema_pin() -> None:
    assert bw.SCHEMA == "aurik.witness_q/1"


def test_kontrast_schwelle() -> None:
    """Q wird nur bei Clean-Kontrast > 0,05 gebucht (fail-closed wie defekt_delta /4)."""
    assert bw.KONTRAST_MIN == 0.05


def test_q_definition() -> None:
    """Q = (P_schaden − P_restauriert) / (P_schaden − P_clean) — Anteil der
    zurückgewonnenen Schadens-Wahrscheinlichkeit."""
    p_s, p_c, p_r = 0.8, 0.1, 0.3  # Kontrast 0,7 > 0,05
    q = (p_s - p_r) / (p_s - p_c)
    assert q == pytest.approx(0.5 / 0.7)
    # vollständige Reparatur ⇒ Q = 1, keine Reparatur ⇒ Q = 0
    assert (p_s - p_c) / (p_s - p_c) == pytest.approx(1.0)
    assert (p_s - p_s) / (p_s - p_c) == pytest.approx(0.0)


def test_buche_fail_closed_ohne_regionen() -> None:
    ergebnis = bw.buche("groove_echo", [], random.Random(42))
    assert ergebnis["q_witness_median"] is None
    assert ergebnis["n_q_bewertbar"] == 0


def test_main_fail_closed_ohne_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fehlendes Witness-Checkpoint ⇒ Exit 2 vor jeder Messung."""
    monkeypatch.setattr(bw, "witness_kanal_verfuegbar", lambda k: False)
    assert bw.main(["--klassen", "groove_echo"]) == 2


def test_klassen_liste_stimmt_mit_checkpoint_pfaden_ueberein() -> None:
    """Die buchbaren Klassen sind exakt die mit kanonischem Checkpoint-Pfad."""
    from backend.core.dsp.witness_detektor import WITNESS_CHECKPOINT_PFADE

    assert set(WITNESS_CHECKPOINT_PFADE.keys()) == {
        "groove_echo",
        "sticky_shed_residue",
        "tape_head_level_dip",
        "sibilance",
        "compression_artifacts",
        "transport_bump",
        "clipping",
    }
