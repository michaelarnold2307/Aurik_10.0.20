#!/usr/bin/env python3
"""Vertragstests fuer scripts/bewerte_in_sample_q.py (Pre-Flight-Gate).

Fahrplan B-0(3) (`docs/reports/current/2026-11-13_aktivierungs_fahrplan.md`):
Vor jeder Holdout-Bewertung muss die In-Sample-Q auf dem Train-Split
(300 deterministische Regionen) >= 0,5 sein. Gepinnt werden die
fail-closed-Pfade (§V6 (copilot-instructions.md)) und die Gate-Semantik.
Zitierdisziplin AGENTS.md §5.
"""

from __future__ import annotations

import pytest

import scripts.bewerte_in_sample_q as isq


def test_fail_closed_manifest_fehlt(tmp_path) -> None:
    assert isq.main(["--klasse", "x", "--manifest", str(tmp_path / "fehlt.jsonl"), "--checkpoint", "c.pt"]) == 2


def test_fail_closed_checkpoint_fehlt(tmp_path) -> None:
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(
        '{"klasse": "groove_echo", "clean": "c.wav", "schaden": "s.wav", "sha256_pcm_schaden": "a"}\n',
        encoding="utf-8",
    )
    assert (
        isq.main(
            [
                "--klasse",
                "groove_echo",
                "--manifest",
                str(manifest),
                "--checkpoint",
                str(tmp_path / "fehlt.pt"),
            ]
        )
        == 2
    )


def test_preflight_zielkonstante() -> None:
    from backend.core.dsp.defekt_delta import REPARATURQUOTE_ZIEL

    assert isq.REPARATURQUOTE_ZIEL == 0.5
    assert REPARATURQUOTE_ZIEL == 0.5
    assert isq._REGIONEN_PREFLIGHT == 300


def test_gate_entscheidung_semantik() -> None:
    """Pre-Flight-Gate: Q >= 0,5 UND mindestens 6 Q-fähige Paare — sonst Ansatz verwerfen."""
    assert isq._gate_entscheidung(0.5, 6)
    assert isq._gate_entscheidung(0.6, 300)
    assert not isq._gate_entscheidung(0.49, 300)  # Q knapp unter Ziel
    assert not isq._gate_entscheidung(0.8, 5)  # zu wenige Paare
    assert not isq._gate_entscheidung(None, 0)  # keine Q-fähigen Regionen


def test_pfadfehler_sind_vom_gate_unterscheidbar(tmp_path) -> None:
    """Exit 2 = Pfad-/Datenfehler (fail-closed), nie mit Gate-Ausgang 0/1 verwechselbar."""
    assert isq.main(["--klasse", "x", "--manifest", str(tmp_path / "fehlt.jsonl"), "--checkpoint", "c.pt"]) == 2
