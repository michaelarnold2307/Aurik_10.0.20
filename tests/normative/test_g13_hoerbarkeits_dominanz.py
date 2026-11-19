#!/usr/bin/env python3
"""[RELEASE_MUST] Hoerbarkeits-Dominanz ueber alle 65 Defektklassen (§G13 (copilot-instructions.md))

Normativer CI-Gate-Test: §G13 (copilot-instructions.md) verpflichtet jede der
65 Defectklassen auf das Ziel „Rest unter Maskierungsschwelle − 6 dB Marge"
mit Reparaturquote Q ≥ 0,5 (defekt_delta), Kontrollregionen Δ ≈ 0 und
Musik-Schutz (Frage B + Hör-Invarianten). Never-worsen bleibt der Boden, die
Verbesserung ist die Pflicht. Diese Tests pinnen Norm-Wortlaut und die
kanonischen Träger — hoerbarkeits-dominanz ist der normative Begriff für die
Verbesserungs-Pflicht über alle defektklassen.

Zitierdisziplin AGENTS.md §5.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_release_must_header_existiert() -> None:
    text = (ROOT / ".github" / "copilot-instructions.md").read_text(encoding="utf-8")
    assert "## [RELEASE_MUST] Hoerbarkeits-Dominanz ueber alle 65 Defektklassen" in text, (
        "RELEASE_MUST-Header fehlt in copilot-instructions.md"
    )
    assert "Kategorie I: Hörbarkeits-Dominanz (Verbesserungs-Pflicht über alle Defektklassen)" in text
    assert "Reparaturquote **Q ≥ 0,5**" in text
    assert "Maskierungsschwelle − **6 dB Marge**" in text
    assert "Seed 42" in text and "pegel-gematcht (Body-Perzentil p25" in text


def test_hoerordnung_reparier_auftrag() -> None:
    text = (ROOT / ".github" / "instructions" / "hoerordnung.instructions.md").read_text(encoding="utf-8")
    assert "**Reparatur-Auftrag (§G13 (copilot-instructions.md), Verbesserungs-Pflicht):**" in text
    assert "Never-worsen ist der Boden, nicht das Ziel" in text
    assert "Q ≥ 0,5" in text and "6 dB Marge" in text


def test_kanonische_konstanten_und_traeger() -> None:
    from backend.core.dsp.audibility_targets import (
        repair_objective_met,
        residual_audibility,
        verify_declick_repair,
    )
    from backend.core.dsp.defekt_delta import REPARATURQUOTE_ZIEL, berechne_defekt_delta

    assert REPARATURQUOTE_ZIEL == 0.5, "§G13-Zielwert: Q >= 0.5 (eine Quelle)"
    assert callable(berechne_defekt_delta)
    assert callable(residual_audibility)
    assert callable(repair_objective_met)
    assert callable(verify_declick_repair)


def test_coverage_gate_65_65() -> None:
    """Jede der 65 DefectTypes trägt Harness-Fall + Phase-Mapping (fail-closed)."""
    ergebnis = subprocess.run(
        [sys.executable, "-B", str(ROOT / "scripts" / "defect_coverage_check.py")],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=300,
    )
    assert ergebnis.returncode == 0, (
        f"defect_coverage_check fehlgeschlagen (Exit {ergebnis.returncode}):\n"
        f"{ergebnis.stdout[-800:]}{ergebnis.stderr[-800:]}"
    )


def test_gebote_verifier_g13_gruen() -> None:
    """Der Verifier erzwingt §G13-Träger (check_g13) — fail-closed im CI."""
    ergebnis = subprocess.run(
        [sys.executable, "-B", str(ROOT / "scripts" / "gebote_verifier.py"), "--ci"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=300,
    )
    assert ergebnis.returncode == 0, (
        f"gebote_verifier --ci fehlgeschlagen (Exit {ergebnis.returncode}):\n"
        f"{ergebnis.stdout[-800:]}{ergebnis.stderr[-800:]}"
    )
