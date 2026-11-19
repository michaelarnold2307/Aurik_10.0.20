"""Vertragstest Frontend-Prosa-Folgewelle (D-K3-107, 2026-11-17).

Fail-closed nach oben: Ein NEUER nutzersichtbarer Prosa-Kandidat ohne ``t()``
— außerhalb der dokumentierten ``EXEMPT``-Tabelle des Scanners — lässt den
Test fallen. Die Tabelle enthält ausschließlich technische Nicht-Texte
(Farben, Fonts, Fehler-Match-Muster, Key-Präfix-Builder, Logger-Berichte).
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from scripts.frontend_prosa_scanner import EXEMPT, scan

_PROJEKT = pathlib.Path(__file__).resolve().parents[2]


def test_keine_neuen_prosa_kandidaten_ausserhalb_exempt() -> None:
    """Null neue Nutzertext-Kandidaten — der 47er- und Folgewelle-Scope ist zu."""
    kandidaten, exempt = scan(_PROJEKT / "Aurik10" / "ui")

    assert kandidaten == [], f"Neue Prosa ohne t(): {kandidaten[:5]}"
    assert exempt, "die dokumentierten technischen Ausnahmen müssen existieren"


def test_exempt_tabelle_ist_vollstaendig_dokumentiert() -> None:
    """Jede Ausnahme braucht Datei-Muster, genau einen Match-Typ und Grund."""
    erlaubt = {"datei"}, {"zeilen", "text_gleich", "text_prefix"}
    for eintrag in EXEMPT:
        assert "datei" in eintrag
        _match = {k for k in eintrag if k in ("zeilen", "text_gleich", "text_prefix")}
        assert len(_match) == 1, eintrag
        assert isinstance(eintrag.get("grund"), str) and eintrag["grund"], eintrag
