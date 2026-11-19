"""Vertragstests: Fenstergeometrie-Schutz im ModernMainWindow (2026-11-19).

GUI-Reifegrad: eine gespeicherte Fensterposition außerhalb aller Bildschirme
(Monitor abgezogen, Auflösungswechsel) ließ das Fenster nach dem Start
unsichtbar erscheinen. Der Guard zentriert dann neu statt zu restaurieren.

Quelltext-Pins (Repo-Konvention für modern_window-Tests, kein QApplication nötig):
  1. `_geometry_sichtbar` existiert und prüft Schnitt mit QApplication.screens().
  2. Der Restore-Pfad ruft den Guard auf und behandelt den Unsichtbar-Fall
     (Warnung + Neuzentrierung statt Restore).
  3. Fail-open: ohne Screen-Liste oder bei Fehlern wird nicht erzwungen.
"""

from __future__ import annotations

from pathlib import Path

MODERN_WINDOW = Path(__file__).resolve().parents[2] / "Aurik10" / "ui" / "modern_window.py"


def _src() -> str:
    return MODERN_WINDOW.read_text(encoding="utf-8")


def test_geometry_guard_existiert() -> None:
    quelle = _src()
    assert "def _geometry_sichtbar(self) -> bool:" in quelle
    assert "any(_frame.intersects(_s.geometry()) for _s in _screens)" in quelle


def test_restore_pfad_nutzt_guard_und_zentriert_neu() -> None:
    quelle = _src()
    assert "if self._geometry_sichtbar():" in quelle
    assert "Gespeicherte Fenstergeometrie liegt außerhalb der Bildschirme" in quelle
    # Reihenfolge: restoreGeometry → Guard-Entscheidung
    assert quelle.index("self.restoreGeometry(_geo)") < quelle.index("if self._geometry_sichtbar():")


def test_guard_fail_open_ohne_screens() -> None:
    """Kein Screen-Layout bekannt ⇒ nichts erzwingen (kein falscher Re-Center)."""
    quelle = _src()
    block = quelle.split("def _geometry_sichtbar(self) -> bool:", 1)[1].split("def _animate_fade_in", 1)[0]
    assert "return True  # Kein Screen-Layout bekannt" in block
    assert "except Exception:" in block
