"""Vertragstests: `_panel_play` ruft existierende Wiedergabe-Logik auf (2026-11-19).

Produktreife-Befund (Paritäts-Register „Bedienung/Echtzeit“): `_panel_play`
(Transport-▶-Knopf) rief zwei nicht existierende Methoden auf
(`self._play_restored()` ohne fertige Restaurierung, `self._play_live_preview()`
während eines laufenden Batches) — ein Klick während/nach diesen Zuständen
löste eine unbehandelte `AttributeError` aus (§V6 (copilot-instructions.md)
Silent-Failure-Verbot: kein Log, keine Nutzer-Meldung). Die bereits
implementierte und verdrahtete `_play_restored_or_preview()` (§v10.999
Live-Mithören) deckt beide Fälle korrekt ab (fertiges Ergebnis ODER
Zwischenstand-Snapshot via Ring-Puffer).

Quelltext-Pins (Repo-Konvention für modern_window-Tests, kein QApplication nötig).
"""

from __future__ import annotations

from pathlib import Path

MODERN_WINDOW = Path(__file__).resolve().parents[2] / "Aurik10" / "ui" / "modern_window.py"


def _src() -> str:
    return MODERN_WINDOW.read_text(encoding="utf-8")


def _panel_play_body() -> str:
    quelle = _src()
    start = quelle.index("def _panel_play(self):")
    ende = quelle.index("def _panel_pause(self):", start)
    return quelle[start:ende]


def test_panel_play_delegiert_an_vorhandene_wiedergabe_logik() -> None:
    block = _panel_play_body()
    assert "self._play_restored_or_preview()" in block


def test_panel_play_ruft_keine_nicht_existierenden_methoden_mehr_auf() -> None:
    block = _panel_play_body()
    assert "self._play_restored()" not in block
    assert "self._play_live_preview()" not in block


def test_play_restored_or_preview_ist_tatsaechlich_definiert() -> None:
    """Regressionsschutz: die Delegations-Zielmethode muss existieren."""
    quelle = _src()
    assert "def _play_restored_or_preview(self) -> None:" in quelle


def test_keine_verwaisten_aufrufe_der_entfernten_methoden_im_gesamten_modul() -> None:
    """`_play_restored`/`_play_live_preview` dürfen nirgends mehr aufgerufen
    werden, da sie nie definiert waren (AttributeError-Risiko)."""
    quelle = _src()
    assert "def _play_restored(self)" not in quelle
    assert "def _play_live_preview(self)" not in quelle
    assert "self._play_live_preview()" not in quelle
