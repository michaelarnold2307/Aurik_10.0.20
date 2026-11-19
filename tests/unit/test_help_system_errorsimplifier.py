"""§GUI-T3 — help_system.ErrorSimplifier: Laien-Fehlertexte (headless).

Die Fehler→Laien-Meldung-Zuordnung ist reine Logik; sie entscheidet, was
Nutzer bei jedem Fehler sehen. Seit dem i18n-Gap-Fix (2026-09-16) sind alle
help.error.*-Keys übersetzt — die Tests assertieren gegen die ÜBERSETZTEN
Texte (Standardsprache de), nie gegen rohe Keys.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt5")  # CI-Minimal-Umgebung (cross-platform)

from Aurik10.i18n import set_language
from Aurik10.ui.help_system import ErrorSimplifier

set_language("de")


def test_memory_patterns() -> None:
    assert (
        ErrorSimplifier.simplify(MemoryError("x"))
        == "Nicht genügend Arbeitsspeicher — bitte andere Anwendungen schließen und erneut versuchen."
    )
    assert (
        ErrorSimplifier.simplify("cannot allocate memory")
        == "Nicht genügend Arbeitsspeicher — bitte andere Anwendungen schließen und erneut versuchen."
    )


def test_priority_memory_before_gpu() -> None:
    """'CUDA out of memory' ist zuerst ein Speicherfehler (Reihenfolge!)."""
    assert (
        ErrorSimplifier.simplify("CUDA out of memory")
        == "Nicht genügend Arbeitsspeicher — bitte andere Anwendungen schließen und erneut versuchen."
    )


def test_file_and_permission() -> None:
    assert (
        ErrorSimplifier.simplify(FileNotFoundError("No such file: x"))
        == "Die Datei wurde nicht gefunden — bitte den Pfad prüfen."
    )
    assert ErrorSimplifier.simplify("Permission denied") == "Zugriff verweigert — bitte die Berechtigungen prüfen."


def test_gpu_pattern() -> None:
    assert (
        ErrorSimplifier.simplify("CUDA error: illegal memory access")
        == "Grafikkarten-Fehler — Aurik verarbeitet auf der CPU weiter."
    )


def test_generic_error_fallback() -> None:
    out = ErrorSimplifier.simplify("WeirdError: something broke")
    assert "help.error.generic" not in out  # kein roher Key (§GUI-T3-i18n)
    assert "schiefgelaufen" in out


def test_non_error_passthrough() -> None:
    assert ErrorSimplifier.simplify("nur ein Hinweis") == "nur ein Hinweis"


def test_get_all_messages_nonempty() -> None:
    msgs = ErrorSimplifier.get_all_messages()
    assert isinstance(msgs, dict) and len(msgs) >= 15  # 16 eindeutige Keys (dedupliziert)
    assert all(isinstance(v, str) and v for v in msgs.values())
    assert "help.error.memory" in msgs
    assert "help.error" not in msgs["help.error.memory"]  # übersetzt, kein roher Key
