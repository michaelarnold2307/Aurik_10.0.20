"""Vertragstests: Genre-Selbst-Labeling (scripts/erstelle_genre_labels.py, 2026-11-20).

Modell-frei: der Ausschnitt ist deterministisch (Mitte, Seed-frei), die Labels
sind MASCHINELL deklariert (kein §4-Ersatz), das Schema ist gepinnt, und der
Lauf ist CPU-only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts import erstelle_genre_labels as g

_ROOT = Path(__file__).resolve().parents[2]


def test_mittel_ausschnitt_deterministisch() -> None:
    x = np.arange(48000 * 10, dtype=np.float32)
    a = g._mittel_ausschnitt(x, 2.0, 48000)
    b = g._mittel_ausschnitt(x, 2.0, 48000)
    assert a.shape[-1] == 96000 and np.array_equal(a, b)
    assert a[0] == x[(x.shape[-1] - 96000) // 2]  # Mitte, nicht Anfang


def test_schema_und_maschinell_deklariert() -> None:
    quelle = (_ROOT / "scripts" / "erstelle_genre_labels.py").read_text(encoding="utf-8")
    assert g.SCHEMA == "aurik.genre_labels/1"
    assert '"methode": "maschinell' in quelle  # kein §4-Ersatz, ehrlich deklariert
    assert "genre_classifier" in quelle
    assert "genre_labels.jsonl" in quelle


def test_fehlertoleranz_pro_titel() -> None:
    """Ein fehlschlagender Titel bricht den Lauf nicht ab (fehler-Eintrag statt Crash)."""
    quelle = (_ROOT / "scripts" / "erstelle_genre_labels.py").read_text(encoding="utf-8")
    assert '"fehler": str(exc)' in quelle
    assert "except Exception as exc:" in quelle
