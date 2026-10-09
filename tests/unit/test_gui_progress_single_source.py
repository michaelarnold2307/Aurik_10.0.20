"""§WP-GUI-2026-10-09: Single-Source-/Zählwährungs-Pins für die Fortschrittsanzeige.

Befund (Tiefenanalyse 2026-10-09): Der Live-Defektzähler wurde von VIER
Schreibern mit unterschiedlichen Texten/Währungen bedient und die Restzähler
wurden von DREI proportionalen Überschreibern verfälscht (Severity×100-
Heuristik, Score-Ratio-Schätzung, Proporz-Propagation aus Chirurgie und
Prozessanteil). Die Anzeige sprang dadurch, zeigte wechselnde Semantiken und
zählte nicht monoton rückwärts.

Diese Tests pinnen die kanonische Ordnung (§G9 copilot-instructions.md: eine
Messung, eine Quelle) im Quelltext von `Aurik10/ui/modern_window.py`:

  1. Das Modell `DefectProgressModel` ist verdrahtet (Import + mindestens
     fünf Verwendungen: Init, Reset, Scan-Seeding, Engine-Exakt, Marker).
  2. Die falschen Zählwährungen sind ENTFERNT (keine Severity×100-Heuristik,
     keine Score-Ratio-Restschätzung, keine Proporz-Propagation).
  3. Alle Live-Label-Schreiber tragen den EINEN kanonischen Text; der alte
     Formatbruch „x/y behoben" existiert nicht mehr.
  4. Der Job-Scheduler-Deferral: Repair-Namen wandern in den Tooltip, sobald
     die Zählung läuft (kein Überschreiben des Countdowns je Phase-Hinweis).
"""

from __future__ import annotations

from pathlib import Path

MODERN_WINDOW = Path(__file__).resolve().parents[2] / "Aurik10" / "ui" / "modern_window.py"


def _src() -> str:
    return MODERN_WINDOW.read_text(encoding="utf-8")


def test_modul_syntaktisch_ok() -> None:
    """Grundwächter: Die Seite kompiliert (kein Import nötig — headless)."""
    import py_compile
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        py_compile.compile(str(MODERN_WINDOW), cfile=str(Path(td) / "mw.pyc"), doraise=True)


def test_defect_progress_model_verdrahtet() -> None:
    src = _src()
    assert "from Aurik10.ui.defect_progress_model import DefectProgressModel" in src
    assert src.count("_defect_progress_model") >= 5, "Modell nicht durchgehend verdrahtet"
    assert "defect_progress_model.set_initial" in src
    assert ".apply_exact(" in src
    assert ".apply_marker_resolved(" in src
    assert ".apply_chirurgie(" not in src, "Chirurgie darf den Hauptcounter nicht speisen"


def test_falsche_zaehlwaehrungen_entfernt() -> None:
    src = _src()
    assert "max(1, int(v * 100))" not in src, "Severity×100-Heuristik ist zurück"
    assert "_score_ratio = max(0.0, min(1.0, _v_cur / _v_init))" not in src, (
        "Score-Ratio-Restschätzung ist zurück (§WP-GUI-2026-10-09)"
    )
    assert "_rem_evt[_k] = max(0, int(_init_val * _ratio))" not in src, "Proporz-Propagation ist zurück"
    assert "_rem_counts[_dk] = _rem" not in src, "Chirurgie-Proporz ist zurück"
    assert "apply_resolved_event_counts as _arec" not in src, "Zweiter Subtraktionsweg ist zurück"


def test_kanonischer_labeltext_und_single_writer() -> None:
    src = _src()
    # Zählungstext (beide Live-Schreiber: _update_defects + _sync_unified_progress)
    assert "offen · {_done_def} behoben" in src
    assert "offen · {_def_resolved} behoben" in src
    # Alter Formatbruch entfernt:
    assert 'setText(f"{_def_resolved}/{_def_total} behoben")' not in src
    # Phase-Hinweise überschreiben den Countdown nicht mehr:
    assert 'setToolTip(f"🔧 {_repair_names}")' in src


def test_badge_liest_kanonischen_snapshot() -> None:
    src = _src()
    assert "_live_progress_snapshot" in src
    assert 'n_active = int(_prog_snap.get("remaining", n_active) or 0)' in src
