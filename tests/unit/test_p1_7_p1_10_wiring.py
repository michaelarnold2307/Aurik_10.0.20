"""§P1-7 + §P1-10 — Verkabelungs-Invarianten (headless, ohne Qt/ML).

Roadmap-Spezifikation (docs/TODOS_SOTA_ROADMAP.md):
- TODO-P1-7: `_run_m1b_targeted_retry` ist (1) im restore()-Tail nach dem
  Hörbarkeits-Gate und (2) in `_restore_chunked` nach der Song-Assembly
  (song-global, letzter Chunk-Scan) verkabelt; `m1b_retry_applied`/
  `m1b_retry_types` landen im Ergebnis-Metadata beider Pfade.
- TODO-P1-10: GUI-Chip-Zähler werden beim ersten Render aus dem aktiven
  Defekt-Set initialisiert (ein Chip je Typ); Behebungen kumulieren in
  `_resolved_defects_accumulated` und werden per `_done_filter` aus dem
  aktiven Set gefiltert; Backend setzt `_resolved_sent_keys` pro restore()
  zurück (§V8 (copilot-instructions.md), Song-Isolation).

Quelltext-Invarianten nach Repo-Konvention (siehe
test_gui_live_display_decimal_format.py) — ohne Qt-Import lauffähig. Die
Verhaltens-Tests liegen in test_m1b_targeted_retry.py bzw.
test_hearing_gates_summary.py.
"""

from __future__ import annotations

import pathlib
import re

_SRC_RESTORER = pathlib.Path("backend/core/unified_restorer_v3.py").read_text(encoding="utf-8")
_SRC_WINDOW = pathlib.Path("Aurik10/ui/modern_window.py").read_text(encoding="utf-8")


def _func_body(src: str, header: str) -> str:
    """Quelltext ab `header` bis zum nächsten 4-Einrückungs-`def` (Methoden-Grenze)."""
    start = src.index(header)
    rest = src[start + len(header) :]
    match = re.search(r"\n    def ", rest)
    end = start + len(header) + match.start() if match else len(src)
    return src[start:end]


_RESTORE = _func_body(_SRC_RESTORER, "def restore(")
_CHUNKED = _func_body(_SRC_RESTORER, "def _restore_chunked(")
_UPDATE_DEFECTS = _func_body(_SRC_WINDOW, "def _update_defects(self, defects):")


# ── §P1-7: m1b-Verkabelung (restore()-Tail + _restore_chunked) ─────────────


def test_m1b_exactly_two_call_sites_restore_tail_and_chunked() -> None:
    """§P1-7: Genau zwei Verkabelungen — restore()-Tail und _restore_chunked
    (kein dritter, inkonsistenter Pfad)."""
    assert _SRC_RESTORER.count("self._run_m1b_targeted_retry(") == 2
    assert "self._run_m1b_targeted_retry(" in _RESTORE
    assert "self._run_m1b_targeted_retry(" in _CHUNKED


def test_m1b_metadata_in_both_wirings() -> None:
    """§P1-7: m1b_retry_applied/m1b_retry_types im Ergebnis-Metadata BEIDER
    Verkabelungen (restore()-Tail + Chunked-Song-Result)."""
    assert 'result.metadata["m1b_retry_applied"] = True' in _RESTORE
    assert 'result.metadata["m1b_retry_types"] = list(_ag_report.improvable_types)' in _RESTORE
    assert '_meta["m1b_retry_applied"] = True' in _CHUNKED
    assert '_meta["m1b_retry_types"] = list(_m1b_chunked_types)' in _CHUNKED


def test_m1b_tail_only_after_failed_audibility_gate() -> None:
    """§P1-7 (1): Der restore()-Tail-Pass läuft nur bei verfehltem
    Hörbarkeits-Gate mit nachbehandlungswürdigen Typen — nie blind."""
    assert "not _chunked_tail_skip and not _ag_report.gate_passed and _ag_report.improvable_types" in _RESTORE


def test_m1b_chunked_song_global_uses_last_chunk_scan() -> None:
    """§P1-7 (2): _restore_chunked nutzt song-global die Per-Defekt-
    Reduktionsdaten des letzten Chunk-Scans (_defect_reduction_per_type)."""
    assert '_m1b_data = getattr(self, "_defect_reduction_per_type", None) or {}' in _CHUNKED
    assert "list(_m1b_rep.improvable_types)" in _CHUNKED


# ── §P1-10: Defekt-Chip-Rückwärtszählung ───────────────────────────────────


def test_chip_counters_initialized_once_at_first_render() -> None:
    """§P1-10 (1): Chip-Zähler werden beim ersten Render aus dem aktiven
    Defekt-Set initialisiert (ein Chip je Typ), nur wenn noch nicht gesetzt."""
    assert (
        'if _status == "correcting" and _active_defects_set and not getattr(self, "_defect_chip_counts", None):'
        in _UPDATE_DEFECTS
    )
    assert "self._defect_chip_counts = dict.fromkeys(sorted(_active_defects_set), 1)" in _UPDATE_DEFECTS
    assert "self._defect_chip_total = len(_active_defects_set)" in _UPDATE_DEFECTS


def test_done_filter_excludes_resolved_from_active_set() -> None:
    """§P1-10 (2): behobene Typen (_done_filter) gelten nicht mehr als aktiv —
    der aktive Defekt-Set wird bei Re-Renders nicht wieder aufgebaut."""
    assert '_done_filter = set(getattr(self, "_defect_chips_done", []))' in _UPDATE_DEFECTS
    assert "not in _done_filter" in _UPDATE_DEFECTS


def test_resolved_fixes_are_accumulated_across_renders() -> None:
    """§P1-10 (2): Behebungen kumulieren in _resolved_defects_accumulated
    (nur der letzte Diff ginge verloren); der Render wendet den kumulierten
    Satz an. Eine leere-Menge-Zuweisung existiert nur beim Neustart des
    Zählens — die Kumulation überschreibt nie."""
    assert _SRC_WINDOW.count("self._resolved_defects_accumulated = set()") == 1
    assert "_res_acc |= {str(_r) for _r in _resolved_now}" in _SRC_WINDOW
    assert _SRC_WINDOW.count("self._resolved_defects_accumulated = _res_acc") == 1
    assert '_resolved_live = sorted(getattr(self, "_resolved_defects_accumulated", set()))' in _UPDATE_DEFECTS


def test_new_scan_resets_counting_state() -> None:
    """§P1-10 / §V8 (copilot-instructions.md): ein neuer Scan (status=detected)
    setzt den Zähl-Zustand zurück — keine Leckage von Behebungen/Chips eines
    vorherigen Songs in den neuen Lauf."""
    assert 'if _status == "detected":' in _UPDATE_DEFECTS
    assert "self._resolved_defects_accumulated = set()" in _UPDATE_DEFECTS
    assert "self._defect_chip_counts = None" in _UPDATE_DEFECTS
    assert "self._defect_chip_total = 0" in _UPDATE_DEFECTS
    assert "self._defect_chips_done = []" in _UPDATE_DEFECTS


def test_batch_progress_applies_resolved_via_pure_helper() -> None:
    """§P1-10: _on_batch_progress rechnet Behebungen über die pure Logik
    apply_resolved_defects (Aurik10/ui/hearing_gates_summary.py) ab — eine
    Quelle der Wahrheit für die Chip-Rückwärtszählung."""
    assert "from Aurik10.ui.hearing_gates_summary import apply_resolved_defects" in _SRC_WINDOW
    assert "_cc, _ct, _cdone = _ard(_cc, _resolved_now)" in _SRC_WINDOW


def test_resolved_sent_keys_reset_per_restore() -> None:
    """§P1-10 (3) / §V8 (copilot-instructions.md): _resolved_sent_keys wird pro
    restore()-Aufruf zurückgesetzt (Song-Isolation); der Diff sendet nur neue
    Keys — sonst bliebe die Chip-Zählung in Folgesongs stumm."""
    assert "self._resolved_sent_keys: set[str] = set()" in _RESTORE
    assert "_res_new = sorted(t for t in _res_snap if t not in _res_sent)" in _RESTORE
