"""D-K3-63: FinalPolish/Dither darf im Chunked-Modus NICHT pro Chunk laufen.

Befund (Lauf 10.12.17, Chunk-0-Log): „§P1-8 FinalPolish (nach m1b)“ +
„§2.73c Noise-Shaped Dither: 16-bit TPDF“ erschienen im CHUNK-0-Durchlauf —
der §P1-8-Block lag ungeschützt vor dem `_chunked_tail_skip`-Guard, während
OneTakeExport (gleiche Fix-Familie 2026-09-17) längst geschützt war. Der
Chunk-Dither lief VOR Assembly/OneTakeExport und widersprach dem eigenen
§V5-Gebot („Dither = letzter Quantisierungsschritt des FINALEN Audio“).

Kontrakte (Source-Pins, Stil `tests/unit/test_p0_1_end_gate_chunking.py`):
1. restore(): FinalPolish-Aufruf liegt hinter `if not _chunked_tail_skip:`.
2. restore(): Abschluss-Narrativ („Restoration vollstaendig“ + Banner)
   liegt hinter `if not _chunked_tail_skip:`.
3. _restore_chunked: FinalPolish läuft EINMAL auf dem assemblierten Song,
   VOR dem OneTakeExport-Block.
4. Genau zwei FinalPolish-Importstellen (restore + Assembly).
"""

from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "backend" / "core" / "unified_restorer_v3.py"


def _src() -> str:
    return SRC.read_text(encoding="utf-8")


def test_final_polish_runs_behind_chunked_tail_skip() -> None:
    """Kontrakt 1: Kein Per-Chunk-FinalPolish (D-K3-63)."""
    src = _src()
    marker = (
        "if not _chunked_tail_skip:\n"
        "            try:\n"
        "                from backend.core.dsp.final_polish import apply_final_polish"
    )
    assert marker in src, "FinalPolish muss hinter dem _chunked_tail_skip-Guard liegen (D-K3-63)"


def test_abschluss_narrativ_guarded() -> None:
    """Kontrakt 2: „Restoration vollstaendig“ + Banner nur am Song-Ende."""
    src = _src()
    log_marker = 'if not _chunked_tail_skip:\n            logger.info(\n                "✅ Restoration vollstaendig'
    assert log_marker in src, "Abschluss-Log darf im Chunk-Pfad nicht laufen (D-K3-63)"
    banner_marker = "if not _chunked_tail_skip:\n            logger.info("
    assert banner_marker in src
    assert "log_restoration_summary(" in src, "Banner-Aufruf fehlt (Regression?)"


def test_song_level_final_polish_after_assembly() -> None:
    """Kontrakt 3: FinalPolish einmalig im Assembly-Block, VOR OneTakeExport."""
    src = _src()
    marker = "from backend.core.dsp.final_polish import apply_final_polish as _afp_song"
    assert marker in src, "Der Assembly-Block muss FinalPolish einmal ausführen (D-K3-63)"
    assert "§P1-8 FinalPolish (nach Assembly)" in src
    pos_polish = src.rindex(marker)
    pos_ote = src.index("from backend.core.one_take_export import OneTakeExport", pos_polish)
    assert pos_polish < pos_ote, "Reihenfolge: FinalPolish VOR OneTakeExport (non-chunked-Kette)"


def test_exactly_two_final_polish_import_sites() -> None:
    """Kontrakt 4: Keine dritte (verwaiste) FinalPolish-Importstelle."""
    src = _src()
    count = src.count("from backend.core.dsp.final_polish import apply_final_polish")
    assert count == 2, f"erwartet 2 Importstellen (restore + Assembly), gefunden {count}"
