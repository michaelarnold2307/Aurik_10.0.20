"""§v10.720 (2026-09-08, Lücke-4 Stufe 1): Ganzsong-Modus — Unit-Gate-Tests.

Verifiziert die reine Chunk-vs-Ganzsong-Entscheidung
(`_should_use_chunked_path`) und die Verdrahtung des Feature-Flags
(`whole_song=True` / `AURIK_WHOLE_SONG=1`) im restore()-Gate.
Der Chunked-Pfad (RAM O(1)) bleibt Default und Fallback.
"""

from __future__ import annotations

from pathlib import Path

from backend.core.unified_restorer_v3 import _should_use_chunked_path

_SR = 48000


def test_short_audio_never_chunked() -> None:
    """≤120 s läuft immer direkt durch (auch ohne Ganzsong-Flag)."""
    assert not _should_use_chunked_path(60 * _SR, _SR, in_chunked=False, whole_song=False)


def test_long_audio_chunked_by_default() -> None:
    """>120 s ohne Flag → Chunked-Pfad (RAM O(1) bleibt Default)."""
    assert _should_use_chunked_path(121 * _SR, _SR, in_chunked=False, whole_song=False)


def test_whole_song_disables_chunking_for_long_audio() -> None:
    """>120 s mit Ganzsong-Flag → EIN Pass auf dem ganzen Song."""
    assert not _should_use_chunked_path(121 * _SR, _SR, in_chunked=False, whole_song=True)


def test_in_chunked_guard_blocks_recursion() -> None:
    """Im Chunk-Kontext wird nie erneut gechunked (Rekursions-Guard §v10.452)."""
    assert not _should_use_chunked_path(121 * _SR, _SR, in_chunked=True, whole_song=False)


def test_restore_gate_wires_env_flag_and_helper() -> None:
    """restore() nutzt den Helper UND den Env-Flag AURIK_WHOLE_SONG (Quellen-Vertrag)."""
    _src = (Path(__file__).resolve().parents[2] / "backend" / "core" / "unified_restorer_v3.py").read_text(
        encoding="utf-8"
    )
    assert 'os.environ.get("AURIK_WHOLE_SONG", "") == "1"' in _src
    assert "_should_use_chunked_path(" in _src
    assert 'kwargs.pop("whole_song", False)' in _src
