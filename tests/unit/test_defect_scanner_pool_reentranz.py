"""Vertragstests für den Pool-Reentranz-Schutz (§PERF-R21e, 2026-11-16).

Pool-Race-Befund: Der Defekt-Scan-Pool (≤8) rief Detektor-Tasks, die intern
den Subband-FM-/Scrape-AM-Pool (≤6) nutzten — verschachtelte Pools ⇒
intermittierende Deadlocks (futex_do_wait) und race-beeinflusste
Pass-Through-Einstufungen. Der Schutz: Im Scan-Worker laufen die inneren
Kerne seriell (threading.local-Flag, bit-identisch, §G5 (copilot-instructions.md)). Modell-frei.
"""

from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[2] / "backend" / "core" / "defect_scanner.py"


def _text() -> str:
    return _SRC.read_text(encoding="utf-8")


def test_reentranz_flag_und_setter_vorhanden() -> None:
    """Flag + Setter existieren und sind verdrahtet."""
    src = _text()
    assert "_SCAN_POOL_AKTIV = threading.local()" in src
    assert "def _set_pool_flag(wert: bool) -> None:" in src
    assert "_SCAN_POOL_AKTIV.flag = wert" in src


def test_scan_pool_setzt_flag_um_den_task() -> None:
    """Der Scan-Pool-Task setzt das Flag VOR dem Detektor und räumt es im finally auf."""
    src = _text()
    assert "_set_pool_flag(True)" in src
    assert "finally:" in src
    assert "_set_pool_flag(False)" in src
    assert "_det_pool.map(_det_task, _det_tasks)" in src


def test_innerer_pool_seriell_im_scan_worker() -> None:
    """FM- UND AM-Pool schalten im Scan-Worker auf 1 Worker (kein Pool im Pool)."""
    src = _text()
    assert 'if getattr(_SCAN_POOL_AKTIV, "flag", False):' in src
    assert src.count('if getattr(_SCAN_POOL_AKTIV, "flag", False):') >= 2
    assert "_fm_workers = 1" in src
    assert "_am_workers = 1" in src


def test_flag_verhalten_thread_lokal() -> None:
    """Der Setter schreibt nur in den aufrufenden Thread (threading.local)."""
    import importlib
    import threading

    scanner = importlib.import_module("backend.core.defect_scanner")
    assert not getattr(scanner._SCAN_POOL_AKTIV, "flag", False)
    scanner._set_pool_flag(True)
    assert scanner._SCAN_POOL_AKTIV.flag is True
    # anderer Thread sieht das Flag NICHT
    ergebnis: list[bool] = []

    def _leser() -> None:
        ergebnis.append(getattr(scanner._SCAN_POOL_AKTIV, "flag", False))

    t = threading.Thread(target=_leser)
    t.start()
    t.join()
    assert ergebnis == [False]
    scanner._set_pool_flag(False)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
