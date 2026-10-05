"""Regressionstest für den Shutdown-Pfad des MemmapPool (§G5 (GEBOTE.md), §V6 (VERBOTEN.md)).

Befund 2026-10-05: Nach einem Lauf mit aktivem ROCm-Pfad konnte ein Prozess nach der
Test-Summary („1 passed") in ``futex_do_wait`` hängen bzw. mit Exit 1 enden
(CLI/GUI-Paritätstest). Ursache: Der atexit-Pfad ``MemmapPool.close()`` nahm ein
reguläres ``threading.Lock``; am Interpreter-Ende beendet Python Daemon-Threads an
beliebiger Stelle — stirbt einer im kritischen Abschnitt, blockiert ``with
self._lock`` für immer.

Der Test belegt beide Invarianten des gehärteten Exit-Pfads: begrenztes Acquire
(≤ 3 s statt unbegrenzt) und Idempotenz. Läuft ohne torch/onnxruntime und ohne
Hardware (dependenz-arm, `tests.instructions.md`).
"""

from __future__ import annotations

import time

import pytest

from backend.core.memmap_pool import MemmapPool


@pytest.mark.unit
def test_close_bounded_when_lock_is_held() -> None:
    """Bei gehaltenem Lock muss close() begrenzt zurückkehren (kein Shutdown-Hang)."""
    pool = MemmapPool()
    pool._lock.acquire()  # simuliert: Daemon-Thread starb im kritischen Abschnitt
    try:
        t0 = time.time()
        pool.close()
        dauer = time.time() - t0
    finally:
        pool._lock.release()
    assert dauer < 3.0, f"close() blockierte {dauer:.1f} s bei gehaltenem Lock — Shutdown-Hang zurück"


@pytest.mark.unit
def test_close_is_idempotent() -> None:
    """Zweiter close()-Aufruf ist ein No-Op und kehrt sofort zurück."""
    pool = MemmapPool()
    pool.close()
    t0 = time.time()
    pool.close()
    assert time.time() - t0 < 1.0
