"""D-K3-67 (2026-10-08): GIL-unabhängiger Phasen-Stack-Watchdog (faulthandler).

Befund (Lauf 10.12.17): §D-K3-55-Heartbeat 0× im Log, obwohl phase_12 30+ Min
lief; /proc-Sampling: exakt 1 von 81 Threads CPU-aktiv in 20 s — der Python-
Heartbeat-Thread kommt unter GIL-Blockade nicht zum Loggen. Fix: Der C-Level-
Watchdog der Stdlib (faulthandler.dump_traceback_later — offizieller Deadlock-
Debug-Mechanismus, dumpt ohne Python-GIL) schreibt ab Phase > Schwelle
periodisch ALLE Thread-Stacks nach logs/phase_watchdog_<pid>.log.

Kontrakte:
1. Funktional: enter → nach > Schwelle liegt ein Stack-Dump in der Datei;
   exit → cancel + Datei geschlossen.
2. Referenz-Zähler: parallele Phasen (enter 2×, exit 1×) halten den Watchdog.
3. Source-Pins: `_start_phase_heartbeat` aktiviert, der finally-Stop
   deaktiviert; ENV AURIK_PHASE_WATCHDOG_S.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

_UV3 = Path(__file__).resolve().parents[2] / "backend" / "core" / "unified_restorer_v3.py"


@pytest.fixture(autouse=True)
def _watchdog_state_sauber():
    """Kontrakte 1/2 messen die absolute Refcount-Semantik von enter/exit.

    Fremd-Leaks aus anderen Testdateien (unpaarte
    ``_start_phase_heartbeat``-Aufrufe ohne ``_phase_watchdog_exit``) dürfen
    diese Messung nicht verfälschen: vor und nach jedem Test auf refs=0
    aufräumen (der Exit ist defensiv: cancel + Datei-Close bei 0).
    """
    from backend.core import unified_restorer_v3 as uv3

    while int(uv3._PHASE_WATCHDOG_STATE.get("refs", 0) or 0) > 0:
        uv3._phase_watchdog_exit()
    yield
    while int(uv3._PHASE_WATCHDOG_STATE.get("refs", 0) or 0) > 0:
        uv3._phase_watchdog_exit()


def test_watchdog_dumpt_und_cancelt() -> None:
    """Kontrakt 1: enter → faulthandler-Stack-Dump nach der Schwelle; exit → cancel."""
    from backend.core import unified_restorer_v3 as uv3

    try:
        uv3._phase_watchdog_enter(silence_s=0.3)
        st = uv3._PHASE_WATCHDOG_STATE
        assert int(st["refs"]) == 1
        wd_path = Path(str(st["path"]))
        time.sleep(1.2)  # Watchdog-Thread dumpt bei repeat=True
        assert wd_path.exists(), "Watchdog-Datei wurde nicht angelegt"
        content = wd_path.read_text(encoding="utf-8", errors="replace")
        assert "Thread 0x" in content or "most recent call first" in content, (
            "kein Stack-Dump in der Watchdog-Datei — faulthandler wirkungslos"
        )
    finally:
        uv3._phase_watchdog_exit()
    assert int(uv3._PHASE_WATCHDOG_STATE["refs"]) == 0
    assert uv3._PHASE_WATCHDOG_STATE["fh"] is None, "Datei-Handle muss nach refs=0 geschlossen sein"


def test_parallele_phasen_referenz_zaehler() -> None:
    """Kontrakt 2: refs>0 hält den Watchdog offen (parallele Phasen)."""
    from backend.core import unified_restorer_v3 as uv3

    try:
        uv3._phase_watchdog_enter(silence_s=600.0)
        uv3._phase_watchdog_enter(silence_s=600.0)
        assert int(uv3._PHASE_WATCHDOG_STATE["refs"]) == 2
        uv3._phase_watchdog_exit()
        assert int(uv3._PHASE_WATCHDOG_STATE["refs"]) == 1
        assert uv3._PHASE_WATCHDOG_STATE["fh"] is not None, "Watchdog darf bei refs>0 nicht schließen"
    finally:
        uv3._phase_watchdog_exit()
    assert int(uv3._PHASE_WATCHDOG_STATE["refs"]) == 0


def test_source_pins_integration() -> None:
    """Kontrakt 3: Heartbeat-Start aktiviert, finally-Stop deaktiviert (Integration)."""
    src = _UV3.read_text(encoding="utf-8")
    assert "_phase_watchdog_enter()" in src
    assert "_phase_watchdog_exit()" in src
    assert "AURIK_PHASE_WATCHDOG_S" in src
    assert "dump_traceback_later(float(silence_s)" in src
    assert "cancel_dump_traceback_later" in src
