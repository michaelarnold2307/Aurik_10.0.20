"""§D-K3-55 (2026-10-07): Phasen-Heartbeat — Log-Zeile jede Minute (§G8 (copilot-instructions.md)).

Produktionsbefund „Trio Schweizer – 13 Tage" (10.12.12): 123 min Log-Stillstand
in einer Langphase bei 100–660 % CPU — OOM_PROBE loggt nur an Phasen-Grenzen,
und der damalige Heartbeat lief NUR mit UI-Callback und loggte NIE. Fix: Der
Heartbeat-Thread läuft immer (ohne Callback nur Log), die Log-Zeile kommt je
Intervall (Default 60 s, `AURIK_PHASE_HB_LOG_S` überschreibbar für Tests/OP),
und der Stop erfolgt garantiert via Event (§G173) + join.

Gepinnt: (1) Heartbeat loggt ohne UI-Callback; (2) Progress-Callbacks kommen
weiter und bleiben im 0–100-Bereich; (3) Log-Format enthält Phasenname und
Minuten; (4) Stop hinterlässt keinen lebenden Thread.
"""

from __future__ import annotations

import logging
import time

from backend.core.unified_restorer_v3 import UnifiedRestorerV3

_LOGGER = "backend.core.unified_restorer_v3"


def _restorer_lite() -> UnifiedRestorerV3:
    """Instanz ohne __init__ — die Heartbeat-Methoden brauchen keine Zustände."""
    return object.__new__(UnifiedRestorerV3)


def test_heartbeat_loggt_ohne_ui_callback(caplog) -> None:
    rst = _restorer_lite()
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        stop, th = rst._start_phase_heartbeat("phase_test_hb", progress_cb=None, log_every_s=0.05)
        assert th.daemon
        time.sleep(0.28)
        stop.set()
        th.join(timeout=2.0)
        assert not th.is_alive()
    msgs = [r.message for r in caplog.records if "Phasen-Heartbeat" in r.message]
    assert len(msgs) >= 2, f"Langphasen-Transparenz fehlt: {msgs}"
    assert all("phase_test_hb" in m for m in msgs)


def test_heartbeat_callbacks_bleiben_erhalten() -> None:
    rst = _restorer_lite()
    calls: list[float] = []
    stop, th = rst._start_phase_heartbeat(
        "phase_test_hb2",
        progress_cb=lambda pct, _msg, _t: calls.append(float(pct)),
        log_every_s=10.0,
    )
    time.sleep(0.3)
    stop.set()
    th.join(timeout=2.0)
    assert calls, "Sub-Progress-Callbacks fehlen"
    assert all(0.0 <= c <= 100.0 for c in calls)


def test_log_step_format_enthaelt_minuten(caplog) -> None:
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        UnifiedRestorerV3._phase_heartbeat_log_step("phase_12_wow_flutter_fix", 180.0)
    assert "Phasen-Heartbeat phase_12_wow_flutter_fix" in caplog.text
    assert "3 min aktiv" in caplog.text
